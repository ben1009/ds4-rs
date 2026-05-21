//! Speculative decoding with the MTP draft model.
//!
//! The algorithm:
//! 1. Main model produces logits → greedy token X
//! 2. MTP drafts token Y from the same hidden state
//! 3. If X ≠ Y, just output X (no speculation benefit)
//! 4. If X = Y, draft additional tokens via MTP
//! 5. Verify drafts by evaluating them through the main model
//! 6. Accept the consecutive prefix that the main model agrees with

use anyhow::Result;

use crate::{
    engine::Engine,
    mtp::{MtpState, MtpWeights, mtp_forward},
    session::Session,
};

/// Configuration for speculative decoding.
pub struct SpecConfig {
    /// Maximum number of draft tokens per step (default 1, max 16).
    pub mtp_draft_tokens: usize,
    /// Top-2 logit margin threshold. If the main model's confidence is below
    /// this, skip speculation (default 3.0).
    pub mtp_margin: f32,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
        }
    }
}

/// Perform one speculative decoding step.
///
/// Returns the accepted tokens (always at least 1). The session state
/// (tokens, pos, logits, KV cache) is updated to reflect all accepted tokens.
/// The caller should read `session.logits()` for the next-token prediction
/// after this returns.
pub fn generate_speculative(
    session: &mut Session,
    engine: &Engine,
    spec_config: &SpecConfig,
) -> Result<Vec<u32>> {
    let main_token = Session::argmax(session.logits())
        .ok_or_else(|| anyhow::anyhow!("speculative: no logits"))?;

    // Check MTP availability.
    if engine.mtp_weights.is_none() || session.mtp_state.is_none() {
        session.eval_token(main_token)?;
        return Ok(vec![main_token]);
    }

    // Confidence gate: skip speculation if the main model is not confident.
    if !confidence_check(session.logits(), spec_config.mtp_margin) {
        session.eval_token(main_token)?;
        return Ok(vec![main_token]);
    }

    // === Drafting phase ===
    // We need to borrow session.mtp_state mutably for mtp_forward, but we
    // can't hold that borrow across session.eval_token calls. So we do all
    // drafting first, collecting draft tokens and hidden snapshots, then
    // release the borrow before verification.

    let main_hidden = session.last_hidden().to_vec();
    let pos = session.pos();

    let drafts_result = {
        let mtp_weight_map = engine.mtp_weights.as_ref().unwrap();
        let mtp_weights = MtpWeights::from_map(mtp_weight_map)?;
        let mtp_state = session.mtp_state.as_mut().unwrap();

        draft_tokens(
            mtp_state,
            &mtp_weights,
            &engine.weights,
            engine,
            main_token,
            pos,
            &main_hidden,
            spec_config.mtp_draft_tokens,
        )?
    };
    // mtp_state borrow is released here.

    let (drafts, hidden_snapshots) = drafts_result;

    // If MTP disagrees with the main model on draft[0], no speculation benefit.
    // draft_tokens returns empty when draft[0] != main_token.
    if drafts.is_empty() {
        session.eval_token(main_token)?;
        return Ok(vec![main_token]);
    }

    // === Verification phase ===
    // Snapshot the KV cache before verification. We use eval_token_no_snapshot
    // during verification so that eval_token_inner does not overwrite this
    // snapshot with per-token rollback state.
    session.snapshot_kv();
    let tokens_before = session.tokens().len();

    // Evaluate main_token (always accepted).
    session.eval_token_no_snapshot(main_token)?;
    let mut accepted = vec![main_token];

    // Verify each subsequent draft token.
    for &draft in drafts.iter().skip(1) {
        let main_pred = Session::argmax(session.logits())
            .ok_or_else(|| anyhow::anyhow!("speculative: empty logits during verification"))?;
        if main_pred != draft {
            break;
        }
        session.eval_token_no_snapshot(draft)?;
        accepted.push(draft);
    }

    // === Rollback if needed ===
    let n_accepted = accepted.len();
    let total_drafts = drafts.len();

    // If we didn't accept all drafts, restore the pre-verification state and
    // re-evaluate only the accepted tokens. We must truncate tokens back to
    // the pre-verification length (eval_token pushes tokens) and restore the
    // KV cache from our snapshot.
    if n_accepted < total_drafts {
        session.restore_kv();
        // Truncate tokens back to pre-verification length. This also resets
        // pos via the tokens.len() calculation in eval_token_inner.
        session.truncate_tokens(tokens_before);
        for &tok in &accepted {
            session.eval_token(tok)?;
        }
    }

    // === MTP state alignment ===
    {
        let mtp_state = session.mtp_state.as_mut().unwrap();

        // Restore prev_hidden to the state after the last accepted draft.
        // hidden_snapshots[i] is MTP state after drafts[i]. The last accepted
        // draft is drafts[n_accepted - 1], so we need hidden_snapshots[n_accepted - 1].
        if n_accepted <= hidden_snapshots.len() {
            mtp_state
                .prev_hidden
                .clone_from(&hidden_snapshots[n_accepted - 1]);
        }

        // Pop rejected entries from MTP KV cache.
        // Each draft (including draft[0]) runs mtp_forward which pushes one KV
        // entry. Accepted MTP KV entries = n_accepted (draft[0]=main_token +
        // n_accepted - 1 subsequent). Total MTP KV entries = total_drafts.
        let rejected = total_drafts.saturating_sub(n_accepted);
        for _ in 0..rejected {
            mtp_state.kv_cache.pop_last_row();
        }
    }

    Ok(accepted)
}

/// Draft tokens using the MTP model. Returns `(drafts, hidden_snapshots)`.
///
/// `drafts[0]` is the MTP prediction using the main model's hidden state.
/// Subsequent drafts use MTP's own `prev_hidden`.
///
/// `hidden_snapshots[i]` is MTP's `prev_hidden` after generating `drafts[i]`.
#[allow(clippy::too_many_arguments)]
fn draft_tokens(
    mtp_state: &mut MtpState,
    mtp_weights: &MtpWeights<'_>,
    main_weights: &crate::model::WeightMap,
    engine: &Engine,
    main_token: u32,
    pos: u32,
    main_hidden: &[f32],
    max_drafts: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let eos = engine.tokenizer.eos_token();
    let max_drafts = max_drafts.min(16);

    // Draft[0]: MTP predicts using the main model's hidden state.
    let draft0 = mtp_forward(
        mtp_state,
        mtp_weights,
        main_weights,
        engine,
        main_token,
        pos,
        main_hidden,
    )?;

    // If draft[0] != main_token, return empty to signal "no speculation".
    if draft0 != main_token {
        return Ok((vec![], vec![]));
    }

    let mut drafts = vec![draft0];
    let mut hidden_snapshots = vec![mtp_state.prev_hidden.clone()];

    // Draft[1..N] using MTP's own prev_hidden.
    for i in 1..max_drafts {
        let prev_token = *drafts.last().unwrap();
        if prev_token == eos {
            break;
        }
        let draft_pos = pos + i as u32;
        let prev_hidden = mtp_state.prev_hidden.clone();

        let draft = mtp_forward(
            mtp_state,
            mtp_weights,
            main_weights,
            engine,
            prev_token,
            draft_pos,
            &prev_hidden,
        )?;

        drafts.push(draft);
        hidden_snapshots.push(mtp_state.prev_hidden.clone());

        if draft == eos {
            break;
        }
    }

    Ok((drafts, hidden_snapshots))
}

/// Check if the main model is confident enough for speculation.
///
/// Returns true if the difference between the top-2 logit values is >= margin.
fn confidence_check(logits: &[f32], margin: f32) -> bool {
    if logits.len() < 2 {
        return false;
    }
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &v in logits {
        if v > top1 {
            top2 = top1;
            top1 = v;
        } else if v > top2 {
            top2 = v;
        }
    }
    (top1 - top2) >= margin
}
