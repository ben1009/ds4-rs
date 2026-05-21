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
    // Take ownership of mtp_state to avoid borrowing session mutably and
    // immutably at the same time. We still need .to_vec() for last_hidden
    // because mtp_state is passed mutably to draft_tokens while main_hidden
    // (from session.last_hidden()) must live across the call. The allocation
    // is bounded by n_embd (typically 16KB).
    let mut mtp_state = session.mtp_state.take().unwrap();
    let main_hidden = session.last_hidden().to_vec();
    let pos = session.pos();

    // Parse MTP weights outside the mutable borrow of mtp_state.
    let mtp_weight_map = engine.mtp_weights.as_ref().unwrap();
    let mtp_weights = MtpWeights::from_map(mtp_weight_map)?;

    let drafts = {
        let s_prev_hidden = &mut session.s_prev_hidden;

        draft_tokens(
            &mut mtp_state,
            &mtp_weights,
            &engine.weights,
            engine,
            main_token,
            pos,
            &main_hidden,
            spec_config.mtp_draft_tokens,
            s_prev_hidden,
        )?
    };
    // Restore mtp_state to session.
    session.mtp_state = Some(mtp_state);

    // If no drafts were generated, skip speculation.
    if drafts.is_empty() {
        session.eval_token(main_token)?;
        return Ok(vec![main_token]);
    }

    // === Verification phase ===
    // Snapshot the KV cache before verification. We use eval_token_no_snapshot
    // during verification so that eval_token_inner does not overwrite this
    // snapshot with per-token rollback state.
    session.snapshot_kv();

    // Evaluate main_token (always accepted).
    // Use a closure to capture errors and restore KV cache on failure.
    let verify_result = (|| -> Result<Vec<u32>> {
        session.eval_token_no_snapshot(main_token)?;
        let mut accepted = vec![main_token];

        // Verify each draft token against the main model's prediction.
        // After evaluating main_token, the main model's logits predict position P+1.
        // drafts[0] is MTP's prediction for position P+1, so we should compare them.
        for &draft in drafts.iter() {
            let main_pred = Session::argmax(session.logits())
                .ok_or_else(|| anyhow::anyhow!("speculative: empty logits during verification"))?;
            if main_pred != draft {
                break;
            }
            session.eval_token_no_snapshot(draft)?;
            accepted.push(draft);
        }

        Ok(accepted)
    })();

    // If verification failed, restore KV cache and propagate the error.
    let accepted = match verify_result {
        Ok(accepted) => accepted,
        Err(e) => {
            session.restore_kv();
            return Err(e);
        }
    };

    // === No rollback needed ===
    // eval_token_no_snapshot is only called for tokens that pass verification,
    // so the KV cache and session tokens are already in the correct state.
    let n_accepted = accepted.len();
    let total_drafts = drafts.len();

    // === MTP state alignment ===
    {
        let mtp_state = session.mtp_state.as_mut().unwrap();
        let n_embd = engine.config.n_embd as usize;

        // Restore prev_hidden to the state after the last accepted draft.
        // hidden_snapshots[i] is MTP state after drafts[i].
        // n_accepted includes main_token, so m = n_accepted - 1 draft tokens accepted.
        // Last accepted draft is drafts[m-1] = drafts[n_accepted - 2].
        if n_accepted >= 2 && n_accepted - 1 <= mtp_state.hidden_snapshot_count() {
            mtp_state.restore_hidden_snapshot(n_accepted - 2, n_embd);
        }

        // Pop rejected entries from MTP KV cache.
        // Each draft runs mtp_forward which pushes one KV entry.
        // Accepted draft tokens = m = n_accepted - 1.
        // Rejected drafts = total_drafts - m = total_drafts - (n_accepted - 1).
        let rejected = total_drafts.saturating_sub(n_accepted - 1);
        for _ in 0..rejected {
            mtp_state.kv_cache.pop_last_row();
        }
    }

    Ok(accepted)
}

/// Draft tokens using the MTP model. Returns the draft tokens.
///
/// `drafts[0]` is the MTP prediction using the main model's hidden state.
/// Subsequent drafts use MTP's own `prev_hidden`.
///
/// Hidden snapshots are stored in `mtp_state.hidden_snapshots_flat` and
/// can be retrieved via `mtp_state.get_hidden_snapshot(i, n_embd)`.
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
    s_prev_hidden: &mut [f32],
) -> Result<Vec<u32>> {
    let eos = engine.tokenizer.eos_token();
    let max_drafts = max_drafts.min(crate::mtp::MAX_DRAFT_TOKENS);
    let n_embd = engine.config.n_embd as usize;

    // Reset snapshot counter at start of drafting.
    mtp_state.reset_hidden_snapshots();

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
        return Ok(vec![]);
    }

    let mut drafts = Vec::with_capacity(max_drafts);
    drafts.push(draft0);
    mtp_state.store_hidden_snapshot(n_embd);

    // Draft[1..N] using MTP's own prev_hidden.
    for i in 1..max_drafts {
        let prev_token = *drafts.last().unwrap();
        if prev_token == eos {
            break;
        }
        let draft_pos = pos + i as u32;

        // Copy prev_hidden to scratch buffer before calling mtp_forward,
        // which will overwrite prev_hidden with the new hidden state.
        s_prev_hidden.copy_from_slice(&mtp_state.prev_hidden);

        let draft = mtp_forward(
            mtp_state,
            mtp_weights,
            main_weights,
            engine,
            prev_token,
            draft_pos,
            s_prev_hidden,
        )?;

        drafts.push(draft);
        mtp_state.store_hidden_snapshot(n_embd);

        if draft == eos {
            break;
        }
    }

    Ok(drafts)
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
