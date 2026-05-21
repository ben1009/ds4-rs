//! MTP (Multi-Token Prediction) draft model for speculative decoding.
//!
//! The MTP model is a single transformer block stored in a separate GGUF file.
//! It shares the main model's embedding table and output projection, but has
//! its own norms, projections, KV cache, and HC head.
//!
//! Tensor naming follows the `mtp.0.*` prefix convention from antirez/ds4.

use anyhow::Result;

use crate::{
    config::ModelConfig,
    engine::Engine,
    model::{
        WeightMap,
        forward::{embed_token, output_hc_weights},
        kv_cache::{KvCache, KvCacheSnapshot},
        layer::LayerWeights,
    },
    ops::{
        hc::{hc_from_plain_embedding, hc_weighted_sum},
        matmul::matmul_row,
        norm::{rms_norm, rms_norm_no_weight},
    },
    session::Session,
};

/// Borrowed weight views for the MTP draft model GGUF.
///
/// The MTP block is architecturally identical to one main-model transformer
/// layer, plus MTP-specific input projections and output norms.
pub struct MtpWeights<'a> {
    /// Embedding projection `[n_embd, n_embd]` Q8_0.
    pub e_proj: crate::ops::matmul::WeightView<'a>,
    /// Hidden state projection `[n_embd, n_embd]` Q8_0.
    pub h_proj: crate::ops::matmul::WeightView<'a>,
    /// Embedding RMSNorm weight `[n_embd]` F32.
    pub enorm: &'a [f32],
    /// Hidden state RMSNorm weight `[n_embd]` F32.
    pub hnorm: &'a [f32],
    /// Output RMSNorm weight `[n_embd]` F32.
    pub norm: &'a [f32],
    /// HC head function `[hc_dim, n_hc]` F16.
    pub hc_head_fn: crate::ops::matmul::WeightView<'a>,
    /// HC head scale scalar F32.
    pub hc_head_scale: &'a [f32],
    /// HC head base `[n_hc]` F32.
    pub hc_head_base: &'a [f32],
    /// Single transformer block (same structure as a main-model layer).
    pub block: LayerWeights<'a>,
}

impl<'a> MtpWeights<'a> {
    /// Load MTP weights from a `WeightMap` (the separately-mmapped MTP GGUF).
    pub fn from_map(map: &'a WeightMap) -> Result<Self> {
        let p = "mtp.0.";
        let n_embd = map.config.n_embd as usize;
        let n_hc = map.config.n_hc as usize;

        Ok(Self {
            e_proj: map.q8_0(&format!("{p}e_proj.weight"))?,
            h_proj: map.q8_0(&format!("{p}h_proj.weight"))?,
            enorm: map.f32_1d(&format!("{p}enorm.weight"), n_embd)?,
            hnorm: map.f32_1d(&format!("{p}hnorm.weight"), n_embd)?,
            norm: map.f32_1d(&format!("{p}norm.weight"), n_embd)?,
            hc_head_fn: map.f16(&format!("{p}hc_head_fn.weight"))?,
            hc_head_scale: map.f32_1d(&format!("{p}hc_head_scale.weight"), 1)?,
            hc_head_base: map.f32_1d(&format!("{p}hc_head_base.weight"), n_hc)?,
            block: LayerWeights::from_prefix(map, p, 0)?,
        })
    }
}

/// Mutable state for the MTP draft model: its own KV cache and hidden state.
pub struct MtpState {
    /// KV cache for the single MTP transformer block (1 layer, no compressor/indexer).
    pub kv_cache: KvCache,
    /// Reusable rollback snapshot for the MTP KV cache.
    kv_snapshot: KvCacheSnapshot,
    /// Collapsed hidden state `[n_embd]` from the last MTP forward step.
    /// Zeros on first use; updated after each `mtp_forward` call.
    pub prev_hidden: Vec<f32>,
    /// Scratch buffer for attention heads `[n_head * head_dim]`.
    pub heads_scratch: Vec<f32>,
    /// Draft logits output buffer `[n_vocab]`.
    pub logits: Vec<f32>,
    /// Pre-allocated flat buffer for hidden state snapshots during speculative drafting.
    /// Layout: `[MAX_DRAFT_TOKENS * n_embd]`. Avoids per-draft heap allocations.
    hidden_snapshots_flat: Vec<f32>,
    /// Number of valid snapshots stored in `hidden_snapshots_flat`.
    hidden_snapshots_count: usize,
    // Reusable scratch buffers for mtp_forward (avoid per-call heap allocations).
    s_plain: Vec<f32>,
    s_enormed: Vec<f32>,
    s_eproj: Vec<f32>,
    s_eproj_hc: Vec<f32>,
    s_hnormed: Vec<f32>,
    s_hproj: Vec<f32>,
    s_hproj_hc: Vec<f32>,
    s_residual_hc: Vec<f32>,
    // Scratch buffers for run_transformer_block.
    s_attn_out: Vec<f32>,
    s_after_attn_hc: Vec<f32>,
    s_ffn_out: Vec<f32>,
    // Reusable scratch buffers for mtp_output_head.
    s_out_flat: Vec<f32>,
    s_out_pre: Vec<f32>,
    s_out_weights: Vec<f32>,
    s_out_plain: Vec<f32>,
    s_out_normed: Vec<f32>,
}

/// Maximum number of draft tokens for pre-allocated buffers.
pub(crate) const MAX_DRAFT_TOKENS: usize = 16;

impl MtpState {
    /// Allocate MTP state. `ctx_size` controls the KV ring capacity.
    pub fn new(config: &ModelConfig, ctx_size: u32) -> Result<Self> {
        let n_embd = config.n_embd as usize;
        let n_vocab = config.n_vocab as usize;
        let n_hc = config.n_hc as usize;
        let q_dim = (config.n_head as usize)
            .checked_mul(config.head_dim as usize)
            .ok_or_else(|| anyhow::anyhow!("MtpState: Q dimension overflow"))?;
        let hc_dim = n_hc
            .checked_mul(n_embd)
            .ok_or_else(|| anyhow::anyhow!("MtpState: HC dim overflow"))?;
        let kv_cache = KvCache::new(1, ctx_size as usize)?;
        let kv_snapshot = KvCacheSnapshot::with_shape(&kv_cache);
        Ok(Self {
            kv_cache,
            kv_snapshot,
            prev_hidden: vec![0.0f32; n_embd],
            heads_scratch: vec![0.0f32; q_dim],
            logits: vec![0.0f32; n_vocab],
            hidden_snapshots_flat: vec![0.0f32; MAX_DRAFT_TOKENS * n_embd],
            hidden_snapshots_count: 0,
            s_plain: vec![0.0f32; n_embd],
            s_enormed: vec![0.0f32; n_embd],
            s_eproj: vec![0.0f32; n_embd],
            s_eproj_hc: vec![0.0f32; hc_dim],
            s_hnormed: vec![0.0f32; n_embd],
            s_hproj: vec![0.0f32; n_embd],
            s_hproj_hc: vec![0.0f32; hc_dim],
            s_residual_hc: vec![0.0f32; hc_dim],
            s_attn_out: vec![0.0f32; n_embd],
            s_after_attn_hc: vec![0.0f32; hc_dim],
            s_ffn_out: vec![0.0f32; n_embd],
            s_out_flat: vec![0.0f32; hc_dim],
            s_out_pre: vec![0.0f32; n_hc],
            s_out_weights: vec![0.0f32; n_hc],
            s_out_plain: vec![0.0f32; n_embd],
            s_out_normed: vec![0.0f32; n_embd],
        })
    }

    /// Store a snapshot of `prev_hidden` into the pre-allocated flat buffer.
    pub fn store_hidden_snapshot(&mut self, n_embd: usize) {
        assert!(
            self.hidden_snapshots_count < MAX_DRAFT_TOKENS,
            "MtpState: hidden snapshot overflow"
        );
        let idx = self.hidden_snapshots_count * n_embd;
        self.hidden_snapshots_flat[idx..idx + n_embd].copy_from_slice(&self.prev_hidden);
        self.hidden_snapshots_count += 1;
    }

    /// Get a slice to a stored hidden snapshot by index.
    pub fn get_hidden_snapshot(&self, index: usize, n_embd: usize) -> &[f32] {
        let idx = index * n_embd;
        &self.hidden_snapshots_flat[idx..idx + n_embd]
    }

    /// Copy a hidden snapshot into prev_hidden by index.
    pub fn restore_hidden_snapshot(&mut self, index: usize, n_embd: usize) {
        let idx = index * n_embd;
        self.prev_hidden
            .copy_from_slice(&self.hidden_snapshots_flat[idx..idx + n_embd]);
    }

    /// Reset the hidden snapshot counter (call at start of drafting).
    pub fn reset_hidden_snapshots(&mut self) {
        self.hidden_snapshots_count = 0;
    }

    /// Get the number of stored hidden snapshots.
    pub fn hidden_snapshot_count(&self) -> usize {
        self.hidden_snapshots_count
    }

    /// Borrow the s_out_plain scratch buffer (for temporary hidden state storage).
    pub fn scratch_plain(&mut self) -> &mut [f32] {
        &mut self.s_out_plain
    }

    /// Zero the hidden state and clear the KV ring.
    pub fn clear(&mut self) {
        self.prev_hidden.fill(0.0);
        self.kv_cache.clear_all();
    }

    /// Snapshot the MTP KV cache into the reusable rollback buffer.
    pub fn snapshot_kv(&mut self) {
        self.kv_cache.snapshot_into(&mut self.kv_snapshot);
    }

    /// Restore the MTP KV cache from the rollback snapshot.
    pub fn restore_kv(&mut self) {
        self.kv_cache.restore(&self.kv_snapshot);
    }
}

/// Run the MTP forward pass for one draft token.
///
/// Returns the draft token id (greedy argmax of MTP logits).
///
/// * `hidden` — the `[n_embd]` input hidden state. For the first draft this is the main model's
///   `session.last_hidden`; for subsequent drafts it is `mtp_state.prev_hidden`.
pub fn mtp_forward(
    mtp_state: &mut MtpState,
    mtp_weights: &MtpWeights<'_>,
    main_weights: &WeightMap,
    engine: &Engine,
    token: u32,
    pos: u32,
    hidden: &[f32],
) -> Result<u32> {
    let config = &engine.config;
    let n_embd = config.n_embd as usize;
    let n_hc = config.n_hc as usize;

    // 1. Embed token from main model's embedding table.
    mtp_state.s_plain.fill(0.0);
    embed_token(main_weights, token, &mut mtp_state.s_plain)?;

    // 2. RMSNorm embedding.
    rms_norm(
        &mtp_state.s_plain,
        mtp_weights.enorm,
        1e-6,
        &mut mtp_state.s_enormed,
    );

    // 3. Project embedding.
    matmul_row(
        mtp_weights.e_proj,
        &mtp_state.s_enormed,
        &mut mtp_state.s_eproj,
    );

    // 4. Expand to HC layout.
    hc_from_plain_embedding(&mut mtp_state.s_eproj_hc, &mtp_state.s_eproj, n_embd, n_hc);

    // 5. Norm hidden state.
    rms_norm(hidden, mtp_weights.hnorm, 1e-6, &mut mtp_state.s_hnormed);

    // 6. Project hidden state.
    matmul_row(
        mtp_weights.h_proj,
        &mtp_state.s_hnormed,
        &mut mtp_state.s_hproj,
    );

    // 7. Expand to HC layout.
    hc_from_plain_embedding(&mut mtp_state.s_hproj_hc, &mtp_state.s_hproj, n_embd, n_hc);

    // 8. Combine: eproj_hc + hproj_hc.
    mtp_state
        .s_residual_hc
        .iter_mut()
        .zip(mtp_state.s_eproj_hc.iter())
        .zip(mtp_state.s_hproj_hc.iter())
        .for_each(|((res, e), h)| *res = e + h);

    // 9. Run the single transformer block (attention + FFN with HC pre/post).
    crate::model::forward::run_transformer_block(
        &mut mtp_state.s_residual_hc,
        engine,
        &mtp_weights.block,
        &mut mtp_state.kv_cache,
        &mut mtp_state.heads_scratch,
        &mut mtp_state.s_attn_out,
        &mut mtp_state.s_after_attn_hc,
        &mut mtp_state.s_ffn_out,
        0, // il = 0 (single block)
        pos as usize,
        token,
    )?;

    // 10. Output head using MTP's HC head + main model's output projection.
    // mtp_output_head writes the HC-collapsed hidden state into s_out_plain.
    mtp_output_head(
        mtp_weights,
        main_weights,
        config,
        &mtp_state.s_residual_hc,
        &mut mtp_state.logits,
        &mut mtp_state.s_out_flat,
        &mut mtp_state.s_out_pre,
        &mut mtp_state.s_out_weights,
        &mut mtp_state.s_out_plain,
        &mut mtp_state.s_out_normed,
    )?;

    // 11. Argmax → draft token.
    let draft_token = Session::argmax(&mtp_state.logits)
        .ok_or_else(|| anyhow::anyhow!("mtp_forward: no valid token"))?;

    // 12. Collapse HC → prev_hidden for next step.
    // Use the learned HC head output (s_out_plain) with RMSNorm, matching
    // how the main model captures last_hidden after HC reduction.
    rms_norm(
        &mtp_state.s_out_plain,
        mtp_weights.norm,
        1e-6,
        &mut mtp_state.prev_hidden,
    );

    Ok(draft_token)
}

/// MTP output head: HC reduction + norm + main model's LM head projection.
///
/// Uses MTP's own HC head weights (`hc_head_fn`, `hc_head_scale`,
/// `hc_head_base`) and `norm`, but the main model's shared `output.weight`.
#[allow(clippy::too_many_arguments)]
fn mtp_output_head(
    mtp_weights: &MtpWeights<'_>,
    main_weights: &WeightMap,
    config: &ModelConfig,
    residual_hc: &[f32],
    logits: &mut [f32],
    flat: &mut [f32],
    pre: &mut [f32],
    weights: &mut [f32],
    plain: &mut [f32],
    normed: &mut [f32],
) -> Result<()> {
    let n_embd = config.n_embd as usize;
    let n_hc = config.n_hc as usize;

    // Learned HC reduction (same formula as main model's output head).
    rms_norm_no_weight(residual_hc, 1e-6, flat);

    matmul_row(mtp_weights.hc_head_fn, flat, pre);

    output_hc_weights(
        pre,
        mtp_weights.hc_head_scale[0],
        mtp_weights.hc_head_base,
        weights,
    );

    hc_weighted_sum(residual_hc, weights, plain, n_embd, n_hc);

    // Output norm (MTP's own norm weight, not the main model's output_norm).
    rms_norm(plain, mtp_weights.norm, 1e-6, normed);

    // LM head projection (shared with main model's output.weight).
    let output_weight = main_weights.q8_0("output.weight")?;
    matmul_row(output_weight, normed, logits);

    Ok(())
}
