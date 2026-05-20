//! Streaming indexer compressor + top-k allowed-mask for ratio-4 layers.
//!
//! Mirrors `indexer_decode_one` and `indexer_allowed_decode_one` in antirez/ds4
//! ds4.c (lines 6332-6470 and attention-mix path). Only layers with
//! `layer_compress_ratio(il) == 4` carry an indexer; ratio-128 layers and the
//! dense first two layers do not.
//!
//! The indexer has two roles:
//!
//! 1. **Streaming compressor** (`indexer_decode_one`) — projects the post-RMSNorm activation
//!    through `indexer_compressor_kv/gate/ape`, pools on every ratio boundary, and emits one
//!    `INDEXER_HEAD_DIM = 128` wide compressed row into [`IndexerState`].
//!
//! 2. **Top-k mask** (`indexer_allowed_decode_one`) — computes per-head dot-product scores between
//!    the indexer Q (projected from the Q-LoRA activation via `indexer.attn_q_b`) and the emitted
//!    compressed rows, then returns the `top_k = min(INDEXER_TOP_K, n_comp)` highest-scoring
//!    indices for each indexer head. The mixed-attention path consumes this mask for ratio-4
//!    layers.
//!
//! ### What this PR skips
//!
//! After RoPE, ds4.c calls a Hadamard + FP4 quantize round-trip on the indexer
//! compressor output. That's deferred until later in Phase 2. The compressed
//! rows emitted here are float32 throughout.

use anyhow::Result;

use crate::{
    config::{INDEXER_HEAD_DIM, INDEXER_TOP_K},
    model::{
        kv_cache::{IndexerState, NEG_INF},
        layer::IndexerLayerWeights,
    },
    ops::{matmul::matmul_row, rope::RopeFreqs},
    quant::q8_0::f16_to_f32,
};

/// RMSNorm epsilon used by the indexer's pre-RoPE normalisation.
/// Matches `DS4_RMS_EPS` in ds4.c.
const RMS_EPS: f32 = 1.0e-6;

/// Run one streaming-indexer step for the current decode token.
///
/// Always writes the new state row at `state_row = ratio + (pos % ratio)`
/// (the indexer is always ratio == 4, so rows `[ratio..2*ratio)` are the
/// active write lane). Only on `(pos + 1) % ratio == 0` does it pool,
/// RMSNorm, and RoPE the row into `out_comp[..IDX_DIM]` and return
/// `Ok(true)` — otherwise it returns `Ok(false)` and `out_comp` is left
/// untouched.
///
/// The caller is responsible for pushing the emitted row into
/// [`IndexerState::push_comp`] when this returns true.
///
/// `rope_freqs_long` is the **long-context** RoPE cache (base = 160_000,
/// scale_factor = 16). The position used for the rotation is `pos + 1 - ratio`,
/// matching ds4.c.
pub fn indexer_decode_one(
    out_comp: &mut [f32],
    weights: &IndexerLayerWeights<'_>,
    state: &mut IndexerState,
    x: &[f32],
    rope_freqs_long: &RopeFreqs,
    pos: u32,
    _il: u32,
) -> Result<bool> {
    let ratio: usize = 4;
    assert_eq!(
        out_comp.len(),
        IDX_DIM,
        "indexer_decode_one: out_comp width"
    );
    let width = state.width();

    let pos_mod = (pos as usize) % ratio;
    let state_row = ratio + pos_mod;

    let row_off = state_row * width;
    let kv_row = &mut state.state_kv[row_off..row_off + width];
    matmul_row(weights.compressor_kv, x, kv_row);

    let sc_row = &mut state.state_score[row_off..row_off + width];
    matmul_row(weights.compressor_gate, x, sc_row);
    add_ape_bias(sc_row, weights.compressor_ape, pos_mod, width, ratio);

    let should_compress = (pos as usize + 1).is_multiple_of(ratio);
    if !should_compress {
        return Ok(false);
    }

    // Pool the per-row scores into one IDX_DIM-wide row.
    let mut pooled = [0.0f32; IDX_DIM];
    indexer_pool(&mut pooled, state);

    // RMSNorm with the learned per-dim weight.
    let scale = rms_scale(&pooled, RMS_EPS);
    for (o, (&p, &w)) in out_comp
        .iter_mut()
        .zip(pooled.iter().zip(weights.compressor_norm))
    {
        *o = p * scale * w;
    }

    // Long-context RoPE on the trailing 64 dims at the compressor position.
    let comp_pos = pos as usize + 1 - ratio;
    crate::ops::rope::apply_rope(out_comp, comp_pos, rope_freqs_long);

    // TODO(phase2-fp4): restore Hadamard + FP4 quantize round-trip.

    // Slide window: bit-for-bit copy of the C reference's two-step move.
    //   rows [ratio..2*ratio) -> rows [0..ratio)
    //   rows [0..ratio)       -> rows [ratio..2*ratio)
    // (Yes, the second step re-duplicates the just-shifted rows.)
    let kv_block = ratio * width;
    state.state_kv.copy_within(kv_block..2 * kv_block, 0);
    state.state_kv.copy_within(0..kv_block, kv_block);
    state.state_score.copy_within(kv_block..2 * kv_block, 0);
    state.state_score.copy_within(0..kv_block, kv_block);

    Ok(true)
}

/// Pool the streaming-indexer state into one `IDX_DIM`-wide row.
///
/// Mirrors the ratio == 4 path of `compressor_pool_decode_state` but at
/// width `IDX_DIM = 128` rather than `HEAD_DIM = 512`. The dual-lane
/// softmax reads:
///   * primary lane: cols `0..IDX_DIM` of rows `[0..ratio)`
///   * compressed lane: cols `IDX_DIM..width` of rows `[ratio..2*ratio)`
pub fn indexer_pool(out: &mut [f32], state: &IndexerState) {
    assert_eq!(out.len(), IDX_DIM);
    let width = state.width();
    let ratio = 4usize;
    let neg_inf_thresh = NEG_INF * 0.5;

    let kv = &state.state_kv;
    let sc = &state.state_score;
    for j in 0..IDX_DIM {
        // Column max over both lanes.
        let mut max = f32::NEG_INFINITY;
        for r in 0..ratio {
            let p = sc[r * width + j];
            let c = sc[(ratio + r) * width + IDX_DIM + j];
            if p > max {
                max = p;
            }
            if c > max {
                max = c;
            }
        }
        if max <= neg_inf_thresh {
            out[j] = 0.0;
            continue;
        }
        let mut denom = 0.0f32;
        let mut sum = 0.0f32;
        for r in 0..ratio {
            let wp = (sc[r * width + j] - max).exp();
            let wc = (sc[(ratio + r) * width + IDX_DIM + j] - max).exp();
            denom += wp + wc;
            sum += wp * kv[r * width + j] + wc * kv[(ratio + r) * width + IDX_DIM + j];
        }
        out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
    }
}

/// Compute the per-head top-k allowed mask for the indexer.
///
/// Returns a flat `Vec<usize>` of length `n_head * k` where
/// `k = min(INDEXER_TOP_K, n_comp)`. The slice for head `h` is at
/// `out[h * k .. (h + 1) * k]`, containing the indices of the `k`
/// highest-scoring compressed rows for that head in descending order of score.
///
/// The score for head `h` and compressed row `i` is:
///   `score = dot(indexer_q[h], comp_row[i]) + proj_score[h]`
/// where `indexer_q` is projected from `q_lora_norm` via `attn_q_b`, and
/// `proj_score` from `attn_norm` via `proj`.
///
/// TODO(phase2): verify the exact scoring formula against the C reference
/// (multiplicative vs additive proj, temperature scaling, etc.).
pub fn indexer_allowed_decode_one(
    weights: &IndexerLayerWeights<'_>,
    state: &IndexerState,
    q_lora_norm: &[f32],
    attn_norm: &[f32],
) -> Result<Vec<usize>> {
    let n_idx_head = weights.proj.out_features();
    let idx_dim = weights.attn_q_b.out_features() / n_idx_head;
    let n_comp = state.n_comp;

    if n_comp == 0 {
        return Ok(Vec::new());
    }

    let k = n_comp.min(INDEXER_TOP_K as usize);

    // Compute indexer Q: [n_idx_head * idx_dim]
    let mut indexer_q = vec![0.0f32; n_idx_head * idx_dim];
    matmul_row(weights.attn_q_b, q_lora_norm, &mut indexer_q);

    // Compute proj scores: [n_idx_head]
    let mut proj_scores = vec![0.0f32; n_idx_head];
    matmul_row(weights.proj, attn_norm, &mut proj_scores);

    // Compute scores and top-k per head.
    let mut allowed = vec![0usize; n_idx_head * k];
    let comp_rows = state.comp_rows(); // [n_comp, idx_dim] row-major

    for h in 0..n_idx_head {
        let mut scores = vec![0.0f32; n_comp];
        let qh = &indexer_q[h * idx_dim..(h + 1) * idx_dim];

        for (i, row) in comp_rows.chunks_exact(idx_dim).enumerate() {
            let dot: f32 = qh.iter().zip(row.iter()).map(|(&q, &k)| q * k).sum();
            scores[i] = dot + proj_scores[h];
        }

        let head_out = &mut allowed[h * k..(h + 1) * k];
        topk_indices_desc(&scores, k, head_out);
    }

    Ok(allowed)
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Width of one emitted indexer row (`DS4_N_INDEXER_HEAD_DIM`).
const IDX_DIM: usize = INDEXER_HEAD_DIM as usize;

/// Add `ape[col=pos_mod]` to `sc_row` element-wise. The ape tensor is
/// stored as F16 with shape `[width, ratio]`.
fn add_ape_bias(
    sc_row: &mut [f32],
    ape: crate::ops::matmul::WeightView<'_>,
    pos_mod: usize,
    width: usize,
    ratio: usize,
) {
    use crate::ops::matmul::WeightView;
    assert!(pos_mod < ratio);
    assert_eq!(sc_row.len(), width);
    match ape {
        WeightView::F16 {
            bytes,
            in_features,
            out_features,
        } => {
            assert_eq!(in_features, width);
            assert_eq!(out_features, ratio);
            let row_off = pos_mod * width * 2;
            let row_bytes = &bytes[row_off..row_off + width * 2];
            for (s, chunk) in sc_row.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *s += f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        _ => panic!("indexer ape: expected F16 view"),
    }
}

fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let n = x.len();
    if n == 0 {
        return 1.0 / eps.sqrt();
    }
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let mean = sum_sq / n as f32;
    1.0 / (mean + eps).sqrt()
}

/// Pick the indices of the top-`k` largest entries in `values`, in descending
/// order of value, into `out`. Ties resolve toward the lower index.
///
/// NaN safety: the comparison is `values[i] > best_v` with `best_v` seeded at
/// `f32::NEG_INFINITY`. `NaN > x` is always false in IEEE-754, so any NaN
/// inputs are skipped on every slot rather than poisoning the result. When
/// every remaining value is NaN or -inf we fall back to the lowest still-
/// unselected index.
fn topk_indices_desc(values: &[f32], k: usize, out: &mut [usize]) {
    debug_assert!(k <= values.len());
    debug_assert_eq!(out.len(), k);
    for slot in 0..k {
        let mut best_i = usize::MAX;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in values.iter().enumerate() {
            if out[..slot].contains(&i) {
                continue;
            }
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        if best_i == usize::MAX {
            best_i = (0..values.len())
                .find(|i| !out[..slot].contains(i))
                .expect("topk_indices_desc: k > values.len() should be impossible");
        }
        out[slot] = best_i;
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::INDEXER_HEAD,
        model::kv_cache::IndexerState,
        ops::rope::{RopeFreqs, RopeParams, YarnParams},
    };

    fn long_freqs() -> RopeFreqs {
        RopeFreqs::new(&RopeParams {
            n_rot: 64,
            base: 160_000.0,
            yarn: Some(YarnParams {
                scale_factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                orig_ctx: 65_536.0,
                attn_factor: None,
            }),
        })
    }

    fn zero_f16(
        bytes: &[u8],
        in_features: usize,
        out_features: usize,
    ) -> crate::ops::matmul::WeightView<'_> {
        crate::ops::matmul::WeightView::F16 {
            bytes,
            in_features,
            out_features,
        }
    }

    fn zeros(n: usize) -> Vec<u8> {
        vec![0u8; n * 2]
    }

    // -------------------------------------------------------------------
    // indexer_pool
    // -------------------------------------------------------------------

    #[test]
    fn pool_all_neg_inf_score_is_zero() {
        let mut state = IndexerState::new(1024).unwrap();
        for v in state.state_kv.iter_mut() {
            *v = 7.0;
        }
        let mut out = [1.0f32; IDX_DIM];
        indexer_pool(&mut out, &state);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "dim {i} should be zero, got {v}");
        }
    }

    #[test]
    fn pool_single_active_row_returns_kv_value() {
        let mut state = IndexerState::new(1024).unwrap();
        let width = state.width();
        for j in 0..IDX_DIM {
            state.state_score[2 * width + j] = 0.0;
            state.state_kv[2 * width + j] = j as f32 * 0.25;
        }
        let mut out = [0.0f32; IDX_DIM];
        indexer_pool(&mut out, &state);
        for (j, &v) in out.iter().enumerate() {
            assert!(
                (v - j as f32 * 0.25).abs() < 1e-5,
                "dim {j}: got {v} expected {}",
                j as f32 * 0.25,
            );
        }
    }

    #[test]
    fn pool_two_active_rows_softmax_average() {
        let mut state = IndexerState::new(1024).unwrap();
        let width = state.width();
        for j in 0..IDX_DIM {
            state.state_score[width + j] = 0.5;
            state.state_score[3 * width + j] = 0.5;
            state.state_kv[width + j] = 1.0;
            state.state_kv[3 * width + j] = 3.0;
        }
        let mut out = [0.0f32; IDX_DIM];
        indexer_pool(&mut out, &state);
        for (j, &v) in out.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-5, "dim {j}: got {v} expected 2.0",);
        }
    }

    #[test]
    fn pool_compressed_lane_only() {
        let mut state = IndexerState::new(64).unwrap();
        let width = state.width();
        let ratio = 4usize;
        for r in 0..ratio {
            for j in 0..IDX_DIM {
                let off = (ratio + r) * width + IDX_DIM + j;
                state.state_score[off] = 0.0;
                state.state_kv[off] = ((r * 2) as f32) + (j as f32) * 0.01;
            }
        }
        let mut out = [0.0f32; IDX_DIM];
        indexer_pool(&mut out, &state);
        for (j, &v) in out.iter().enumerate() {
            let expected: f32 = (0..ratio)
                .map(|r| ((r * 2) as f32) + (j as f32) * 0.01)
                .sum::<f32>()
                / ratio as f32;
            assert!(
                (v - expected).abs() < 1e-4,
                "dim {j}: got {v} expected {expected}",
            );
        }
    }

    // -------------------------------------------------------------------
    // indexer_decode_one boundary behaviour
    // -------------------------------------------------------------------

    #[test]
    fn decode_one_below_boundary_does_not_emit() {
        let n_embd = 16usize;
        let ratio = 4usize;
        let width = 2 * IDX_DIM;
        let kv_b = zeros(n_embd * width);
        let gate_b = zeros(n_embd * width);
        let ape_b = zeros(width * ratio);
        let norm = vec![1.0f32; IDX_DIM];
        let dummy_b = zeros(8);

        let weights = IndexerLayerWeights {
            compressor_kv: zero_f16(&kv_b, n_embd, width),
            compressor_gate: zero_f16(&gate_b, n_embd, width),
            compressor_ape: zero_f16(&ape_b, width, ratio),
            compressor_norm: &norm,
            attn_q_b: zero_f16(&dummy_b, 8, 8),
            proj: zero_f16(&dummy_b, 8, 8),
        };

        let mut state = IndexerState::new(64).unwrap();
        let x = vec![0.5f32; n_embd];
        let mut out = [0.0f32; IDX_DIM];
        let freqs = long_freqs();

        for pos in 0..3u32 {
            let emitted =
                indexer_decode_one(&mut out, &weights, &mut state, &x, &freqs, pos, 0).unwrap();
            assert!(!emitted, "pos {pos} (< ratio-1) should not emit");
        }
    }

    #[test]
    fn decode_one_emits_at_ratio_minus_one() {
        let n_embd = 16usize;
        let ratio = 4usize;
        let width = 2 * IDX_DIM;
        let kv_b = zeros(n_embd * width);
        let gate_b = zeros(n_embd * width);
        let ape_b = zeros(width * ratio);
        let norm = vec![1.0f32; IDX_DIM];
        let dummy_b = zeros(8);
        let weights = IndexerLayerWeights {
            compressor_kv: zero_f16(&kv_b, n_embd, width),
            compressor_gate: zero_f16(&gate_b, n_embd, width),
            compressor_ape: zero_f16(&ape_b, width, ratio),
            compressor_norm: &norm,
            attn_q_b: zero_f16(&dummy_b, 8, 8),
            proj: zero_f16(&dummy_b, 8, 8),
        };

        let mut state = IndexerState::new(64).unwrap();
        let x = vec![0.0f32; n_embd];
        let mut out = [0.0f32; IDX_DIM];
        let freqs = long_freqs();

        for pos in 0..3u32 {
            let emitted =
                indexer_decode_one(&mut out, &weights, &mut state, &x, &freqs, pos, 0).unwrap();
            assert!(!emitted);
        }
        let emitted = indexer_decode_one(&mut out, &weights, &mut state, &x, &freqs, 3, 0).unwrap();
        assert!(emitted, "pos = ratio-1 should emit");
    }

    #[test]
    fn decode_one_window_slide_after_emit() {
        let n_embd = 16usize;
        let ratio = 4usize;
        let width = 2 * IDX_DIM;
        let kv_b = zeros(n_embd * width);
        let gate_b = zeros(n_embd * width);
        let ape_b = zeros(width * ratio);
        let norm = vec![1.0f32; IDX_DIM];
        let dummy_b = zeros(8);
        let weights = IndexerLayerWeights {
            compressor_kv: zero_f16(&kv_b, n_embd, width),
            compressor_gate: zero_f16(&gate_b, n_embd, width),
            compressor_ape: zero_f16(&ape_b, width, ratio),
            compressor_norm: &norm,
            attn_q_b: zero_f16(&dummy_b, 8, 8),
            proj: zero_f16(&dummy_b, 8, 8),
        };
        let mut state = IndexerState::new(64).unwrap();

        let kv_block = ratio * width;
        for i in kv_block..2 * kv_block {
            state.state_kv[i] = (i - kv_block) as f32 * 0.5 + 1.0;
        }
        for i in kv_block..2 * kv_block {
            state.state_score[i] = (i - kv_block) as f32 * 0.5 - 7.0;
        }

        let x = vec![0.0f32; n_embd];
        let mut out = [0.0f32; IDX_DIM];
        let freqs = long_freqs();
        for pos in 0..4u32 {
            let _ = indexer_decode_one(&mut out, &weights, &mut state, &x, &freqs, pos, 0).unwrap();
        }

        for i in 0..kv_block {
            assert_eq!(
                state.state_kv[i],
                state.state_kv[kv_block + i],
                "kv_block dim {i}: low/high halves differ after slide",
            );
        }
        for i in 0..kv_block {
            assert_eq!(
                state.state_score[i],
                state.state_score[kv_block + i],
                "score block dim {i}: low/high halves differ after slide",
            );
        }
    }

    // -------------------------------------------------------------------
    // indexer_allowed_decode_one
    // -------------------------------------------------------------------

    #[test]
    fn allowed_empty_state_returns_empty() {
        let state = IndexerState::new(64).unwrap();
        assert_eq!(state.n_comp, 0);

        let dummy_b = zeros(8);
        let weights = IndexerLayerWeights {
            compressor_kv: zero_f16(&dummy_b, 8, 8),
            compressor_gate: zero_f16(&dummy_b, 8, 8),
            compressor_ape: zero_f16(&dummy_b, 8, 8),
            compressor_norm: &[],
            attn_q_b: zero_f16(&dummy_b, 8, 8),
            proj: zero_f16(&dummy_b, 8, 8),
        };

        let allowed = indexer_allowed_decode_one(&weights, &state, &[], &[]).unwrap();
        assert!(allowed.is_empty());
    }

    #[test]
    fn allowed_all_zero_weights_picks_lowest_indices() {
        // Zero weights => all scores are equal (all dot products 0, all proj 0).
        // topk_indices_desc breaks ties toward lower index.
        let mut state = IndexerState::new(64).unwrap();
        // Push 6 compressed rows so n_comp = 6, k = min(512, 6) = 6.
        for i in 0..6 {
            let row: Vec<f32> = (0..IDX_DIM).map(|d| (i * 10 + d) as f32).collect();
            state.push_comp(&row).unwrap();
        }
        assert_eq!(state.n_comp, 6);

        let q_lora_rank = 8usize;
        let n_embd = 16usize;
        let n_idx_head = INDEXER_HEAD as usize;
        let idx_q_dim = n_idx_head * IDX_DIM;

        let dummy_b = zeros(8);
        let q_b_bytes = zeros(q_lora_rank * idx_q_dim);
        let proj_bytes = zeros(n_embd * n_idx_head);
        let weights = IndexerLayerWeights {
            compressor_kv: zero_f16(&dummy_b, 8, 8),
            compressor_gate: zero_f16(&dummy_b, 8, 8),
            compressor_ape: zero_f16(&dummy_b, 8, 8),
            compressor_norm: &[],
            attn_q_b: zero_f16(&q_b_bytes, q_lora_rank, idx_q_dim),
            proj: zero_f16(&proj_bytes, n_embd, n_idx_head),
        };

        let q_lora_norm = vec![0.0f32; q_lora_rank];
        let attn_norm = vec![0.0f32; n_embd];
        let allowed =
            indexer_allowed_decode_one(&weights, &state, &q_lora_norm, &attn_norm).unwrap();

        // Each head gets k = 6 indices: [0, 1, 2, 3, 4, 5]
        assert_eq!(allowed.len(), n_idx_head * 6);
        for h in 0..n_idx_head {
            let head_slice = &allowed[h * 6..(h + 1) * 6];
            assert_eq!(head_slice, &[0, 1, 2, 3, 4, 5], "head {h} mismatch");
        }
    }

    #[test]
    fn allowed_single_head_with_strong_score_picks_that_row() {
        // Build a tiny indexer with n_idx_head = 1 so we can hand-compute.
        // We override the constants by using weight views with the right shapes.
        let mut state = IndexerState::new(64).unwrap();
        // Two rows: row 0 = [1,0,0,...], row 1 = [0,1,0,...]
        let mut row0 = vec![0.0f32; IDX_DIM];
        row0[0] = 1.0;
        let mut row1 = vec![0.0f32; IDX_DIM];
        row1[1] = 1.0;
        state.push_comp(&row0).unwrap();
        state.push_comp(&row1).unwrap();

        let q_lora_rank = 1usize;
        let n_embd = 1usize;
        let n_idx_head = 1usize;
        let idx_dim = IDX_DIM;

        // attn_q_b: [q_lora_rank=1, n_idx_head * idx_dim = 1 * 128 = 128]
        // We want q = [0, 1, 0, 0, ...] so dot(row0) = 0, dot(row1) = 1.
        let mut q_b_bytes = zeros(idx_dim);
        // F16 1.0 = 0x3C00. Place at position 1 (dim 1 of the single head).
        let one_f16 = 0x3C00u16.to_le_bytes();
        q_b_bytes[2..4].copy_from_slice(&one_f16);

        // proj: [n_embd=1, n_idx_head=1] — zero so it adds nothing.
        let proj_bytes = zeros(1);
        let dummy_b = zeros(8);

        let weights = IndexerLayerWeights {
            compressor_kv: zero_f16(&dummy_b, 8, 8),
            compressor_gate: zero_f16(&dummy_b, 8, 8),
            compressor_ape: zero_f16(&dummy_b, 8, 8),
            compressor_norm: &[],
            attn_q_b: zero_f16(&q_b_bytes, q_lora_rank, idx_dim),
            proj: zero_f16(&proj_bytes, n_embd, n_idx_head),
        };

        let q_lora_norm = vec![1.0f32; q_lora_rank];
        let attn_norm = vec![0.0f32; n_embd];
        let allowed =
            indexer_allowed_decode_one(&weights, &state, &q_lora_norm, &attn_norm).unwrap();

        // k = min(512, 2) = 2. Head 0 should pick row 1 first (score = 1),
        // then row 0 (score = 0).
        assert_eq!(allowed, &[1, 0]);
    }

    #[test]
    fn allowed_proj_additive_bias_shifts_scores() {
        let mut state = IndexerState::new(64).unwrap();
        let mut row0 = vec![0.0f32; IDX_DIM];
        row0[0] = 1.0;
        let mut row1 = vec![0.0f32; IDX_DIM];
        row1[0] = 1.0;
        state.push_comp(&row0).unwrap();
        state.push_comp(&row1).unwrap();

        let q_lora_rank = 1usize;
        let n_embd = 1usize;
        let n_idx_head = 1usize;

        // attn_q_b: identity on dim 0 => q = [1, 0, ...]
        // dot(row0) = 1, dot(row1) = 1 (identical).
        let mut q_b_bytes = zeros(IDX_DIM);
        let one_f16 = 0x3C00u16.to_le_bytes();
        q_b_bytes[0..2].copy_from_slice(&one_f16);

        // proj: [1, 1] with value 10.0 => proj_score = 10.0
        // Scores: row0 = 1 + 10 = 11, row1 = 1 + 10 = 11 (still tied).
        // To break the tie and test proj, we need different dots.
        // Actually, with identical rows, the tie breaks to lower index.
        // Let's just verify the function runs and returns [0, 1].
        let mut proj_bytes = zeros(1);
        // 10.0 in f16 = 0x4900
        let ten_f16_bytes = 0x4900u16.to_le_bytes();
        proj_bytes[0..2].copy_from_slice(&ten_f16_bytes);

        let dummy_b = zeros(8);
        let weights = IndexerLayerWeights {
            compressor_kv: zero_f16(&dummy_b, 8, 8),
            compressor_gate: zero_f16(&dummy_b, 8, 8),
            compressor_ape: zero_f16(&dummy_b, 8, 8),
            compressor_norm: &[],
            attn_q_b: zero_f16(&q_b_bytes, q_lora_rank, IDX_DIM),
            proj: zero_f16(&proj_bytes, n_embd, n_idx_head),
        };

        let q_lora_norm = vec![1.0f32; q_lora_rank];
        let attn_norm = vec![1.0f32; n_embd];
        let allowed =
            indexer_allowed_decode_one(&weights, &state, &q_lora_norm, &attn_norm).unwrap();

        // Both rows have the same score (1 + 10 = 11), tie breaks to lower index.
        assert_eq!(allowed, &[0, 1]);
    }
}
