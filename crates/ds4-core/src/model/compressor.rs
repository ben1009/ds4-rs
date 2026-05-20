//! Streaming compressor for the attention path.
//!
//! Mirrors `compressor_decode_one` and `compressor_pool_decode_state` in
//! antirez/ds4 ds4.c (lines 6459-6610). For each decode token at absolute
//! position `pos`, this projects the post-RMSNorm activation through
//! `attn_compressor_kv` / `attn_compressor_gate`, adds the `ape` positional
//! bias for `pos % ratio`, writes the per-token state row, and on every
//! `ratio`-boundary token emits one pooled, RoPE'd, RMSNorm'd compressed-KV
//! row that the caller pushes into [`crate::model::kv_cache::CompressorState`].
//!
//! The compressed rows are consumed by the mixed-attention path for ratio-4
//! layers.
//!
//! ### What this PR skips
//!
//! After RoPE, ds4.c calls `dsv4_fp8_kv_quantize_row_inplace_cpu` (an FP8
//! E4M3 round-trip on the non-RoPE half). That's deferred until later in
//! Phase 2 — see the `TODO(phase2-fp8)` marker below. The compressed rows
//! emitted here are float32 throughout.

use anyhow::Result;

use crate::{
    model::{
        kv_cache::{CompressorState, HEAD_DIM, NEG_INF},
        layer::CompressorLayerWeights,
    },
    ops::{matmul::matmul_row, rope::RopeFreqs},
    quant::q8_0::f16_to_f32,
};

/// RMSNorm epsilon used by the compressor's pre-RoPE normalisation.
/// Matches `DS4_RMS_EPS` in ds4.c.
const RMS_EPS: f32 = 1.0e-6;

/// Run one streaming-compressor step for the current decode token.
///
/// Always writes the new state row at `state_row = (ratio == 4 ? ratio : 0)
/// + (pos % ratio)`. Only on `(pos + 1) % ratio == 0` does it pool, RMSNorm,
/// and RoPE the row into `out_comp[..HEAD_DIM]` and return `Ok(true)` —
/// otherwise it returns `Ok(false)` and `out_comp` is left untouched.
///
/// The caller is responsible for pushing the emitted row into
/// [`CompressorState::push_comp`] when this returns true.
///
/// `rope_freqs_long` is the **long-context** RoPE cache (base = 160_000,
/// scale_factor = 16). The position used for the rotation is `pos + 1 - ratio`,
/// matching ds4.c.
pub fn compressor_decode_one(
    out_comp: &mut [f32],
    weights: &CompressorLayerWeights<'_>,
    state: &mut CompressorState,
    x: &[f32],
    rope_freqs_long: &RopeFreqs,
    pos: u32,
    _il: u32,
) -> Result<bool> {
    let ratio = state.ratio as usize;
    assert!(ratio != 0, "compressor_decode_one called on dense layer");
    assert_eq!(out_comp.len(), HEAD_DIM);
    let width = state.width();

    // Project to KV / score rows.
    let pos_mod = (pos as usize) % ratio;
    let state_row = if state.ratio == 4 {
        ratio + pos_mod
    } else {
        pos_mod
    };

    let row_off = state_row * width;
    let kv_row = &mut state.state_kv[row_off..row_off + width];
    matmul_row(weights.kv, x, kv_row);

    let sc_row = &mut state.state_score[row_off..row_off + width];
    matmul_row(weights.gate, x, sc_row);
    add_ape_bias(sc_row, weights.ape, pos_mod, width, ratio);

    let should_compress = (pos as usize + 1).is_multiple_of(ratio);
    if !should_compress {
        return Ok(false);
    }

    // Pool the per-row scores into one HEAD_DIM-wide row.
    let mut pooled = [0.0f32; HEAD_DIM];
    compressor_pool(&mut pooled, state);

    // RMSNorm with the learned per-dim weight.
    let scale = rms_scale(&pooled, RMS_EPS);
    for (o, (&p, &w)) in out_comp.iter_mut().zip(pooled.iter().zip(weights.norm)) {
        *o = p * scale * w;
    }

    // Long-context RoPE on the trailing 64 dims at the compressor position.
    // The compressed row covers the window `[pos+1-ratio, pos]` so its
    // anchor position is the window start.
    let comp_pos = pos as usize + 1 - ratio;
    crate::ops::rope::apply_rope(out_comp, comp_pos, rope_freqs_long);

    // TODO(phase2-fp8): restore FP8 KV round-trip
    // (`dsv4_fp8_kv_quantize_row_inplace_cpu` on the non-RoPE half).

    if state.ratio == 4 {
        // Slide window: bit-for-bit copy of the C reference's two-step move.
        //   rows [ratio..2*ratio) -> rows [0..ratio)
        //   rows [0..ratio)       -> rows [ratio..2*ratio)
        // (Yes, the second step re-duplicates the just-shifted rows.)
        let kv_block = ratio * width;
        state.state_kv.copy_within(kv_block..2 * kv_block, 0);
        state.state_kv.copy_within(0..kv_block, kv_block);
        state.state_score.copy_within(kv_block..2 * kv_block, 0);
        state.state_score.copy_within(0..kv_block, kv_block);
    }

    Ok(true)
}

/// Pool the streaming-compressor state into one `HEAD_DIM`-wide row.
///
/// Mirrors `compressor_pool_decode_state` in ds4.c (line 6459). For each
/// output dim `j`:
///
/// * `ratio != 4`: softmax over `state_score[r*width + j]` for `r in 0..ratio`, weighted sum of
///   `state_kv[r*width + j]`.
/// * `ratio == 4`: dual-lane softmax — `r in 0..ratio` selects from the *primary* lane (cols
///   `0..head_dim`) of rows `[0..ratio)`, *plus* the *compressed* lane (cols `head_dim..width`) of
///   rows `[ratio..2*ratio)`.
///
/// When the column max is at or below `NEG_INF * 0.5`, the output is exactly
/// zero (no real scores in that column).
pub fn compressor_pool(out: &mut [f32], state: &CompressorState) {
    assert_eq!(out.len(), HEAD_DIM);
    let width = state.width();
    let ratio = state.ratio as usize;
    let neg_inf_thresh = NEG_INF * 0.5;

    if state.ratio == 4 {
        let kv = &state.state_kv;
        let sc = &state.state_score;
        for j in 0..HEAD_DIM {
            // Column max over both lanes.
            let mut max = f32::NEG_INFINITY;
            for r in 0..ratio {
                let p = sc[r * width + j];
                let c = sc[(ratio + r) * width + HEAD_DIM + j];
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
                let wc = (sc[(ratio + r) * width + HEAD_DIM + j] - max).exp();
                denom += wp + wc;
                sum += wp * kv[r * width + j] + wc * kv[(ratio + r) * width + HEAD_DIM + j];
            }
            out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
        }
    } else {
        let kv = &state.state_kv;
        let sc = &state.state_score;
        for j in 0..HEAD_DIM {
            let mut max = f32::NEG_INFINITY;
            for r in 0..ratio {
                let v = sc[r * width + j];
                if v > max {
                    max = v;
                }
            }
            if max <= neg_inf_thresh {
                out[j] = 0.0;
                continue;
            }
            let mut denom = 0.0f32;
            let mut sum = 0.0f32;
            for r in 0..ratio {
                let w = (sc[r * width + j] - max).exp();
                denom += w;
                sum += w * kv[r * width + j];
            }
            out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
        }
    }
}

/// Add `ape[col=pos_mod]` to `sc_row` element-wise. The ape tensor is
/// stored as F16 with shape `[width, ratio]` (ne0=width inner, ne1=ratio
/// outer in GGUF), so column `pos_mod` is row `pos_mod` of `width` F16
/// values. The kernel dequants on the fly — matches the C reference's
/// `tensor_2d_value` per-element read (slow, but clear; the perf path is
/// PR-future).
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
            // F16: 2 bytes per element. Row `pos_mod` of `width` values.
            let row_off = pos_mod * width * 2;
            let row_bytes = &bytes[row_off..row_off + width * 2];
            for (s, chunk) in sc_row.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *s += f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        _ => panic!("compressor ape: expected F16 view"),
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

#[cfg(test)]
// NOTE: These tests are intentionally excluded from the Miri CI matrix.
// Each test takes ~40s under Miri because the pool/decode kernels loop
// over 512-dim vectors and Miri interprets every memory access. There
// is no `unsafe` in this module, so Miri adds limited value here.
mod tests {
    use super::*;
    use crate::{
        model::kv_cache::CompressorState,
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

    // -------------------------------------------------------------------
    // compressor_pool truth table
    // -------------------------------------------------------------------

    #[test]
    fn pool_all_neg_inf_score_is_zero() {
        // Default state has score = NEG_INF everywhere → pool returns 0
        // for every output dim regardless of state_kv.
        let mut state = CompressorState::new(128, 1024).unwrap();
        for v in state.state_kv.iter_mut() {
            *v = 7.0;
        }
        let mut out = [1.0f32; HEAD_DIM];
        compressor_pool(&mut out, &state);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "dim {i} should be zero, got {v}");
        }
    }

    #[test]
    fn pool_ratio_128_single_active_row_returns_kv_value() {
        // One non-NEG_INF score per dim → softmax weight is 1.0, output
        // equals state_kv at that row.
        let mut state = CompressorState::new(128, 1024).unwrap();
        let width = state.width();
        // Activate row 7 with score 0.0; the rest stay at NEG_INF.
        for j in 0..HEAD_DIM {
            state.state_score[7 * width + j] = 0.0;
            state.state_kv[7 * width + j] = j as f32 * 0.25;
        }
        let mut out = [0.0f32; HEAD_DIM];
        compressor_pool(&mut out, &state);
        for (j, &v) in out.iter().enumerate() {
            assert!(
                (v - j as f32 * 0.25).abs() < 1e-5,
                "dim {j}: got {v} expected {}",
                j as f32 * 0.25,
            );
        }
    }

    #[test]
    fn pool_ratio_128_two_active_rows_softmax_average() {
        // Two equal scores → output is mean of the two kv values.
        let mut state = CompressorState::new(128, 1024).unwrap();
        let width = state.width();
        for j in 0..HEAD_DIM {
            state.state_score[width + j] = 0.5;
            state.state_score[5 * width + j] = 0.5;
            state.state_kv[width + j] = 1.0;
            state.state_kv[5 * width + j] = 3.0;
        }
        let mut out = [0.0f32; HEAD_DIM];
        compressor_pool(&mut out, &state);
        for (j, &v) in out.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-5, "dim {j}: got {v} expected 2.0",);
        }
    }

    #[test]
    fn pool_ratio_4_primary_lane_only() {
        // ratio=4: scores in primary lane only — output equals the softmax
        // of state_kv across rows [0..ratio).
        let mut state = CompressorState::new(4, 64).unwrap();
        let width = state.width();
        let ratio = 4usize;
        // Set scores 0.0 across ratio rows in primary lane (cols 0..head_dim).
        for r in 0..ratio {
            for j in 0..HEAD_DIM {
                state.state_score[r * width + j] = 0.0;
                state.state_kv[r * width + j] = (r as f32) + (j as f32) * 0.001;
            }
        }
        // Compressed lane stays NEG_INF — should contribute nothing.
        let mut out = [0.0f32; HEAD_DIM];
        compressor_pool(&mut out, &state);
        for (j, &v) in out.iter().enumerate() {
            // Equal weights → arithmetic mean of state_kv across ratio rows.
            let expected: f32 = (0..ratio)
                .map(|r| (r as f32) + (j as f32) * 0.001)
                .sum::<f32>()
                / ratio as f32;
            assert!(
                (v - expected).abs() < 1e-4,
                "dim {j}: got {v} expected {expected}",
            );
        }
    }

    #[test]
    fn pool_ratio_4_compressed_lane_only() {
        // ratio=4: scores only in the compressed lane (cols head_dim..width)
        // of rows [ratio..2*ratio). The pool reads kv from the compressed
        // lane of those rows — verify with a hand-computed softmax.
        let mut state = CompressorState::new(4, 64).unwrap();
        let width = state.width();
        let ratio = 4usize;
        for r in 0..ratio {
            for j in 0..HEAD_DIM {
                let off = (ratio + r) * width + HEAD_DIM + j;
                state.state_score[off] = 0.0;
                state.state_kv[off] = ((r * 2) as f32) + (j as f32) * 0.01;
            }
        }
        let mut out = [0.0f32; HEAD_DIM];
        compressor_pool(&mut out, &state);
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

    #[test]
    fn pool_ratio_4_dominant_score_picks_one_entry() {
        // One huge primary-lane score on row 2, dim 0: softmax → 1.0 weight
        // there; output[0] should equal state_kv at that slot.
        let mut state = CompressorState::new(4, 64).unwrap();
        let width = state.width();
        let ratio = 4usize;
        // Seed everything else with NEG_INF (default) and put a single big
        // score at the chosen slot.
        state.state_score[2 * width] = 100.0;
        state.state_kv[2 * width] = 42.5;
        // Also seed an irrelevant compressed-lane slot so we know it's
        // ignored.
        state.state_score[(ratio + 1) * width + HEAD_DIM] = -50.0;
        state.state_kv[(ratio + 1) * width + HEAD_DIM] = -999.0;

        let mut out = [0.0f32; HEAD_DIM];
        compressor_pool(&mut out, &state);
        assert!(
            (out[0] - 42.5).abs() < 1e-3,
            "dim 0 should pick the dominant score, got {}",
            out[0],
        );
        // All other dims have NEG_INF only → 0.
        for (j, &v) in out.iter().enumerate().skip(1) {
            assert_eq!(v, 0.0, "dim {j} should be 0, got {v}");
        }
    }

    // -------------------------------------------------------------------
    // compressor_decode_one boundary behaviour
    //
    // Build the compressor weights from a synthetic in-memory F16 byte
    // buffer (zeros) so matmuls produce zero state rows. That makes the
    // emit boundary easy to assert without computing any softmaxes.
    // -------------------------------------------------------------------

    /// Build an F16 view backed by a Vec<u8> of zeros with shape
    /// `[in_features, out_features]`. Caller owns the bytes.
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

    #[test]
    fn decode_one_below_boundary_does_not_emit() {
        let n_embd = 16usize;
        let ratio = 4u32;
        let coff = 2usize;
        let width = coff * HEAD_DIM;
        let kv_b = zeros(n_embd * width);
        let gate_b = zeros(n_embd * width);
        let ape_b = zeros(width * ratio as usize);
        let norm = vec![1.0f32; HEAD_DIM];

        let weights = CompressorLayerWeights {
            kv: zero_f16(&kv_b, n_embd, width),
            gate: zero_f16(&gate_b, n_embd, width),
            ape: zero_f16(&ape_b, width, ratio as usize),
            norm: &norm,
        };

        let mut state = CompressorState::new(ratio, 64).unwrap();
        let x = vec![0.5f32; n_embd];
        let mut out = [0.0f32; HEAD_DIM];
        let freqs = long_freqs();

        for pos in 0..3u32 {
            let emitted =
                compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, pos, 0).unwrap();
            assert!(!emitted, "pos {pos} (< ratio-1) should not emit");
        }
    }

    #[test]
    fn decode_one_emits_at_ratio_minus_one() {
        let n_embd = 16usize;
        let ratio = 4u32;
        let coff = 2usize;
        let width = coff * HEAD_DIM;
        let kv_b = zeros(n_embd * width);
        let gate_b = zeros(n_embd * width);
        let ape_b = zeros(width * ratio as usize);
        let norm = vec![1.0f32; HEAD_DIM];
        let weights = CompressorLayerWeights {
            kv: zero_f16(&kv_b, n_embd, width),
            gate: zero_f16(&gate_b, n_embd, width),
            ape: zero_f16(&ape_b, width, ratio as usize),
            norm: &norm,
        };

        let mut state = CompressorState::new(ratio, 64).unwrap();
        let x = vec![0.0f32; n_embd];
        let mut out = [0.0f32; HEAD_DIM];
        let freqs = long_freqs();

        for pos in 0..3u32 {
            let emitted =
                compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, pos, 0).unwrap();
            assert!(!emitted);
        }
        let emitted =
            compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, 3, 0).unwrap();
        assert!(emitted, "pos = ratio-1 should emit");
        // Next ratio window: pos 4..7; only pos=7 emits.
        for pos in 4..7u32 {
            let emitted =
                compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, pos, 0).unwrap();
            assert!(!emitted);
        }
        let emitted =
            compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, 7, 0).unwrap();
        assert!(emitted, "pos = 2*ratio-1 should emit again");
    }

    #[test]
    fn decode_one_ratio_4_window_slide_after_emit() {
        // After an emit on a ratio=4 layer, rows [0..ratio) must mirror the
        // bytes from rows [ratio..2*ratio) (then re-duplicated back). We
        // hand-seed the state so the slide is easy to assert.
        let n_embd = 16usize;
        let ratio = 4u32;
        let coff = 2usize;
        let width = coff * HEAD_DIM;
        let kv_b = zeros(n_embd * width);
        let gate_b = zeros(n_embd * width);
        let ape_b = zeros(width * ratio as usize);
        let norm = vec![1.0f32; HEAD_DIM];
        let weights = CompressorLayerWeights {
            kv: zero_f16(&kv_b, n_embd, width),
            gate: zero_f16(&gate_b, n_embd, width),
            ape: zero_f16(&ape_b, width, ratio as usize),
            norm: &norm,
        };
        let mut state = CompressorState::new(ratio, 64).unwrap();

        // Mark rows [ratio..2*ratio) of state_kv with a recognisable seed
        // before the emit. Pos 0..2 just write to rows [0..3); the emit at
        // pos = 3 writes to row `ratio + 3 = 7`. After the slide step,
        // rows 0..ratio should equal the seed we put into rows ratio..2*ratio
        // *prior* to emit, with row 3 overwritten by the kv_cur computed at
        // pos=3 (which is zeros since matmul weights are zeros — but the
        // ape bias is also zero so it stays zero).
        let kv_block = ratio as usize * width;
        for i in kv_block..2 * kv_block {
            state.state_kv[i] = (i - kv_block) as f32 * 0.5 + 1.0;
        }
        // Pre-write the same pattern to scores so we can verify both halves.
        for i in kv_block..2 * kv_block {
            state.state_score[i] = (i - kv_block) as f32 * 0.5 - 7.0;
        }

        let x = vec![0.0f32; n_embd];
        let mut out = [0.0f32; HEAD_DIM];
        let freqs = long_freqs();
        for pos in 0..4u32 {
            let _ =
                compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, pos, 0).unwrap();
        }

        // After the emit + slide, rows [0..ratio) and rows [ratio..2*ratio)
        // must match each other byte-for-byte (the C does row [0..ratio) <-
        // [ratio..2*ratio) then [ratio..2*ratio) <- [0..ratio)).
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

    #[test]
    fn decode_one_writes_state_row_at_pos_mod() {
        // Hand-compute pos_mod placement: zero weights mean kv/gate matmuls
        // are zero, but the ape bias makes the score row non-zero at the
        // ape slice indexed by pos_mod. Use pos = 1 on a ratio=128 layer
        // so the state row is `pos_mod = 1`.
        let n_embd = 8usize;
        let ratio = 128u32;
        let width = HEAD_DIM; // coff = 1
        let kv_b = zeros(n_embd * width);
        let gate_b = zeros(n_embd * width);
        // Ape: pos=0 row all-zero, pos=1 row a recognisable F16 seed.
        let mut ape_b = zeros(width * ratio as usize);
        // F16 1.5 = 0x3E00; place at row pos_mod=1, all `width` columns.
        let mark = 0x3E00u16.to_le_bytes();
        for j in 0..width {
            let off = (width + j) * 2;
            ape_b[off..off + 2].copy_from_slice(&mark);
        }
        let norm = vec![1.0f32; HEAD_DIM];
        let weights = CompressorLayerWeights {
            kv: zero_f16(&kv_b, n_embd, width),
            gate: zero_f16(&gate_b, n_embd, width),
            ape: zero_f16(&ape_b, width, ratio as usize),
            norm: &norm,
        };

        let mut state = CompressorState::new(ratio, 1024).unwrap();
        let x = vec![0.0f32; n_embd];
        let mut out = [0.0f32; HEAD_DIM];
        let freqs = long_freqs();
        // pos = 0 stamps row 0 (gate row 0 = 0, ape row 0 = 0).
        let _ = compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, 0, 0).unwrap();
        // pos = 1 stamps row 1 with ape value 1.5.
        let _ = compressor_decode_one(&mut out, &weights, &mut state, &x, &freqs, 1, 0).unwrap();

        // Row 0 score: gate matmul writes 0, ape row 0 is 0 → 0.0.
        for j in 0..width {
            assert_eq!(state.state_score[j], 0.0);
        }
        // Row 1 score = matmul (zero) + ape (1.5).
        for j in 0..width {
            assert!(
                (state.state_score[width + j] - 1.5).abs() < 1e-3,
                "row 1 dim {j}: {}",
                state.state_score[width + j],
            );
        }
        // No other row was touched.
        for r in 2..ratio as usize {
            for j in 0..width {
                assert_eq!(state.state_score[r * width + j], NEG_INF);
            }
        }
    }
}
