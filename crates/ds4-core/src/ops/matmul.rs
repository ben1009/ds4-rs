//! Matmul dispatch + per-dtype dot kernels.
//!
//! See rfcs/0002-forward-pass.md §3.8. Two signatures share the same dot
//! kernels:
//!
//! * [`matmul_row`] — single activation vector (decode path).
//! * [`matmul_batch`] — `M` activation rows against the same weight, so each weight block is
//!   dequant'd once per prompt in prefill rather than once per token.
//!
//! PR #1 implements only the `Q8_0` weight dtype. Later PRs add F16, IQ2_XXS,
//! Q2_K, IQ4_K, Q4_K by extending [`WeightView`] and the dispatch arm — no
//! signature changes.

use crate::quant::q8_0;

/// A weight matrix with shape `[out_features, in_features]` (row-major).
///
/// Storage depends on the dtype; consumers never see raw bytes directly.
#[derive(Clone, Copy, Debug)]
pub enum WeightView<'a> {
    /// Q8_0: `out_features × in_features / 32` blocks of 34 bytes each.
    /// Rows are laid out contiguously; each row is `in_features / 32` blocks.
    Q8_0 {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
}

impl WeightView<'_> {
    pub fn out_features(&self) -> usize {
        match self {
            Self::Q8_0 { out_features, .. } => *out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Q8_0 { in_features, .. } => *in_features,
        }
    }
}

/// `out[n] = sum_k weight[n, k] * act[k]`
///
/// Shapes: `weight = [N, K]`, `act = [K]`, `out = [N]`.
pub fn matmul_row(weight: WeightView<'_>, act: &[f32], out: &mut [f32]) {
    assert_eq!(
        act.len(),
        weight.in_features(),
        "matmul_row: act len {} != in_features {}",
        act.len(),
        weight.in_features(),
    );
    assert_eq!(
        out.len(),
        weight.out_features(),
        "matmul_row: out len {} != out_features {}",
        out.len(),
        weight.out_features(),
    );
    match weight {
        WeightView::Q8_0 {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_row_q8_0(bytes, out_features, in_features, act, out);
        }
    }
}

/// `out[m, n] = sum_k weight[n, k] * acts[m, k]`
///
/// Shapes: `weight = [N, K]`, `acts = [M, K]`, `out = [M, N]`.
///
/// Order is row-then-weight: for each weight row we sweep the M activation
/// rows, so each weight block is dequant'd once per call rather than once per
/// activation row.
pub fn matmul_batch(weight: WeightView<'_>, acts: &[f32], out: &mut [f32], m: usize) {
    let k = weight.in_features();
    let n = weight.out_features();
    assert_eq!(
        acts.len(),
        m * k,
        "matmul_batch: acts len {} != m*k = {}*{}",
        acts.len(),
        m,
        k,
    );
    assert_eq!(
        out.len(),
        m * n,
        "matmul_batch: out len {} != m*n = {}*{}",
        out.len(),
        m,
        n,
    );
    match weight {
        WeightView::Q8_0 {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_batch_q8_0(bytes, out_features, in_features, acts, out, m);
        }
    }
}

fn matmul_row_q8_0(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    act: &[f32],
    out: &mut [f32],
) {
    assert_eq!(
        in_features % q8_0::BLOCK_SIZE,
        0,
        "matmul_row_q8_0: in_features {in_features} not multiple of {}",
        q8_0::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / q8_0::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * q8_0::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_row_q8_0: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_row_q8_0: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );
    for n in 0..out_features {
        let row = &bytes[n * bytes_per_row..(n + 1) * bytes_per_row];
        out[n] = dot_q8_0_f32_row(row, act);
    }
}

/// Batched Q8_0 × f32 matmul: dequant each weight block's f16 scale exactly
/// once per call and sweep the M activation rows across it.
///
/// Arithmetic exactly mirrors the row kernel:
///   `out[m, n] = sum_over_blocks( d_b * sum_i( q_i[b] * acts[m, col_b + i] ) )`
/// so batch and row paths agree bit-for-bit (modulo intra-block integer
/// summation order, which is identical).
///
/// This is the RFC 0002 §3.8 `matmul_batch` contract — prefill pays the
/// f16→f32 conversion of each block's scale once and reuses the 32 i8
/// quants across all M rows, rather than redecoding the scale + quants for
/// every (row, block) pair.
fn matmul_batch_q8_0(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    acts: &[f32],
    out: &mut [f32],
    m: usize,
) {
    assert_eq!(
        in_features % q8_0::BLOCK_SIZE,
        0,
        "matmul_batch_q8_0: in_features {in_features} not multiple of {}",
        q8_0::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / q8_0::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * q8_0::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_batch_q8_0: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_batch_q8_0: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    out.fill(0.0);

    // Precompute per-row flat offsets once per call, with checked_mul so a
    // pathological (m, in_features, out_features) tuple fails fast instead of
    // silently wrapping usize. Also hoists `row * in_features` /
    // `row * out_features` out of the innermost loop.
    let mut act_row_off: Vec<usize> = Vec::with_capacity(m);
    let mut out_row_off: Vec<usize> = Vec::with_capacity(m);
    for row in 0..m {
        act_row_off.push(
            row.checked_mul(in_features)
                .expect("matmul_batch_q8_0: act row offset overflowed usize"),
        );
        out_row_off.push(
            row.checked_mul(out_features)
                .expect("matmul_batch_q8_0: out row offset overflowed usize"),
        );
    }

    for n in 0..out_features {
        let wrow = &bytes[n * bytes_per_row..(n + 1) * bytes_per_row];
        for (block_idx, block) in wrow.chunks_exact(q8_0::BYTES_PER_BLOCK).enumerate() {
            let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            // Borrow the 32 i8 quants directly; no f32 materialisation so the
            // per-row inner product stays bit-identical to the row kernel.
            let qs = &block[2..q8_0::BYTES_PER_BLOCK];
            let col_start = block_idx * q8_0::BLOCK_SIZE;
            for row in 0..m {
                let act_start = act_row_off[row] + col_start;
                let act_chunk = &acts[act_start..act_start + q8_0::BLOCK_SIZE];
                let mut acc = 0f32;
                for i in 0..q8_0::BLOCK_SIZE {
                    acc += (qs[i] as i8) as f32 * act_chunk[i];
                }
                out[out_row_off[row] + n] += d * acc;
            }
        }
    }
}

/// `sum_k weight[k] * act[k]` for a single Q8_0-quantised row.
///
/// `weight_row` is `K/32` blocks of 34 bytes; `act` is `K` f32s. Walks block-
/// by-block so each block's f16 scale and 32 i8 quants stay in L1.
fn dot_q8_0_f32_row(weight_row: &[u8], act: &[f32]) -> f32 {
    let k = act.len();
    debug_assert_eq!(
        weight_row.len(),
        (k / q8_0::BLOCK_SIZE) * q8_0::BYTES_PER_BLOCK,
    );
    let mut sum = 0f32;
    for (block, act_chunk) in weight_row
        .chunks_exact(q8_0::BYTES_PER_BLOCK)
        .zip(act.chunks_exact(q8_0::BLOCK_SIZE))
    {
        sum += dot_q8_0_f32_block(block.try_into().unwrap(), act_chunk.try_into().unwrap());
    }
    sum
}

/// One 32-wide Q8_0 block × 32-wide f32 chunk → f32 scalar.
///
/// `d * sum_i qs[i] * act[i]`, with `qs[i]` an i8 and `d` the f16 scale.
fn dot_q8_0_f32_block(block: &[u8; q8_0::BYTES_PER_BLOCK], act: &[f32; q8_0::BLOCK_SIZE]) -> f32 {
    let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut acc = 0f32;
    for i in 0..q8_0::BLOCK_SIZE {
        let q = block[2 + i] as i8;
        acc += q as f32 * act[i];
    }
    d * acc
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build one Q8_0 block with f16 scale 1.0 and given i8 quants.
    fn block_scale_one(quants: [i8; 32]) -> [u8; q8_0::BYTES_PER_BLOCK] {
        let mut b = [0u8; q8_0::BYTES_PER_BLOCK];
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
        for (i, q) in quants.iter().enumerate() {
            b[2 + i] = *q as u8;
        }
        b
    }

    // Helper: build a weight of shape [N, K] where every entry is `val`,
    // packed as Q8_0 with scale 1.0.
    fn weight_constant_i8(n: usize, k: usize, val: i8) -> Vec<u8> {
        assert_eq!(k % q8_0::BLOCK_SIZE, 0);
        let blocks_per_row = k / q8_0::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(n * blocks_per_row * q8_0::BYTES_PER_BLOCK);
        for _ in 0..n {
            for _ in 0..blocks_per_row {
                bytes.extend_from_slice(&block_scale_one([val; 32]));
            }
        }
        bytes
    }

    #[test]
    fn dot_block_all_ones_identity() {
        // quants all 1, scale 1.0, act all 1.0 => sum = 32
        let block = block_scale_one([1; 32]);
        let act = [1.0f32; 32];
        assert_eq!(dot_q8_0_f32_block(&block, &act), 32.0);
    }

    #[test]
    fn dot_block_mixed_sign() {
        let mut quants = [0i8; 32];
        quants[0] = 10;
        quants[1] = -5;
        let block = block_scale_one(quants);
        let mut act = [0f32; 32];
        act[0] = 3.0;
        act[1] = 2.0;
        // 1.0 * (10*3 + -5*2 + 0..) = 20
        assert_eq!(dot_q8_0_f32_block(&block, &act), 20.0);
    }

    #[test]
    fn matmul_row_constant_weight() {
        // W = [[2,2,2,...]] (1×32), act = [1,1,..,1] => out[0] = 64
        let bytes = weight_constant_i8(1, 32, 2);
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: 1,
            in_features: 32,
        };
        let act = vec![1f32; 32];
        let mut out = vec![0f32; 1];
        matmul_row(w, &act, &mut out);
        assert_eq!(out[0], 64.0);
    }

    #[test]
    fn matmul_row_multiple_blocks_and_outputs() {
        // 4 × 64 weight, all 1s; act = 0..64.
        // Each output row sums act => 64*63/2 = 2016.
        let n = 4;
        let k = 64;
        let bytes = weight_constant_i8(n, k, 1);
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act: Vec<f32> = (0..k).map(|i| i as f32).collect();
        let mut out = vec![0f32; n];
        matmul_row(w, &act, &mut out);
        let expect: f32 = act.iter().sum();
        for &v in &out {
            assert_eq!(v, expect);
        }
    }

    #[test]
    fn matmul_batch_matches_row_by_row() {
        // Verify matmul_batch with M rows == M calls to matmul_row.
        let n = 3;
        let k = 64;
        // W with structured values: row r has all quants = r + 1.
        let blocks_per_row = k / q8_0::BLOCK_SIZE;
        let mut bytes = Vec::new();
        for r in 0..n {
            for _ in 0..blocks_per_row {
                bytes.extend_from_slice(&block_scale_one([(r as i8) + 1; 32]));
            }
        }
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 5;
        let acts: Vec<f32> = (0..(m * k)).map(|i| ((i % 7) as f32) - 3.0).collect();

        // Reference: M calls to matmul_row.
        let mut ref_out = vec![0f32; m * n];
        for row in 0..m {
            let a = &acts[row * k..(row + 1) * k];
            let mut r = vec![0f32; n];
            matmul_row(w, a, &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        // Batch path.
        let mut batch_out = vec![0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(batch_out, ref_out);
    }

    #[test]
    fn matmul_batch_bit_exact_with_nonunit_scales() {
        // Row and batch paths must agree bit-for-bit even when block scales
        // are non-power-of-two f16 values — catches a reordering regression
        // where the batch path accumulates `sum((d*q)*a)` instead of the
        // row path's `d*sum(q*a)`.
        let n = 2;
        let k = 96; // 3 blocks per row, matches `scales.len()` below

        // Non-power-of-two f16 scales so the two formulations round
        // differently without a per-block rearrangement.
        // 0.3 ≈ 0x3533, -0.7 ≈ 0xB99A, 1.5 = 0x3E00
        let scales: [u16; 3] = [0x3533, 0xB99A, 0x3E00];

        let mut bytes = Vec::new();
        for r in 0..n {
            for (b, scale) in scales.iter().enumerate() {
                let mut block = [0u8; q8_0::BYTES_PER_BLOCK];
                block[0..2].copy_from_slice(&scale.to_le_bytes());
                for i in 0..q8_0::BLOCK_SIZE {
                    // Mix of positive/negative quants; varies per row + col.
                    let v = (r as i32 * 13 + b as i32 * 5 + i as i32 * 3).rem_euclid(256) - 128;
                    block[2 + i] = v as u8;
                }
                bytes.extend_from_slice(&block);
            }
        }
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };

        let m = 4;
        let acts: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.017 - 1.3).sin())
            .collect();

        let mut ref_out = vec![0f32; m * n];
        for row in 0..m {
            let mut r = vec![0f32; n];
            matmul_row(w, &acts[row * k..(row + 1) * k], &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        let mut batch_out = vec![0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(
            batch_out, ref_out,
            "batch and row paths must be bit-identical across non-unit scales",
        );
    }

    #[test]
    #[should_panic(expected = "act len")]
    fn matmul_row_rejects_wrong_act_len() {
        let bytes = weight_constant_i8(1, 32, 1);
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: 1,
            in_features: 32,
        };
        let mut out = vec![0f32; 1];
        matmul_row(w, &[1.0; 16], &mut out);
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn matmul_row_rejects_wrong_out_len() {
        let bytes = weight_constant_i8(1, 32, 1);
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: 1,
            in_features: 32,
        };
        let mut out = vec![0f32; 2];
        matmul_row(w, &[1.0; 32], &mut out);
    }
}
