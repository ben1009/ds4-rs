//! Matmul dispatch + per-dtype dot kernels.
//!
//! See rfcs/0002-forward-pass.md §3.8. Two signatures share the same dot
//! kernels:
//!
//! * [`matmul_row`] — single activation vector (decode path).
//! * [`matmul_batch`] — `M` activation rows against the same weight, so each weight block is
//!   dequant'd once per prompt in prefill rather than once per token.
//!
//! Implemented weight dtypes cover Q8_0, F16, Q2_K, IQ2_XXS, Q4_K, IQ4_XS,
//! and IQ4_NL.

use crate::quant::{iq2_xxs, iq4_nl, iq4_xs, q2_k, q4_k, q8_0, q8_k};

/// A weight matrix with shape `[out_features, in_features]` (row-major).
///
/// Storage depends on the dtype; consumers never see raw bytes directly.
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub enum WeightView<'a> {
    /// Q8_0: `out_features × in_features / 32` blocks of 34 bytes each.
    /// Rows are laid out contiguously; each row is `in_features / 32` blocks.
    Q8_0 {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    /// F16: `out_features × in_features` little-endian f16 values, 2 bytes each.
    F16 {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    /// Q2_K: `out_features × in_features / 256` blocks of 84 bytes each.
    /// K-quant 2-bit weights; matmul quantises activation to Q8_K first.
    Q2_K {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    /// IQ2_XXS: `out_features × in_features / 256` blocks of 66 bytes each.
    /// Importance-quant 2-bit weights; matmul quantises activation to Q8_K first.
    IQ2_XXS {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    /// Q4_K: `out_features × in_features / 256` blocks of 144 bytes each.
    /// K-quant 4-bit weights with 6-bit super-block scales/mins; matmul
    /// quantises activation to Q8_K first.
    Q4_K {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    /// IQ4_XS: `out_features × in_features / 256` blocks of 136 bytes each.
    /// Importance-quant 4-bit weights with a 16-entry signed codebook; matmul
    /// quantises activation to Q8_K first.
    IQ4_XS {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    /// IQ4_NL: `out_features × in_features / 32` blocks of 18 bytes each.
    /// Importance-quant 4-bit weights with a 16-entry signed codebook; matmul
    /// dots against f32 activations directly (no Q8_K pre-quantisation).
    IQ4_NL {
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    /// Placeholder for dtypes not yet supported by the matmul kernels.
    /// Model loading succeeds, but calling matmul on this variant panics.
    Unsupported {
        dtype_name: &'static str,
        out_features: usize,
        in_features: usize,
    },
}

impl WeightView<'_> {
    pub fn out_features(&self) -> usize {
        match self {
            Self::Q8_0 { out_features, .. }
            | Self::F16 { out_features, .. }
            | Self::Q2_K { out_features, .. }
            | Self::IQ2_XXS { out_features, .. }
            | Self::Q4_K { out_features, .. }
            | Self::IQ4_XS { out_features, .. }
            | Self::IQ4_NL { out_features, .. }
            | Self::Unsupported { out_features, .. } => *out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Q8_0 { in_features, .. }
            | Self::F16 { in_features, .. }
            | Self::Q2_K { in_features, .. }
            | Self::IQ2_XXS { in_features, .. }
            | Self::Q4_K { in_features, .. }
            | Self::IQ4_XS { in_features, .. }
            | Self::IQ4_NL { in_features, .. }
            | Self::Unsupported { in_features, .. } => *in_features,
        }
    }

    /// Byte size of one expert's slice for a 3D routed-expert tensor.
    ///
    /// `out_features` and `in_features` are the per-expert dimensions
    /// (the loader reads `dims[0]` and `dims[1]` and drops the expert axis
    /// from `dims[2]`). The full tensor's bytes are an outer-stride
    /// concatenation of `n_expert` per-expert blocks of this size.
    ///
    /// Multiplications are `checked_mul`'d so a pathological dimension tuple
    /// fails fast instead of silently wrapping `usize`.
    fn per_expert_bytes(&self) -> usize {
        let blocks = |out_features: usize, in_features: usize, block: usize, bytes: usize| {
            out_features
                .checked_mul(in_features)
                .and_then(|n| n.checked_div(block))
                .and_then(|n| n.checked_mul(bytes))
                .expect("per_expert_bytes: dimension overflow")
        };
        match *self {
            Self::Q8_0 {
                out_features,
                in_features,
                ..
            } => blocks(
                out_features,
                in_features,
                q8_0::BLOCK_SIZE,
                q8_0::BYTES_PER_BLOCK,
            ),
            Self::F16 {
                out_features,
                in_features,
                ..
            } => out_features
                .checked_mul(in_features)
                .and_then(|n| n.checked_mul(2))
                .expect("per_expert_bytes: F16 dimension overflow"),
            Self::Q2_K {
                out_features,
                in_features,
                ..
            } => blocks(
                out_features,
                in_features,
                q2_k::BLOCK_SIZE,
                q2_k::BYTES_PER_BLOCK,
            ),
            Self::IQ2_XXS {
                out_features,
                in_features,
                ..
            } => blocks(
                out_features,
                in_features,
                iq2_xxs::BLOCK_SIZE,
                iq2_xxs::BYTES_PER_BLOCK,
            ),
            Self::Q4_K {
                out_features,
                in_features,
                ..
            } => blocks(
                out_features,
                in_features,
                q4_k::BLOCK_SIZE,
                q4_k::BYTES_PER_BLOCK,
            ),
            Self::IQ4_XS {
                out_features,
                in_features,
                ..
            } => blocks(
                out_features,
                in_features,
                iq4_xs::BLOCK_SIZE,
                iq4_xs::BYTES_PER_BLOCK,
            ),
            Self::IQ4_NL {
                out_features,
                in_features,
                ..
            } => blocks(
                out_features,
                in_features,
                iq4_nl::BLOCK_SIZE,
                iq4_nl::BYTES_PER_BLOCK,
            ),
            Self::Unsupported { .. } => 0,
        }
    }
}

/// Build a per-expert sub-view from a 3D routed-expert weight tensor.
///
/// The DS4 routed-expert tensors (`ffn_gate_exps`, `ffn_up_exps`,
/// `ffn_down_exps`) are stored as `n_expert` contiguous matrices, each
/// `[out_features, in_features]` in this codebase's convention. The per-expert
/// byte stride is `out_features * in_features / block_size * bytes_per_block`
/// (or `out_features * in_features * 2` for F16).
///
/// The returned view points at one expert's matrix and can be fed directly
/// into [`matmul_row`] / [`matmul_batch`].
///
/// Panics if `expert_idx >= n_expert` or if the parent's byte length is not
/// `n_expert * per_expert_bytes` (i.e. the tensor is not actually a 3D
/// routed-expert pack).
pub fn expert_subview(
    parent: WeightView<'_>,
    expert_idx: usize,
    n_expert: usize,
) -> WeightView<'_> {
    assert!(
        expert_idx < n_expert,
        "expert_subview: expert_idx {expert_idx} >= n_expert {n_expert}",
    );
    let per_expert = parent.per_expert_bytes();
    let total_expected = n_expert
        .checked_mul(per_expert)
        .expect("expert_subview: total byte budget overflowed usize");
    let off = expert_idx
        .checked_mul(per_expert)
        .expect("expert_subview: expert offset overflowed usize");
    match parent {
        WeightView::Q8_0 {
            bytes,
            out_features,
            in_features,
        } => {
            assert_eq!(
                bytes.len(),
                total_expected,
                "expert_subview Q8_0: bytes len {} != n_expert ({n_expert}) * per_expert ({per_expert})",
                bytes.len(),
            );
            WeightView::Q8_0 {
                bytes: &bytes[off..off + per_expert],
                out_features,
                in_features,
            }
        }
        WeightView::F16 {
            bytes,
            out_features,
            in_features,
        } => {
            assert_eq!(
                bytes.len(),
                total_expected,
                "expert_subview F16: bytes len {} != n_expert ({n_expert}) * per_expert ({per_expert})",
                bytes.len(),
            );
            WeightView::F16 {
                bytes: &bytes[off..off + per_expert],
                out_features,
                in_features,
            }
        }
        WeightView::Q2_K {
            bytes,
            out_features,
            in_features,
        } => {
            assert_eq!(
                bytes.len(),
                total_expected,
                "expert_subview Q2_K: bytes len {} != n_expert ({n_expert}) * per_expert ({per_expert})",
                bytes.len(),
            );
            WeightView::Q2_K {
                bytes: &bytes[off..off + per_expert],
                out_features,
                in_features,
            }
        }
        WeightView::IQ2_XXS {
            bytes,
            out_features,
            in_features,
        } => {
            assert_eq!(
                bytes.len(),
                total_expected,
                "expert_subview IQ2_XXS: bytes len {} != n_expert ({n_expert}) * per_expert ({per_expert})",
                bytes.len(),
            );
            WeightView::IQ2_XXS {
                bytes: &bytes[off..off + per_expert],
                out_features,
                in_features,
            }
        }
        WeightView::Q4_K {
            bytes,
            out_features,
            in_features,
        } => {
            assert_eq!(
                bytes.len(),
                total_expected,
                "expert_subview Q4_K: bytes len {} != n_expert ({n_expert}) * per_expert ({per_expert})",
                bytes.len(),
            );
            WeightView::Q4_K {
                bytes: &bytes[off..off + per_expert],
                out_features,
                in_features,
            }
        }
        WeightView::IQ4_XS {
            bytes,
            out_features,
            in_features,
        } => {
            assert_eq!(
                bytes.len(),
                total_expected,
                "expert_subview IQ4_XS: bytes len {} != n_expert ({n_expert}) * per_expert ({per_expert})",
                bytes.len(),
            );
            WeightView::IQ4_XS {
                bytes: &bytes[off..off + per_expert],
                out_features,
                in_features,
            }
        }
        WeightView::IQ4_NL {
            bytes,
            out_features,
            in_features,
        } => {
            assert_eq!(
                bytes.len(),
                total_expected,
                "expert_subview IQ4_NL: bytes len {} != n_expert ({n_expert}) * per_expert ({per_expert})",
                bytes.len(),
            );
            WeightView::IQ4_NL {
                bytes: &bytes[off..off + per_expert],
                out_features,
                in_features,
            }
        }
        WeightView::Unsupported { dtype_name, .. } => {
            panic!("expert_subview: unsupported weight dtype {dtype_name}");
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
        WeightView::F16 {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_row_f16(bytes, out_features, in_features, act, out);
        }
        WeightView::Q2_K {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_row_q2_k(bytes, out_features, in_features, act, out);
        }
        WeightView::IQ2_XXS {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_row_iq2_xxs(bytes, out_features, in_features, act, out);
        }
        WeightView::Q4_K {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_row_q4_k(bytes, out_features, in_features, act, out);
        }
        WeightView::IQ4_XS {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_row_iq4_xs(bytes, out_features, in_features, act, out);
        }
        WeightView::IQ4_NL {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_row_iq4_nl(bytes, out_features, in_features, act, out);
        }
        WeightView::Unsupported { dtype_name, .. } => {
            panic!("matmul_row: unsupported weight dtype {dtype_name}");
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
        WeightView::F16 {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_batch_f16(bytes, out_features, in_features, acts, out, m);
        }
        WeightView::Q2_K {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_batch_q2_k(bytes, out_features, in_features, acts, out, m);
        }
        WeightView::IQ2_XXS {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_batch_iq2_xxs(bytes, out_features, in_features, acts, out, m);
        }
        WeightView::Q4_K {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_batch_q4_k(bytes, out_features, in_features, acts, out, m);
        }
        WeightView::IQ4_XS {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_batch_iq4_xs(bytes, out_features, in_features, acts, out, m);
        }
        WeightView::IQ4_NL {
            bytes,
            out_features,
            in_features,
        } => {
            matmul_batch_iq4_nl(bytes, out_features, in_features, acts, out, m);
        }
        WeightView::Unsupported { dtype_name, .. } => {
            panic!("matmul_batch: unsupported weight dtype {dtype_name}");
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

// ---------------------------------------------------------------------------
// F16 weight matmul
// ---------------------------------------------------------------------------

fn matmul_row_f16(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    act: &[f32],
    out: &mut [f32],
) {
    let expected_bytes = out_features
        .checked_mul(in_features)
        .and_then(|n| n.checked_mul(2))
        .expect("matmul_row_f16: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_row_f16: bytes len {} != {out_features}*{in_features}*2",
        bytes.len(),
    );

    let row_stride = in_features
        .checked_mul(2)
        .expect("matmul_row_f16: row stride overflowed usize");
    for (row_bytes, out_n) in bytes
        .chunks_exact(row_stride)
        .zip(out.iter_mut())
        .take(out_features)
    {
        *out_n = dot_f16_f32_row(row_bytes, act);
    }
}

fn matmul_batch_f16(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    acts: &[f32],
    out: &mut [f32],
    m: usize,
) {
    let expected_bytes = out_features
        .checked_mul(in_features)
        .and_then(|n| n.checked_mul(2))
        .expect("matmul_batch_f16: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_batch_f16: bytes len {} != {out_features}*{in_features}*2",
        bytes.len(),
    );
    debug_assert_eq!(acts.len(), m * in_features);
    debug_assert_eq!(out.len(), m * out_features);

    out.fill(0.0);

    let row_stride = in_features
        .checked_mul(2)
        .expect("matmul_batch_f16: row stride overflowed usize");

    for (n, row_bytes) in bytes
        .chunks_exact(row_stride)
        .enumerate()
        .take(out_features)
    {
        for (k, chunk) in row_bytes.chunks_exact(2).enumerate() {
            let w = q8_0::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            for (out_row, act_row) in out
                .chunks_exact_mut(out_features)
                .zip(acts.chunks_exact(in_features))
            {
                out_row[n] += w * act_row[k];
            }
        }
    }
}

/// `sum_k weight[k] * act[k]` for a single F16 row.
fn dot_f16_f32_row(weight_row: &[u8], act: &[f32]) -> f32 {
    let k = act.len();
    debug_assert_eq!(weight_row.len(), k * 2);
    let mut sum = 0.0f32;
    for (chunk, &a) in weight_row.chunks_exact(2).zip(act) {
        let w = q8_0::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        sum += w * a;
    }
    sum
}

// ---------------------------------------------------------------------------
// Q2_K weight matmul
// ---------------------------------------------------------------------------

fn matmul_row_q2_k(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    act: &[f32],
    out: &mut [f32],
) {
    assert!(
        in_features.is_multiple_of(q2_k::BLOCK_SIZE),
        "matmul_row_q2_k: in_features {in_features} not multiple of {}",
        q2_k::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / q2_k::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * q2_k::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_row_q2_k: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_row_q2_k: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    // Quantise the single activation row to Q8_K.
    let mut q8_bytes = vec![0u8; blocks_per_row * q8_k::BYTES_PER_BLOCK];
    q8_k::quantize(act, &mut q8_bytes);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        let mut sum = 0.0f32;
        for (block, q8_block) in wrow
            .chunks_exact(q2_k::BYTES_PER_BLOCK)
            .zip(q8_bytes.chunks_exact(q8_k::BYTES_PER_BLOCK))
        {
            sum += q2_k::dot_q2k_q8k_block(block.try_into().unwrap(), q8_block);
        }
        out[n] = sum;
    }
}

fn matmul_batch_q2_k(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    acts: &[f32],
    out: &mut [f32],
    m: usize,
) {
    assert!(
        in_features.is_multiple_of(q2_k::BLOCK_SIZE),
        "matmul_batch_q2_k: in_features {in_features} not multiple of {}",
        q2_k::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / q2_k::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * q2_k::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_batch_q2_k: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_batch_q2_k: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    // Quantise all M activation rows to Q8_K.
    let q8_row_bytes = blocks_per_row * q8_k::BYTES_PER_BLOCK;
    let mut q8_bytes = vec![0u8; m * q8_row_bytes];
    q8_k::quantize(acts, &mut q8_bytes);

    out.fill(0.0);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        for (block_idx, block) in wrow.chunks_exact(q2_k::BYTES_PER_BLOCK).enumerate() {
            let block_arr: &[u8; q2_k::BYTES_PER_BLOCK] = block.try_into().unwrap();
            let q8_block_offset = block_idx
                .checked_mul(q8_k::BYTES_PER_BLOCK)
                .expect("matmul_batch_q2_k: q8 block offset overflowed usize");
            for (out_row, q8_row) in out
                .chunks_exact_mut(out_features)
                .zip(q8_bytes.chunks_exact(q8_row_bytes))
            {
                let q8_block = &q8_row[q8_block_offset..q8_block_offset + q8_k::BYTES_PER_BLOCK];
                out_row[n] += q2_k::dot_q2k_q8k_block(block_arr, q8_block);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IQ2_XXS weight matmul
// ---------------------------------------------------------------------------

fn matmul_row_iq2_xxs(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    act: &[f32],
    out: &mut [f32],
) {
    assert!(
        in_features.is_multiple_of(iq2_xxs::BLOCK_SIZE),
        "matmul_row_iq2_xxs: in_features {in_features} not multiple of {}",
        iq2_xxs::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / iq2_xxs::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * iq2_xxs::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_row_iq2_xxs: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_row_iq2_xxs: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    let mut q8_bytes = vec![0u8; blocks_per_row * q8_k::BYTES_PER_BLOCK];
    q8_k::quantize(act, &mut q8_bytes);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        let mut sum = 0.0f32;
        for (block, q8_block) in wrow
            .chunks_exact(iq2_xxs::BYTES_PER_BLOCK)
            .zip(q8_bytes.chunks_exact(q8_k::BYTES_PER_BLOCK))
        {
            sum += iq2_xxs::dot_iq2xxs_q8k_block(block.try_into().unwrap(), q8_block);
        }
        out[n] = sum;
    }
}

fn matmul_batch_iq2_xxs(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    acts: &[f32],
    out: &mut [f32],
    m: usize,
) {
    assert!(
        in_features.is_multiple_of(iq2_xxs::BLOCK_SIZE),
        "matmul_batch_iq2_xxs: in_features {in_features} not multiple of {}",
        iq2_xxs::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / iq2_xxs::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * iq2_xxs::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_batch_iq2_xxs: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_batch_iq2_xxs: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    let q8_row_bytes = blocks_per_row * q8_k::BYTES_PER_BLOCK;
    let mut q8_bytes = vec![0u8; m * q8_row_bytes];
    q8_k::quantize(acts, &mut q8_bytes);

    out.fill(0.0);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        for (block_idx, block) in wrow.chunks_exact(iq2_xxs::BYTES_PER_BLOCK).enumerate() {
            let block_arr: &[u8; iq2_xxs::BYTES_PER_BLOCK] = block.try_into().unwrap();
            let q8_block_offset = block_idx
                .checked_mul(q8_k::BYTES_PER_BLOCK)
                .expect("matmul_batch_iq2_xxs: q8 block offset overflowed usize");
            for (out_row, q8_row) in out
                .chunks_exact_mut(out_features)
                .zip(q8_bytes.chunks_exact(q8_row_bytes))
            {
                let q8_block = &q8_row[q8_block_offset..q8_block_offset + q8_k::BYTES_PER_BLOCK];
                out_row[n] += iq2_xxs::dot_iq2xxs_q8k_block(block_arr, q8_block);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K weight matmul
// ---------------------------------------------------------------------------

fn matmul_row_q4_k(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    act: &[f32],
    out: &mut [f32],
) {
    assert!(
        in_features.is_multiple_of(q4_k::BLOCK_SIZE),
        "matmul_row_q4_k: in_features {in_features} not multiple of {}",
        q4_k::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / q4_k::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * q4_k::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_row_q4_k: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_row_q4_k: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    let mut q8_bytes = vec![0u8; blocks_per_row * q8_k::BYTES_PER_BLOCK];
    q8_k::quantize(act, &mut q8_bytes);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        let mut sum = 0.0f32;
        for (block, q8_block) in wrow
            .chunks_exact(q4_k::BYTES_PER_BLOCK)
            .zip(q8_bytes.chunks_exact(q8_k::BYTES_PER_BLOCK))
        {
            sum += q4_k::dot_q4k_q8k_block(block.try_into().unwrap(), q8_block);
        }
        out[n] = sum;
    }
}

fn matmul_batch_q4_k(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    acts: &[f32],
    out: &mut [f32],
    m: usize,
) {
    assert!(
        in_features.is_multiple_of(q4_k::BLOCK_SIZE),
        "matmul_batch_q4_k: in_features {in_features} not multiple of {}",
        q4_k::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / q4_k::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * q4_k::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_batch_q4_k: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_batch_q4_k: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    let q8_row_bytes = blocks_per_row * q8_k::BYTES_PER_BLOCK;
    let mut q8_bytes = vec![0u8; m * q8_row_bytes];
    q8_k::quantize(acts, &mut q8_bytes);

    out.fill(0.0);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        for (block_idx, block) in wrow.chunks_exact(q4_k::BYTES_PER_BLOCK).enumerate() {
            let block_arr: &[u8; q4_k::BYTES_PER_BLOCK] = block.try_into().unwrap();
            let q8_block_offset = block_idx
                .checked_mul(q8_k::BYTES_PER_BLOCK)
                .expect("matmul_batch_q4_k: q8 block offset overflowed usize");
            for (out_row, q8_row) in out
                .chunks_exact_mut(out_features)
                .zip(q8_bytes.chunks_exact(q8_row_bytes))
            {
                let q8_block = &q8_row[q8_block_offset..q8_block_offset + q8_k::BYTES_PER_BLOCK];
                out_row[n] += q4_k::dot_q4k_q8k_block(block_arr, q8_block);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IQ4_XS weight matmul
// ---------------------------------------------------------------------------

fn matmul_row_iq4_xs(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    act: &[f32],
    out: &mut [f32],
) {
    assert!(
        in_features.is_multiple_of(iq4_xs::BLOCK_SIZE),
        "matmul_row_iq4_xs: in_features {in_features} not multiple of {}",
        iq4_xs::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / iq4_xs::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * iq4_xs::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_row_iq4_xs: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_row_iq4_xs: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    let mut q8_bytes = vec![0u8; blocks_per_row * q8_k::BYTES_PER_BLOCK];
    q8_k::quantize(act, &mut q8_bytes);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        let mut sum = 0.0f32;
        for (block, q8_block) in wrow
            .chunks_exact(iq4_xs::BYTES_PER_BLOCK)
            .zip(q8_bytes.chunks_exact(q8_k::BYTES_PER_BLOCK))
        {
            sum += iq4_xs::dot_iq4xs_q8k_block(block.try_into().unwrap(), q8_block);
        }
        out[n] = sum;
    }
}

fn matmul_batch_iq4_xs(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    acts: &[f32],
    out: &mut [f32],
    m: usize,
) {
    assert!(
        in_features.is_multiple_of(iq4_xs::BLOCK_SIZE),
        "matmul_batch_iq4_xs: in_features {in_features} not multiple of {}",
        iq4_xs::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / iq4_xs::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * iq4_xs::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_batch_iq4_xs: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_batch_iq4_xs: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    let q8_row_bytes = blocks_per_row * q8_k::BYTES_PER_BLOCK;
    let mut q8_bytes = vec![0u8; m * q8_row_bytes];
    q8_k::quantize(acts, &mut q8_bytes);

    out.fill(0.0);

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        for (block_idx, block) in wrow.chunks_exact(iq4_xs::BYTES_PER_BLOCK).enumerate() {
            let block_arr: &[u8; iq4_xs::BYTES_PER_BLOCK] = block.try_into().unwrap();
            let q8_block_offset = block_idx
                .checked_mul(q8_k::BYTES_PER_BLOCK)
                .expect("matmul_batch_iq4_xs: q8 block offset overflowed usize");
            for (out_row, q8_row) in out
                .chunks_exact_mut(out_features)
                .zip(q8_bytes.chunks_exact(q8_row_bytes))
            {
                let q8_block = &q8_row[q8_block_offset..q8_block_offset + q8_k::BYTES_PER_BLOCK];
                out_row[n] += iq4_xs::dot_iq4xs_q8k_block(block_arr, q8_block);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IQ4_NL weight matmul (32-element blocks, f32 activations)
// ---------------------------------------------------------------------------

fn matmul_row_iq4_nl(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    act: &[f32],
    out: &mut [f32],
) {
    assert!(
        in_features.is_multiple_of(iq4_nl::BLOCK_SIZE),
        "matmul_row_iq4_nl: in_features {in_features} not multiple of {}",
        iq4_nl::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / iq4_nl::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * iq4_nl::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_row_iq4_nl: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_row_iq4_nl: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        let mut sum = 0.0f32;
        for (block, act_chunk) in wrow
            .chunks_exact(iq4_nl::BYTES_PER_BLOCK)
            .zip(act.chunks_exact(iq4_nl::BLOCK_SIZE))
        {
            sum += iq4_nl::dot_iq4nl_f32_block(
                block.try_into().unwrap(),
                act_chunk.try_into().unwrap(),
            );
        }
        out[n] = sum;
    }
}

fn matmul_batch_iq4_nl(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    acts: &[f32],
    out: &mut [f32],
    _m: usize,
) {
    assert!(
        in_features.is_multiple_of(iq4_nl::BLOCK_SIZE),
        "matmul_batch_iq4_nl: in_features {in_features} not multiple of {}",
        iq4_nl::BLOCK_SIZE,
    );
    let blocks_per_row = in_features / iq4_nl::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * iq4_nl::BYTES_PER_BLOCK;
    let expected_bytes = out_features
        .checked_mul(bytes_per_row)
        .expect("matmul_batch_iq4_nl: bytes budget overflowed usize");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "matmul_batch_iq4_nl: bytes len {} != {out_features} * {bytes_per_row}",
        bytes.len(),
    );

    out.fill(0.0);

    // Dequantise each weight block once into a 32-wide f32 buffer, then sweep
    // the M activation rows against it. Mirrors the Q8_0 batch shape so each
    // block's f16 scale + 16 codebook lookups are paid once per call.
    let mut deq = [0.0f32; iq4_nl::BLOCK_SIZE];
    for (n, wrow) in bytes
        .chunks_exact(bytes_per_row)
        .enumerate()
        .take(out_features)
    {
        for (block_idx, block) in wrow.chunks_exact(iq4_nl::BYTES_PER_BLOCK).enumerate() {
            iq4_nl::dequant_block(block.try_into().unwrap(), &mut deq);
            let col_start = block_idx * iq4_nl::BLOCK_SIZE;
            for (out_row, act_row) in out
                .chunks_exact_mut(out_features)
                .zip(acts.chunks_exact(in_features))
            {
                let act_chunk = &act_row[col_start..col_start + iq4_nl::BLOCK_SIZE];
                let mut acc = 0.0f32;
                for i in 0..iq4_nl::BLOCK_SIZE {
                    acc += deq[i] * act_chunk[i];
                }
                out_row[n] += acc;
            }
        }
    }
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

    // Helper: build F16 bytes for a weight matrix where every entry is `val`.
    fn weight_constant_f16(n: usize, k: usize, val: f32) -> Vec<u8> {
        // Hard-coded f16 bit patterns for common test values.
        let bits: u16 = if val == 1.0 {
            0x3C00
        } else if val == 2.0 {
            0x4000
        } else if val == 0.5 {
            0x3800
        } else if val == -1.0 {
            0xBC00
        } else {
            panic!("weight_constant_f16: unhandled test value {val}")
        };
        let mut bytes = Vec::with_capacity(n * k * 2);
        for _ in 0..(n * k) {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn matmul_row_f16_basic() {
        // 2 × 3 weight: [[1, 1, 1], [2, 2, 2]]
        let bytes = weight_constant_f16(2, 3, 1.0);
        // Overwrite second row to be 2.0.
        let mut bytes = bytes;
        for b in bytes[3 * 2..].chunks_exact_mut(2) {
            b.copy_from_slice(&0x4000u16.to_le_bytes());
        }
        let w = WeightView::F16 {
            bytes: &bytes,
            out_features: 2,
            in_features: 3,
        };
        let act = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 2];
        matmul_row(w, &act, &mut out);
        assert_eq!(out[0], 6.0); // 1*1 + 1*2 + 1*3
        assert_eq!(out[1], 12.0); // 2*1 + 2*2 + 2*3
    }

    #[test]
    fn matmul_batch_f16_matches_row_by_row() {
        let n = 3;
        let k = 8;
        // Each row r has value (r + 1) as f16.
        let mut bytes = Vec::with_capacity(n * k * 2);
        for r in 0..n {
            let _bits = ((r + 1) as f32).to_bits();
            // Actually we need f16 bits. For small integers, f16 representation:
            // 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200
            let bits16: u16 = match r + 1 {
                1 => 0x3C00,
                2 => 0x4000,
                3 => 0x4200,
                _ => unreachable!(),
            };
            for _ in 0..k {
                bytes.extend_from_slice(&bits16.to_le_bytes());
            }
        }
        let w = WeightView::F16 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 4;
        let acts: Vec<f32> = (0..(m * k)).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut ref_out = vec![0.0f32; m * n];
        for row in 0..m {
            let mut r = vec![0.0f32; n];
            matmul_row(w, &acts[row * k..(row + 1) * k], &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        let mut batch_out = vec![0.0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(batch_out, ref_out);
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

    #[test]
    fn weight_view_accessors() {
        let bytes = weight_constant_i8(7, 64, 1);
        let q = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: 7,
            in_features: 64,
        };
        assert_eq!(q.out_features(), 7);
        assert_eq!(q.in_features(), 64);

        let f_bytes = weight_constant_f16(3, 5, 1.0);
        let f = WeightView::F16 {
            bytes: &f_bytes,
            out_features: 3,
            in_features: 5,
        };
        assert_eq!(f.out_features(), 3);
        assert_eq!(f.in_features(), 5);
    }

    #[test]
    fn dot_block_negative_scale() {
        // f16 -1.0 = 0xBC00
        let mut block = block_scale_one([1; 32]);
        block[0..2].copy_from_slice(&0xBC00u16.to_le_bytes());
        let act = [1.0f32; 32];
        assert_eq!(dot_q8_0_f32_block(&block, &act), -32.0);
    }

    #[test]
    fn dot_block_zero_quants_is_zero() {
        let block = block_scale_one([0; 32]);
        let act = [3.5f32; 32];
        assert_eq!(dot_q8_0_f32_block(&block, &act), 0.0);
    }

    #[test]
    fn dot_block_zero_act_is_zero() {
        let block = block_scale_one([7; 32]);
        let act = [0.0f32; 32];
        assert_eq!(dot_q8_0_f32_block(&block, &act), 0.0);
    }

    #[test]
    fn matmul_batch_m_one_matches_row() {
        let n = 4;
        let k = 64;
        let bytes = weight_constant_i8(n, k, 2);
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.25 - 1.0).collect();

        let mut row_out = vec![0f32; n];
        matmul_row(w, &act, &mut row_out);

        let mut batch_out = vec![0f32; n];
        matmul_batch(w, &act, &mut batch_out, 1);

        assert_eq!(row_out, batch_out);
    }

    #[test]
    fn matmul_row_zero_act_yields_zero() {
        let n = 3;
        let k = 32;
        let bytes = weight_constant_i8(n, k, 5);
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act = vec![0.0f32; k];
        let mut out = vec![123.0f32; n];
        matmul_row(w, &act, &mut out);
        assert_eq!(out, vec![0.0f32; n]);
    }

    #[test]
    fn matmul_batch_overwrites_existing_out() {
        let n = 2;
        let k = 32;
        let bytes = weight_constant_i8(n, k, 1);
        let w = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 3;
        let acts = vec![1.0f32; m * k];

        let mut out = vec![999.0f32; m * n];
        matmul_batch(w, &acts, &mut out, m);

        for &v in &out {
            assert_eq!(v, 32.0);
        }
    }

    #[test]
    fn matmul_row_f16_negative_weights() {
        let bytes = weight_constant_f16(1, 4, -1.0);
        let w = WeightView::F16 {
            bytes: &bytes,
            out_features: 1,
            in_features: 4,
        };
        let act = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 1];
        matmul_row(w, &act, &mut out);
        assert_eq!(out[0], -10.0);
    }

    #[test]
    fn matmul_row_f16_fractional_weights() {
        let bytes = weight_constant_f16(1, 4, 0.5);
        let w = WeightView::F16 {
            bytes: &bytes,
            out_features: 1,
            in_features: 4,
        };
        let act = vec![2.0f32, 4.0, 6.0, 8.0];
        let mut out = vec![0.0f32; 1];
        matmul_row(w, &act, &mut out);
        assert_eq!(out[0], 10.0);
    }

    #[test]
    fn matmul_batch_m_one_f16_matches_row() {
        let n = 2;
        let k = 6;
        let bytes = weight_constant_f16(n, k, 2.0);
        let w = WeightView::F16 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.5 - 1.0).collect();

        let mut row_out = vec![0.0f32; n];
        matmul_row(w, &act, &mut row_out);

        let mut batch_out = vec![0.0f32; n];
        matmul_batch(w, &act, &mut batch_out, 1);

        assert_eq!(row_out, batch_out);
    }

    #[test]
    #[should_panic(expected = "act len")]
    fn matmul_row_f16_rejects_wrong_act_len() {
        let bytes = weight_constant_f16(1, 4, 1.0);
        let w = WeightView::F16 {
            bytes: &bytes,
            out_features: 1,
            in_features: 4,
        };
        let mut out = vec![0.0f32; 1];
        matmul_row(w, &[1.0; 8], &mut out);
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn matmul_row_f16_rejects_wrong_out_len() {
        let bytes = weight_constant_f16(1, 4, 1.0);
        let w = WeightView::F16 {
            bytes: &bytes,
            out_features: 1,
            in_features: 4,
        };
        let mut out = vec![0.0f32; 5];
        matmul_row(w, &[1.0; 4], &mut out);
    }

    // -----------------------------------------------------------------------
    // Q2_K integration tests
    // -----------------------------------------------------------------------

    /// Build Q2_K bytes where every weight is `val`.
    ///
    /// `val` must be representable as `d * scale * q` with integer q in 0..3.
    /// For unit tests we only need 0.0 or 1.0.
    fn weight_constant_q2_k(n: usize, k: usize, val: f32) -> Vec<u8> {
        assert_eq!(k % q2_k::BLOCK_SIZE, 0, "k must be multiple of 256");
        let blocks_per_row = k / q2_k::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(n * blocks_per_row * q2_k::BYTES_PER_BLOCK);
        for _ in 0..n {
            for _ in 0..blocks_per_row {
                let mut block = [0u8; q2_k::BYTES_PER_BLOCK];
                if val == 1.0 {
                    // d = 1.0, dmin = 0.0, scale = 1, min = 0, q = 1
                    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
                    for b in &mut block[4..20] {
                        *b = 0x01; // scale=1, min=0
                    }
                    for b in &mut block[20..84] {
                        *b = 0x55; // four quant=1 values per byte
                    }
                } else if val == 0.0 {
                    // all zeros -> all weights 0.0
                } else {
                    panic!("weight_constant_q2_k: unhandled test value {val}");
                }
                bytes.extend_from_slice(&block);
            }
        }
        bytes
    }

    #[test]
    fn matmul_row_q2_k_basic() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_q2_k(n, k, 1.0);
        let w = WeightView::Q2_K {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act: Vec<f32> = (0..k).map(|i| (i % 4) as f32).collect();
        // sum of act = 64 * (0+1+2+3) = 384 per row
        let mut out = vec![0.0f32; n];
        matmul_row(w, &act, &mut out);
        assert!((out[0] - 384.0).abs() < 0.5, "out[0] = {} != 384", out[0]);
        assert!((out[1] - 384.0).abs() < 0.5, "out[1] = {} != 384", out[1]);
    }

    #[test]
    fn matmul_batch_q2_k_matches_row_by_row() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_q2_k(n, k, 1.0);
        let w = WeightView::Q2_K {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 3;
        let acts: Vec<f32> = (0..(m * k)).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut ref_out = vec![0.0f32; m * n];
        for row in 0..m {
            let mut r = vec![0.0f32; n];
            matmul_row(w, &acts[row * k..(row + 1) * k], &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        let mut batch_out = vec![0.0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(batch_out, ref_out);
    }

    #[test]
    fn matmul_row_q2_k_zero_act_yields_zero() {
        let n = 3;
        let k = 256;
        let bytes = weight_constant_q2_k(n, k, 1.0);
        let w = WeightView::Q2_K {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let mut out = vec![123.0f32; n];
        matmul_row(w, &[0.0f32; 256], &mut out);
        assert_eq!(out, vec![0.0f32; n]);
    }

    // -----------------------------------------------------------------------
    // IQ2_XXS integration tests
    // -----------------------------------------------------------------------

    /// Build IQ2_XXS bytes where every weight is `val`.
    fn weight_constant_iq2_xxs(n: usize, k: usize, val: f32) -> Vec<u8> {
        assert_eq!(k % iq2_xxs::BLOCK_SIZE, 0, "k must be multiple of 256");
        let blocks_per_row = k / iq2_xxs::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(n * blocks_per_row * iq2_xxs::BYTES_PER_BLOCK);
        for _ in 0..n {
            for _ in 0..blocks_per_row {
                let mut block = [0u8; iq2_xxs::BYTES_PER_BLOCK];
                if val == 1.0 {
                    // d = 1.0, grid=0, sign=0, scale=0 -> weight = 1.0
                    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
                    // qs[32] all zero -> grid 0, sign 0, scale 0
                } else if val == 0.0 {
                    // d = 0.0 -> all weights 0.0
                } else {
                    panic!("weight_constant_iq2_xxs: unhandled test value {val}");
                }
                bytes.extend_from_slice(&block);
            }
        }
        bytes
    }

    #[test]
    fn matmul_row_iq2_xxs_basic() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_iq2_xxs(n, k, 1.0);
        let w = WeightView::IQ2_XXS {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act: Vec<f32> = (0..k).map(|i| (i % 4) as f32).collect();
        let mut out = vec![0.0f32; n];
        matmul_row(w, &act, &mut out);
        assert!((out[0] - 384.0).abs() < 0.5, "out[0] = {} != 384", out[0]);
        assert!((out[1] - 384.0).abs() < 0.5, "out[1] = {} != 384", out[1]);
    }

    #[test]
    fn matmul_batch_iq2_xxs_matches_row_by_row() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_iq2_xxs(n, k, 1.0);
        let w = WeightView::IQ2_XXS {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 3;
        let acts: Vec<f32> = (0..(m * k)).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut ref_out = vec![0.0f32; m * n];
        for row in 0..m {
            let mut r = vec![0.0f32; n];
            matmul_row(w, &acts[row * k..(row + 1) * k], &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        let mut batch_out = vec![0.0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(batch_out, ref_out);
    }

    #[test]
    fn matmul_row_iq2_xxs_zero_act_yields_zero() {
        let n = 3;
        let k = 256;
        let bytes = weight_constant_iq2_xxs(n, k, 1.0);
        let w = WeightView::IQ2_XXS {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let mut out = vec![123.0f32; n];
        matmul_row(w, &[0.0f32; 256], &mut out);
        assert_eq!(out, vec![0.0f32; n]);
    }

    // -----------------------------------------------------------------------
    // Q4_K integration tests
    // -----------------------------------------------------------------------

    /// Q4_K bytes where every weight = 1.0 across all 8 sub-blocks.
    fn weight_constant_q4_k_one(n: usize, k: usize) -> Vec<u8> {
        use crate::quant::q4_k;
        assert_eq!(k % q4_k::BLOCK_SIZE, 0);
        let blocks_per_row = k / q4_k::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(n * blocks_per_row * q4_k::BYTES_PER_BLOCK);
        for _ in 0..n {
            for _ in 0..blocks_per_row {
                let mut block = [0u8; q4_k::BYTES_PER_BLOCK];
                // d=1.0
                block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
                // scales: encode (sc=1, m=0) for every sub-block.
                // (See dequant_block_unit_scale_no_min in q4_k.rs.)
                for s in &mut block[q4_k::offset::SCALES..q4_k::offset::SCALES + 4] {
                    *s = 0x01;
                }
                for s in &mut block[q4_k::offset::SCALES + 8..q4_k::offset::SCALES + 12] {
                    *s = 0x01;
                }
                // qs all 0x11 -> low and high nibbles = 1
                for q in &mut block[q4_k::offset::QS..q4_k::offset::QS + 128] {
                    *q = 0x11;
                }
                bytes.extend_from_slice(&block);
            }
        }
        bytes
    }

    #[test]
    fn matmul_row_q4_k_basic() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_q4_k_one(n, k);
        let w = WeightView::Q4_K {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act: Vec<f32> = (0..k).map(|i| (i % 4) as f32).collect();
        // sum = 64 * (0+1+2+3) = 384 per row
        let mut out = vec![0.0f32; n];
        matmul_row(w, &act, &mut out);
        assert!((out[0] - 384.0).abs() < 0.5);
        assert!((out[1] - 384.0).abs() < 0.5);
    }

    #[test]
    fn matmul_batch_q4_k_matches_row_by_row() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_q4_k_one(n, k);
        let w = WeightView::Q4_K {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 3;
        let acts: Vec<f32> = (0..(m * k)).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut ref_out = vec![0.0f32; m * n];
        for row in 0..m {
            let mut r = vec![0.0f32; n];
            matmul_row(w, &acts[row * k..(row + 1) * k], &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        let mut batch_out = vec![0.0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(batch_out, ref_out);
    }

    #[test]
    fn matmul_row_q4_k_zero_act_yields_zero() {
        let n = 3;
        let k = 256;
        let bytes = weight_constant_q4_k_one(n, k);
        let w = WeightView::Q4_K {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let mut out = vec![123.0f32; n];
        matmul_row(w, &[0.0f32; 256], &mut out);
        assert_eq!(out, vec![0.0f32; n]);
    }

    // -----------------------------------------------------------------------
    // IQ4_XS integration tests
    // -----------------------------------------------------------------------

    /// IQ4_XS bytes where every weight = -127.0 (codebook[0] with signed scale 1).
    fn weight_constant_iq4_xs_neg127(n: usize, k: usize) -> Vec<u8> {
        use crate::quant::iq4_xs;
        assert_eq!(k % iq4_xs::BLOCK_SIZE, 0);
        let blocks_per_row = k / iq4_xs::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(n * blocks_per_row * iq4_xs::BYTES_PER_BLOCK);
        for _ in 0..n {
            for _ in 0..blocks_per_row {
                let mut block = [0u8; iq4_xs::BYTES_PER_BLOCK];
                // d=1.0
                block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
                // scales_h = 0xAAAA (every sub-block raw scale high2 = 0b10),
                // scales_l = 0x11 each (low4 = 1) -> raw = 0x21 = 33 -> signed = 1.
                block[2..4].copy_from_slice(&0xAAAAu16.to_le_bytes());
                for s in &mut block[iq4_xs::offset::SCALES_L..iq4_xs::offset::SCALES_L + 4] {
                    *s = 0x11;
                }
                // qs all zero -> codebook[0] = -127 in every position
                bytes.extend_from_slice(&block);
            }
        }
        bytes
    }

    #[test]
    fn matmul_row_iq4_xs_basic() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_iq4_xs_neg127(n, k);
        let w = WeightView::IQ4_XS {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act = vec![1.0f32; k];
        // Each weight = -127, so dot = -127 * 256 = -32512.
        let mut out = vec![0.0f32; n];
        matmul_row(w, &act, &mut out);
        // Q8_K quantises act=[1; 256] losslessly only for the max-abs element;
        // the others round to -127/128. Tolerance proportional to that step.
        for &v in &out {
            assert!((v - -32512.0).abs() < 32512.0 / 128.0 + 1.0, "{v}");
        }
    }

    #[test]
    fn matmul_batch_iq4_xs_matches_row_by_row() {
        let n = 2;
        let k = 256;
        let bytes = weight_constant_iq4_xs_neg127(n, k);
        let w = WeightView::IQ4_XS {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 3;
        let acts: Vec<f32> = (0..(m * k)).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut ref_out = vec![0.0f32; m * n];
        for row in 0..m {
            let mut r = vec![0.0f32; n];
            matmul_row(w, &acts[row * k..(row + 1) * k], &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        let mut batch_out = vec![0.0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(batch_out, ref_out);
    }

    #[test]
    fn matmul_row_iq4_xs_zero_act_yields_zero() {
        let n = 3;
        let k = 256;
        let bytes = weight_constant_iq4_xs_neg127(n, k);
        let w = WeightView::IQ4_XS {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let mut out = vec![123.0f32; n];
        matmul_row(w, &[0.0f32; 256], &mut out);
        assert_eq!(out, vec![0.0f32; n]);
    }

    // -----------------------------------------------------------------------
    // IQ4_NL integration tests
    // -----------------------------------------------------------------------

    /// IQ4_NL bytes where every weight = -127.0 (codebook[0] with d=1.0).
    fn weight_constant_iq4_nl_neg127(n: usize, k: usize) -> Vec<u8> {
        use crate::quant::iq4_nl;
        assert_eq!(k % iq4_nl::BLOCK_SIZE, 0);
        let blocks_per_row = k / iq4_nl::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(n * blocks_per_row * iq4_nl::BYTES_PER_BLOCK);
        for _ in 0..n {
            for _ in 0..blocks_per_row {
                let mut block = [0u8; iq4_nl::BYTES_PER_BLOCK];
                block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
                // qs all zero -> all codebook indices = 0 = -127
                bytes.extend_from_slice(&block);
            }
        }
        bytes
    }

    #[test]
    fn matmul_row_iq4_nl_basic() {
        let n = 2;
        let k = 64;
        let bytes = weight_constant_iq4_nl_neg127(n, k);
        let w = WeightView::IQ4_NL {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act = vec![1.0f32; k];
        // Each weight = -127, dot = -127 * 64 = -8128.
        let mut out = vec![0.0f32; n];
        matmul_row(w, &act, &mut out);
        for &v in &out {
            assert_eq!(v, -8128.0);
        }
    }

    #[test]
    fn matmul_batch_iq4_nl_matches_row_by_row() {
        let n = 3;
        let k = 64;
        let bytes = weight_constant_iq4_nl_neg127(n, k);
        let w = WeightView::IQ4_NL {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let m = 4;
        let acts: Vec<f32> = (0..(m * k)).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut ref_out = vec![0.0f32; m * n];
        for row in 0..m {
            let mut r = vec![0.0f32; n];
            matmul_row(w, &acts[row * k..(row + 1) * k], &mut r);
            ref_out[row * n..(row + 1) * n].copy_from_slice(&r);
        }

        let mut batch_out = vec![0.0f32; m * n];
        matmul_batch(w, &acts, &mut batch_out, m);

        assert_eq!(batch_out, ref_out);
    }

    #[test]
    fn matmul_row_iq4_nl_zero_act_yields_zero() {
        let n = 3;
        let k = 64;
        let bytes = weight_constant_iq4_nl_neg127(n, k);
        let w = WeightView::IQ4_NL {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let mut out = vec![123.0f32; n];
        matmul_row(w, &vec![0.0f32; k], &mut out);
        assert_eq!(out, vec![0.0f32; n]);
    }

    // -----------------------------------------------------------------------
    // expert_subview
    // -----------------------------------------------------------------------

    #[test]
    fn expert_subview_q8_0_yields_distinct_per_expert_results() {
        // Build 3 "experts": each a 1×32 Q8_0 matrix where every quant equals
        // (eid + 1). Pulling out expert eid and dotting against [1; 32]
        // should yield 32 * (eid + 1).
        let n_expert = 3;
        let n = 1;
        let k = 32;
        let mut bytes = Vec::new();
        for eid in 0..n_expert {
            let v = (eid as i8) + 1;
            bytes.extend_from_slice(&block_scale_one([v; 32]));
        }
        let parent = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: n,
            in_features: k,
        };
        let act = vec![1.0f32; k];
        for eid in 0..n_expert {
            let sub = expert_subview(parent, eid, n_expert);
            let mut out = vec![0.0f32; n];
            matmul_row(sub, &act, &mut out);
            assert_eq!(out[0], 32.0 * (eid as f32 + 1.0));
        }
    }

    #[test]
    #[should_panic(expected = "expert_idx")]
    fn expert_subview_rejects_out_of_range_eid() {
        let bytes = weight_constant_i8(1, 32, 1);
        let parent = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: 1,
            in_features: 32,
        };
        let _ = expert_subview(parent, 1, 1);
    }

    #[test]
    #[should_panic(expected = "expert_subview Q8_0")]
    fn expert_subview_rejects_byte_length_mismatch() {
        // n_expert=2 but bytes only cover 1 expert.
        let bytes = weight_constant_i8(1, 32, 1);
        let parent = WeightView::Q8_0 {
            bytes: &bytes,
            out_features: 1,
            in_features: 32,
        };
        let _ = expert_subview(parent, 0, 2);
    }
}
