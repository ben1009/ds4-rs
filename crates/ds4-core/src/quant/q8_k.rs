//! Q8_K activation pre-quantization.
//!
//! See rfcs/0002-forward-pass.md §3.2 + §3.8. DS4 Flash's IQ2_XXS / Q2_K /
//! IQ4_K / Q4_K weight matmuls operate on Q8_K-quantised activation rows
//! (the "block_q8_K" struct in ggml / antirez/ds4). This module writes one
//! 256-element activation row into the Q8_K block layout; later PRs that
//! implement the mixed-dtype dot kernels will call this first.
//!
//! Block layout (292 bytes, 256 elements):
//! * `d`      : f32 scale           (4 bytes, offset 0)
//! * `qs[i]`  : 256 × i8 quants     (256 bytes, offset 4)
//! * `bsums[j]`: 16 × i16           (32 bytes, offset 260) — partial sums of `qs[16j .. 16j+16]`,
//!   precomputed so mixed-dtype kernels don't have to re-fold them each call.
//!
//! Quantisation follows the ggml reference:
//! 1. Find the signed value `m` in the block with largest absolute value.
//! 2. If `m == 0`: zero the whole block.
//! 3. Otherwise: `iscale = -128.0 / m` `qs[i]  = min(127, round_ties_even(iscale * x[i]))`  — i8
//!    clamp `bsums[j] = sum_{k in 0..16} qs[16j + k]`            — i16 `d = 1.0 / iscale`  (so `d *
//!    qs[max_idx] ≈ m`, round-trip exact)
//!
//! Reference: `ds4_quantize_row_q8_K` in antirez/ds4 ds4.c line 1657.

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 292;

/// Byte offsets inside one Q8_K block.
pub mod offset {
    pub const D: usize = 0; // f32
    pub const QS: usize = 4; // i8 × 256
    pub const BSUMS: usize = 260; // i16 × 16
}

/// Quantise one 256-element f32 row into a 292-byte Q8_K block.
///
/// Panics if `x.len() != BLOCK_SIZE` or `out.len() != BYTES_PER_BLOCK`.
pub fn quantize_block(x: &[f32], out: &mut [u8]) {
    assert_eq!(
        x.len(),
        BLOCK_SIZE,
        "q8_k::quantize_block: x len {} != {BLOCK_SIZE}",
        x.len(),
    );
    assert_eq!(
        out.len(),
        BYTES_PER_BLOCK,
        "q8_k::quantize_block: out len {} != {BYTES_PER_BLOCK}",
        out.len(),
    );

    // Signed max by absolute value — preserves the sign so iscale can place
    // that element at -128 (matching the ggml convention, see module doc).
    let mut amax = 0.0f32;
    let mut m = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > amax {
            amax = av;
            m = v;
        }
    }

    if m == 0.0 {
        out.fill(0);
        return;
    }

    let iscale = -128.0f32 / m;

    // qs[i] = clamp_i8_upper(round_ties_even(iscale * x[i])).
    // Lower bound is saturated by the f32→i8 range (iscale * m = -128).
    for i in 0..BLOCK_SIZE {
        let v = round_ties_even_i32(iscale * x[i]);
        let clamped = v.min(127);
        out[offset::QS + i] = clamped as i8 as u8;
    }

    // bsums[j] = i16 partial sum of 16 consecutive qs entries.
    for j in 0..16 {
        let mut sum = 0i32;
        for k in 0..16 {
            sum += (out[offset::QS + j * 16 + k] as i8) as i32;
        }
        let bsum = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let bytes = bsum.to_le_bytes();
        out[offset::BSUMS + j * 2] = bytes[0];
        out[offset::BSUMS + j * 2 + 1] = bytes[1];
    }

    // d = 1/iscale = -m/128. Dequant = d * qs -> element with |max| rounds
    // back to m.
    let d = 1.0f32 / iscale;
    out[offset::D..offset::D + 4].copy_from_slice(&d.to_le_bytes());
}

/// Quantise a contiguous f32 array into Q8_K blocks. `x.len()` must be a
/// multiple of `BLOCK_SIZE`; `out.len()` must be the corresponding multiple
/// of `BYTES_PER_BLOCK`.
pub fn quantize(x: &[f32], out: &mut [u8]) {
    assert_eq!(
        x.len() % BLOCK_SIZE,
        0,
        "q8_k::quantize: x len {} not multiple of {BLOCK_SIZE}",
        x.len(),
    );
    let n_blocks = x.len() / BLOCK_SIZE;
    assert_eq!(
        out.len(),
        n_blocks * BYTES_PER_BLOCK,
        "q8_k::quantize: out len {} != {n_blocks} * {BYTES_PER_BLOCK}",
        out.len(),
    );
    for (xblock, oblock) in x
        .chunks_exact(BLOCK_SIZE)
        .zip(out.chunks_exact_mut(BYTES_PER_BLOCK))
    {
        quantize_block(xblock, oblock);
    }
}

/// Round an f32 to the nearest i32 with ties-to-even rounding.
///
/// Matches ggml's `nearest_int` (the "magic number" bit trick) and
/// `f32::round_ties_even`, which is cross-platform bit-deterministic.
fn round_ties_even_i32(v: f32) -> i32 {
    v.round_ties_even() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(d: f32, qs: [i8; BLOCK_SIZE]) -> Vec<u8> {
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        out[offset::D..offset::D + 4].copy_from_slice(&d.to_le_bytes());
        for (i, q) in qs.iter().enumerate() {
            out[offset::QS + i] = *q as u8;
        }
        for j in 0..16 {
            let mut s = 0i32;
            for k in 0..16 {
                s += qs[j * 16 + k] as i32;
            }
            let bs = (s.clamp(i16::MIN as i32, i16::MAX as i32) as i16).to_le_bytes();
            out[offset::BSUMS + j * 2] = bs[0];
            out[offset::BSUMS + j * 2 + 1] = bs[1];
        }
        out
    }

    fn read_d(block: &[u8]) -> f32 {
        f32::from_le_bytes(block[offset::D..offset::D + 4].try_into().unwrap())
    }

    fn read_qs(block: &[u8]) -> [i8; BLOCK_SIZE] {
        let mut qs = [0i8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            qs[i] = block[offset::QS + i] as i8;
        }
        qs
    }

    fn read_bsums(block: &[u8]) -> [i16; 16] {
        let mut b = [0i16; 16];
        for j in 0..16 {
            b[j] = i16::from_le_bytes([
                block[offset::BSUMS + j * 2],
                block[offset::BSUMS + j * 2 + 1],
            ]);
        }
        b
    }

    #[test]
    fn zero_input_produces_zero_block() {
        let x = vec![0f32; BLOCK_SIZE];
        let mut out = vec![0xFFu8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        assert_eq!(out, vec![0u8; BYTES_PER_BLOCK]);
    }

    #[test]
    fn constant_positive_input_roundtrips_on_max() {
        // x = [3.0; 256] — every element is the max. d * qs must reproduce
        // 3.0 exactly (within i8 representation).
        let x = vec![3.0f32; BLOCK_SIZE];
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let d = read_d(&out);
        let qs = read_qs(&out);
        // iscale = -128/3 => qs[i] = round(-128/3 * 3) = -128.
        for &q in &qs {
            assert_eq!(q, -128);
        }
        // d = 1/iscale = -3/128. dequant = d * qs = (-3/128) * (-128) = 3.0.
        assert!((d * (qs[0] as f32) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn constant_negative_input_roundtrips_on_max() {
        let x = vec![-5.0f32; BLOCK_SIZE];
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let d = read_d(&out);
        let qs = read_qs(&out);
        // m = -5. iscale = -128/-5 = 25.6 => qs[i] = round(25.6 * -5) = -128.
        // Actually: iscale * x = (-128/-5) * -5 = -128*(-5)/(-5) = -128. So
        // qs[i] = -128 everywhere; dequant = d * -128.
        for &q in &qs {
            assert_eq!(q, -128);
        }
        // d = 1/iscale = -5/128. dequant = (-5/128) * (-128) = 5.0... wait,
        // we wanted -5 back. Check: the sign convention leaves the max-abs
        // element's reconstructed value equal to m only because d absorbs
        // the sign from iscale. For negative m, d is +, qs is negative,
        // product is negative. Let's verify.
        assert!((d * (qs[0] as f32) - -5.0).abs() < 1e-6);
    }

    #[test]
    fn mixed_signs_roundtrip_max_element() {
        // Pick a block where the max abs element is unique and signed.
        let mut x = vec![0f32; BLOCK_SIZE];
        x[10] = 7.5; // positive max
        x[20] = -3.0; // smaller negative
        x[30] = 1.2;
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let d = read_d(&out);
        let qs = read_qs(&out);
        // The positive max should saturate to -128.
        assert_eq!(qs[10], -128);
        let recon_max = d * qs[10] as f32;
        assert!((recon_max - 7.5).abs() < 1e-4);
        // The other non-zero elements should be nonzero small values.
        assert!(qs[20] != 0 && qs[20] > 0); // -3.0 * iscale (-) => positive
        assert!(qs[30] != 0 && qs[30] < 0); // +1.2 * iscale (-) => negative
    }

    #[test]
    fn bsums_match_manual_sum() {
        let mut x = vec![0f32; BLOCK_SIZE];
        // Sweep a range of values so each 16-wide group gets a distinct sum.
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as i32) - 128) as f32 * 0.05; // ≈ -6.4 .. +6.35
        }
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let qs = read_qs(&out);
        let bsums = read_bsums(&out);
        for j in 0..16 {
            let manual: i32 = qs[j * 16..j * 16 + 16].iter().map(|&q| q as i32).sum();
            let expected = manual.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            assert_eq!(bsums[j], expected, "bsum group {j}");
        }
    }

    #[test]
    fn multi_block_quantize() {
        // 3 blocks back-to-back — each must be quantised independently.
        let n = 3;
        let mut x = vec![0f32; n * BLOCK_SIZE];
        for b in 0..n {
            for i in 0..BLOCK_SIZE {
                x[b * BLOCK_SIZE + i] = (b as f32 + 1.0) * ((i as f32) * 0.01 - 1.28);
            }
        }
        let mut out = vec![0u8; n * BYTES_PER_BLOCK];
        quantize(&x, &mut out);
        // Each block has its own scale — read d for each and confirm they
        // differ monotonically with the input magnitude.
        let d0 = read_d(&out[..BYTES_PER_BLOCK]);
        let d1 = read_d(&out[BYTES_PER_BLOCK..2 * BYTES_PER_BLOCK]);
        let d2 = read_d(&out[2 * BYTES_PER_BLOCK..]);
        assert!(d0.abs() < d1.abs() && d1.abs() < d2.abs());
    }

    #[test]
    #[should_panic(expected = "not multiple of 256")]
    fn quantize_rejects_partial_block() {
        let x = vec![0f32; 100];
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize(&x, &mut out);
    }

    #[test]
    fn round_ties_even_is_bit_exact() {
        // Sanity: round_ties_even matches the expected ties-to-even outputs
        // on a handful of critical inputs. Rust's f32::round_ties_even is
        // cross-platform bit-deterministic, so this guards against a
        // regression that silently swapped it for `.round()` (ties-away).
        assert_eq!(round_ties_even_i32(0.5), 0);
        assert_eq!(round_ties_even_i32(1.5), 2);
        assert_eq!(round_ties_even_i32(2.5), 2);
        assert_eq!(round_ties_even_i32(-0.5), 0);
        assert_eq!(round_ties_even_i32(-1.5), -2);
        assert_eq!(round_ties_even_i32(-2.5), -2);
        assert_eq!(round_ties_even_i32(1.4), 1);
        assert_eq!(round_ties_even_i32(-1.4), -1);
    }

    #[test]
    fn build_block_helper_matches_quantize_output() {
        // Sanity on the test helper — build_block with d+qs should produce
        // the same 292 bytes as quantize_block for an input whose quants
        // we can hand-pick.
        let x = vec![0f32; BLOCK_SIZE];
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let helper = build_block(0.0, [0i8; BLOCK_SIZE]);
        assert_eq!(out, helper);
    }
}
