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

    // Fuse quantisation + bsum in one pass over x/qs, 16 elements at a time.
    // Each 16-wide slab of qs fills one entry of bsums, so we accumulate the
    // sum as we write.
    //
    // Each qs entry is in [-128, 127] (after the clamp); a sum over 16 of
    // them lives in [-2048, 2032], which fits in i16 without any further
    // clamping.
    //
    // Why `.clamp(-128, 127)` rather than `.min(127)`: if `m` is subnormal,
    // `iscale` can overflow to ±inf, which takes `iscale * x[i]` to ±inf or
    // NaN. Rust's saturating `as i32` cast maps -inf to `i32::MIN`, and the
    // subsequent wrapping `as i8` would silently turn that into 0. The
    // two-sided clamp pins it to -128 instead. Upper bound is `.min(127)`'s
    // job under the typical path.
    let (qs_out, tail) = out[offset::QS..].split_at_mut(BLOCK_SIZE);
    let bsums_out = &mut tail[..32];
    for (x_slab, (qs_slab, bsum_slab)) in x.chunks_exact(16).zip(
        qs_out
            .chunks_exact_mut(16)
            .zip(bsums_out.chunks_exact_mut(2)),
    ) {
        let mut bsum = 0i16;
        for (qbyte, &xv) in qs_slab.iter_mut().zip(x_slab) {
            let q = round_ties_even_i32(iscale * xv).clamp(-128, 127) as i8;
            *qbyte = q as u8;
            bsum += q as i16;
        }
        bsum_slab.copy_from_slice(&bsum.to_le_bytes());
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
    assert!(
        x.len().is_multiple_of(BLOCK_SIZE),
        "q8_k::quantize: x len {} not multiple of {BLOCK_SIZE}",
        x.len(),
    );
    let n_blocks = x.len() / BLOCK_SIZE;
    let expected = n_blocks
        .checked_mul(BYTES_PER_BLOCK)
        .expect("q8_k::quantize: output size overflowed usize");
    assert_eq!(
        out.len(),
        expected,
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

    fn dequant(out: &[u8]) -> Vec<f32> {
        let d = read_d(out);
        let qs = read_qs(out);
        qs.iter().map(|&q| d * q as f32).collect()
    }

    #[test]
    fn round_trip_monotonic_ramp_within_quantization_step() {
        let mut x = vec![0f32; BLOCK_SIZE];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 - 128.0) * 0.1;
        }
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let recon = dequant(&out);
        // amax = 12.7 -> step size = 12.7 / 128 ~= 0.0992. Round-trip error
        // should be bounded by half that.
        let step = 12.7 / 128.0;
        for (xv, rv) in x.iter().zip(recon.iter()) {
            assert!(
                (xv - rv).abs() <= step,
                "input {xv}, recon {rv}, step {step}",
            );
        }
    }

    #[test]
    fn round_trip_alternating_signs_preserves_pattern() {
        let mut x = vec![0f32; BLOCK_SIZE];
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
        }
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let recon = dequant(&out);
        // m = +1, iscale = -128. Positive elements clamp to -128 -> recon
        // exactly +1. Negative elements would map to +128 but clamp to +127,
        // so recon is -127/128. Error magnitude bounded by one step (1/128).
        let step = 1.0 / 128.0;
        for (xv, rv) in x.iter().zip(recon.iter()) {
            assert_eq!(xv.signum(), rv.signum());
            assert!((xv - rv).abs() <= step + 1e-6);
        }
    }

    #[test]
    fn round_trip_near_int8_saturation() {
        // amax exactly 1.0 -> iscale = -128, qs spans full -128..127 range
        // depending on input. Each integer step in qs maps to 1/128 in x.
        let mut x = vec![0f32; BLOCK_SIZE];
        for (i, v) in x.iter_mut().enumerate() {
            // Distribute across [-1.0, 1.0)
            *v = (i as f32 - 128.0) / 128.0;
        }
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let qs = read_qs(&out);
        // Element with x = -1.0 must hit -128 exactly (the chosen max).
        assert_eq!(qs[0], -128);
        let recon = dequant(&out);
        for (xv, rv) in x.iter().zip(recon.iter()) {
            assert!((xv - rv).abs() <= 1.0 / 128.0 + 1e-6);
        }
    }

    #[test]
    fn round_trip_very_small_magnitudes() {
        let mut x = vec![0f32; BLOCK_SIZE];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 - 128.0) * 1e-20;
        }
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let recon = dequant(&out);
        // amax = 128 * 1e-20 = 1.28e-18. Step = amax / 128 = 1e-20.
        let step = 1.28e-18 / 128.0;
        for (xv, rv) in x.iter().zip(recon.iter()) {
            assert!((xv - rv).abs() <= step + 1e-25, "input {xv}, recon {rv}",);
        }
    }

    #[test]
    fn d_sign_matches_negative_m() {
        // d = -m/128. m negative -> d positive.
        let mut x = vec![0.0f32; BLOCK_SIZE];
        x[0] = -2.0;
        x[1] = 1.0;
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let d = read_d(&out);
        assert!(d > 0.0);
        assert!((d - (2.0 / 128.0)).abs() < 1e-7);
    }

    #[test]
    fn d_sign_matches_positive_m() {
        // d = -m/128. m positive -> d negative.
        let mut x = vec![0.0f32; BLOCK_SIZE];
        x[0] = 2.0;
        x[1] = -1.0;
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let d = read_d(&out);
        assert!(d < 0.0);
        assert!((d - (-2.0 / 128.0)).abs() < 1e-7);
    }

    #[test]
    fn first_max_wins_when_ties_in_abs() {
        // Two equal-abs candidates with opposite signs. The first one in scan
        // order is picked (strict > comparison in the loop).
        let mut x = vec![0.0f32; BLOCK_SIZE];
        x[5] = 4.0;
        x[200] = -4.0;
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let d = read_d(&out);
        // m = +4, iscale = -32, d = -4/128 = -0.03125
        assert!((d - (-4.0 / 128.0)).abs() < 1e-7);
    }

    #[test]
    fn bsums_in_i16_range_for_full_saturation() {
        // Every qs ends up at -128 -> each 16-group sums to -2048, which fits
        // in i16. Verify no clamping kicks in.
        let x = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let bsums = read_bsums(&out);
        for &b in &bsums {
            assert_eq!(b, -2048);
        }
    }

    #[test]
    fn quantize_zero_length_input() {
        let x: Vec<f32> = Vec::new();
        let mut out: Vec<u8> = Vec::new();
        quantize(&x, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "x len")]
    fn quantize_block_rejects_short_x() {
        let x = vec![0f32; BLOCK_SIZE - 1];
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn quantize_block_rejects_wrong_out_len() {
        let x = vec![0f32; BLOCK_SIZE];
        let mut out = vec![0u8; BYTES_PER_BLOCK - 1];
        quantize_block(&x, &mut out);
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn quantize_rejects_mismatched_out_len() {
        let x = vec![0f32; BLOCK_SIZE * 2];
        let mut out = vec![0u8; BYTES_PER_BLOCK]; // should be 2 * BYTES_PER_BLOCK
        quantize(&x, &mut out);
    }

    #[test]
    fn block_boundary_independence() {
        // First block is huge, second is tiny. Each must get its own scale.
        let mut x = vec![0f32; BLOCK_SIZE * 2];
        for v in x[..BLOCK_SIZE].iter_mut() {
            *v = 1000.0;
        }
        for v in x[BLOCK_SIZE..].iter_mut() {
            *v = 0.001;
        }
        let mut out = vec![0u8; BYTES_PER_BLOCK * 2];
        quantize(&x, &mut out);

        let d0 = read_d(&out[..BYTES_PER_BLOCK]);
        let d1 = read_d(&out[BYTES_PER_BLOCK..]);
        assert!((d0 - (-1000.0 / 128.0)).abs() < 1e-3);
        assert!((d1 - (-0.001 / 128.0)).abs() < 1e-9);

        let qs0 = read_qs(&out[..BYTES_PER_BLOCK]);
        let qs1 = read_qs(&out[BYTES_PER_BLOCK..]);
        for &q in &qs0 {
            assert_eq!(q, -128);
        }
        for &q in &qs1 {
            assert_eq!(q, -128);
        }
    }

    #[test]
    fn quantize_overwrites_existing_output() {
        // Pre-fill output with garbage; quantize_block must overwrite it
        // entirely (especially the bsums area, which earlier code zeroed).
        let mut x = vec![0f32; BLOCK_SIZE];
        x[0] = 1.0;
        let mut out = vec![0xFFu8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let qs = read_qs(&out);
        let bsums = read_bsums(&out);
        // qs[0] = -128, all other qs = 0 -> first bsum group sums to -128,
        // remaining groups sum to 0.
        assert_eq!(qs[0], -128);
        for &q in &qs[1..] {
            assert_eq!(q, 0);
        }
        assert_eq!(bsums[0], -128);
        for &b in &bsums[1..] {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn bsums_consistent_with_qs_for_random_input() {
        // Pseudo-random sweep — confirm bsums always equal the live sum of
        // qs across every 16-wide slab.
        let mut x = vec![0f32; BLOCK_SIZE];
        let mut state: u64 = 0xcafef00dd15ea5e5;
        for v in x.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 32) as u32;
            *v = (bits as f32 / u32::MAX as f32 - 0.5) * 8.0;
        }
        let mut out = vec![0u8; BYTES_PER_BLOCK];
        quantize_block(&x, &mut out);
        let qs = read_qs(&out);
        let bsums = read_bsums(&out);
        for j in 0..16 {
            let expected: i32 = qs[j * 16..j * 16 + 16].iter().map(|&q| q as i32).sum();
            assert_eq!(bsums[j] as i32, expected, "group {j}");
        }
    }
}
