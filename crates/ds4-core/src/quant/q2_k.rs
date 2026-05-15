//! Q2_K block dequantisation + Q8_K dot kernel.
//!
//! Block layout (84 bytes, 256 elements):
//! * `d`       : f16 scale           (2 bytes, offset 0)
//! * `dmin`    : f16 min scale       (2 bytes, offset 2)
//! * `scales`  : 16 × u8             (16 bytes, offset 4)
//!   Each byte packs one 4-bit scale index (low nibble) and one 4-bit min
//!   index (high nibble) for a 16-element sub-block.
//! * `qs`      : 64 × u8             (64 bytes, offset 20)
//!   Packed 2-bit quants: 256 values, 4 per byte.
//!
//! Dequantised value for element `e` in sub-block `b`:
//!   `x = d * (scales[b] & 0xF) * q - dmin * (scales[b] >> 4)`
//!
//! The Q8_K dot kernel follows the antirez/ds4 scalar reference:
//!   `dot = dall * isum - dmin * summs`
//! where `isum` is the weighted sum of Q2_K quants dotted against Q8_K qs,
//! and `summs` folds the Q8_K precomputed bsums against the min offsets.
//!
//! Reference: `block_q2_K` and `ds4_vec_dot_q2_K_q8_K` in antirez/ds4 ds4.c.

use crate::quant::q8_0;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 84;

/// Byte offsets inside one Q2_K block.
pub mod offset {
    pub const D: usize = 0; // f16
    pub const DMIN: usize = 2; // f16
    pub const SCALES: usize = 4; // u8 × 16
    pub const QS: usize = 20; // u8 × 64
}

/// Dequantise one 84-byte Q2_K block into 256 f32s.
///
/// Panics if `out` is not exactly `BLOCK_SIZE` long.
pub fn dequant_block(block: &[u8; BYTES_PER_BLOCK], out: &mut [f32]) {
    assert_eq!(
        out.len(),
        BLOCK_SIZE,
        "q2_k::dequant_block: out len {} != {BLOCK_SIZE}",
        out.len(),
    );

    let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = q8_0::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[offset::SCALES..offset::SCALES + 16];
    let qs = &block[offset::QS..offset::QS + 64];

    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut out_off = 0usize;

    for _ in 0..(BLOCK_SIZE / 128) {
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc0 = scales[is];
            let dl0 = d * ((sc0 & 0x0F) as f32);
            let ml0 = dmin * ((sc0 >> 4) as f32);
            for l in 0..16 {
                let q = ((qs[q_off + l] >> shift) & 3) as i8;
                out[out_off + l] = dl0 * q as f32 - ml0;
            }
            is += 1;
            out_off += 16;

            let sc1 = scales[is];
            let dl1 = d * ((sc1 & 0x0F) as f32);
            let ml1 = dmin * ((sc1 >> 4) as f32);
            for l in 0..16 {
                let q = ((qs[q_off + 16 + l] >> shift) & 3) as i8;
                out[out_off + l] = dl1 * q as f32 - ml1;
            }
            is += 1;
            out_off += 16;

            shift += 2;
        }
        q_off += 32;
    }
}

/// Dequantise a contiguous sequence of Q2_K blocks.
///
/// `bytes.len()` must be a multiple of `BYTES_PER_BLOCK` and `out.len()` must
/// be the corresponding multiple of `BLOCK_SIZE`.
pub fn dequant(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(
        bytes.len() % BYTES_PER_BLOCK,
        0,
        "q2_k::dequant: bytes len {} not multiple of {BYTES_PER_BLOCK}",
        bytes.len(),
    );
    let n_blocks = bytes.len() / BYTES_PER_BLOCK;
    assert_eq!(
        out.len(),
        n_blocks * BLOCK_SIZE,
        "q2_k::dequant: out len {} != {n_blocks} * {BLOCK_SIZE}",
        out.len(),
    );
    for (i, chunk) in bytes.chunks_exact(BYTES_PER_BLOCK).enumerate() {
        let block: &[u8; BYTES_PER_BLOCK] = chunk.try_into().unwrap();
        dequant_block(block, &mut out[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]);
    }
}

/// Dot product of one Q2_K weight block against one Q8_K activation block.
///
/// Both blocks cover exactly 256 elements. The Q8_K block is the
/// `block_q8_K` layout from `quant/q8_k` (f32 scale + 256 i8 quants +
/// 16 i16 bsums).
///
/// Returns the scalar `sum_k weight[k] * act[k]`.
pub fn dot_q2k_q8k_block(q2_block: &[u8; BYTES_PER_BLOCK], q8_block: &[u8]) -> f32 {
    use crate::quant::q8_k;

    // Q8_K block layout sanity.
    debug_assert_eq!(q8_block.len(), q8_k::BYTES_PER_BLOCK);

    let q2_d = q8_0::f16_to_f32(u16::from_le_bytes([q2_block[0], q2_block[1]]));
    let q2_dmin = q8_0::f16_to_f32(u16::from_le_bytes([q2_block[2], q2_block[3]]));

    let q2_scales = &q2_block[offset::SCALES..offset::SCALES + 16];
    let q2_qs = &q2_block[offset::QS..offset::QS + 64];

    let q8_d = f32::from_le_bytes(q8_block[q8_k::offset::D..q8_k::offset::D + 4].try_into().unwrap());
    let q8_qs = &q8_block[q8_k::offset::QS..q8_k::offset::QS + 256];
    let q8_bsums = &q8_block[q8_k::offset::BSUMS..q8_k::offset::BSUMS + 32];

    let dall = q8_d * q2_d;
    let dmin = q8_d * q2_dmin;

    // Pre-fold the min term using Q8_K bsums.
    let mut summs = 0i32;
    for j in 0..16 {
        let bsum = i16::from_le_bytes([q8_bsums[j * 2], q8_bsums[j * 2 + 1]]) as i32;
        summs += bsum * ((q2_scales[j] >> 4) as i32);
    }

    let mut isum = 0i32;
    let mut is = 0usize;
    let mut q2_off = 0usize;
    let mut q8_off = 0usize;

    for _ in 0..(BLOCK_SIZE / 128) {
        let mut shift = 0u32;
        for _ in 0..4 {
            let d0 = (q2_scales[is] & 0x0F) as i32;
            isum += d0 * dot_q2_16(&q2_qs[q2_off..q2_off + 16], &q8_qs[q8_off..q8_off + 16], shift);
            is += 1;
            q8_off += 16;

            let d1 = (q2_scales[is] & 0x0F) as i32;
            isum += d1 * dot_q2_16(&q2_qs[q2_off + 16..q2_off + 32], &q8_qs[q8_off..q8_off + 16], shift);
            is += 1;
            q8_off += 16;

            shift += 2;
        }
        q2_off += 32;
    }

    dall * isum as f32 - dmin * summs as f32
}

/// 16-wide dot product: Q2_K quants (shifted out of packed bytes) × Q8_K i8s.
fn dot_q2_16(q2: &[u8], q8: &[u8], shift: u32) -> i32 {
    debug_assert_eq!(q2.len(), 16);
    debug_assert_eq!(q8.len(), 16);
    let mut sum = 0i32;
    for i in 0..16 {
        let q = ((q2[i] >> shift) & 3) as i32;
        sum += q8[i] as i8 as i32 * q;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(d_bits: u16, dmin_bits: u16, scales: [u8; 16], qs: [u8; 64]) -> [u8; BYTES_PER_BLOCK] {
        let mut block = [0u8; BYTES_PER_BLOCK];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
        block[offset::SCALES..offset::SCALES + 16].copy_from_slice(&scales);
        block[offset::QS..offset::QS + 64].copy_from_slice(&qs);
        block
    }

    #[test]
    fn dequant_block_all_zero() {
        let block = build_block(0x0000, 0x0000, [0u8; 16], [0u8; 64]);
        let mut out = [123.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_block_unit_scale_no_min() {
        // d = 1.0 (0x3C00), dmin = 0.0 (0x0000)
        // scales all 1 (low nibble = 1), so dl = 1.0 for every sub-block.
        // qs all 0b01_01_01_01 = 0x55, so every quant = 1.
        // Expected: every element = 1.0 * 1 * 1 - 0 = 1.0
        let scales = [0x01u8; 16];
        let qs = [0x55u8; 64];
        let block = build_block(0x3C00, 0x0000, scales, qs);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn dequant_block_with_min() {
        // d = 1.0, dmin = 1.0
        // scales = 0x11: scale=1, min=1
        // qs all 0 (quant=0) -> out = 0 - 1*1 = -1.0
        let scales = [0x11u8; 16];
        let qs = [0x00u8; 64];
        let block = build_block(0x3C00, 0x3C00, scales, qs);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, -1.0);
        }
    }

    #[test]
    fn dequant_multi_block() {
        let b0 = build_block(0x3C00, 0x0000, [0x01; 16], [0x55; 64]);
        let b1 = build_block(0x4000, 0x0000, [0x02; 16], [0x55; 64]);
        // b0: d=1.0, scale=1 -> 1.0
        // b1: d=2.0, scale=2 -> 4.0
        let mut bytes = Vec::with_capacity(BYTES_PER_BLOCK * 2);
        bytes.extend_from_slice(&b0);
        bytes.extend_from_slice(&b1);
        let mut out = vec![0.0f32; BLOCK_SIZE * 2];
        dequant(&bytes, &mut out);
        assert!(out[..BLOCK_SIZE].iter().all(|&v| v == 1.0));
        assert!(out[BLOCK_SIZE..].iter().all(|&v| v == 4.0));
    }

    #[test]
    fn dot_block_against_zeros_is_zero() {
        let q2 = build_block(0x3C00, 0x0000, [0x01; 16], [0x55; 64]);
        let q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        assert_eq!(dot_q2k_q8k_block(&q2, &q8), 0.0);
    }

    #[test]
    fn dot_block_all_ones_against_constant_q8() {
        // Q2_K: d=1.0, dmin=0, all scales=1, all qs=1 -> every weight = 1.0
        // Q8_K: d=1.0, all qs=2 -> dot = 256 * 1.0 * 2 = 512
        let q2 = build_block(0x3C00, 0x0000, [0x01; 16], [0x55; 64]);

        let mut q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        q8[0..4].copy_from_slice(&1.0f32.to_le_bytes()); // d = 1.0
        for i in 0..256 {
            q8[4 + i] = 2i8 as u8; // qs = 2
        }
        for j in 0..16 {
            let bsum = (16i16 * 2i16).to_le_bytes();
            q8[260 + j * 2] = bsum[0];
            q8[260 + j * 2 + 1] = bsum[1];
        }

        assert_eq!(dot_q2k_q8k_block(&q2, &q8), 512.0);
    }

    #[test]
    fn dot_block_min_term_only() {
        // Q2_K: d=1.0, dmin=1.0, all scales=0x11 (scale=1, min=1), all qs=0
        //        -> every weight = 0 - 1*1 = -1.0
        // Q8_K: d=1.0, all qs=1 -> dot = 256 * (-1.0) * 1 = -256
        let q2 = build_block(0x3C00, 0x3C00, [0x11; 16], [0x00; 64]);

        let mut q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        q8[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..256 {
            q8[4 + i] = 1i8 as u8;
        }
        for j in 0..16 {
            let bsum = (16i16).to_le_bytes();
            q8[260 + j * 2] = bsum[0];
            q8[260 + j * 2 + 1] = bsum[1];
        }

        assert_eq!(dot_q2k_q8k_block(&q2, &q8), -256.0);
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn dequant_block_rejects_short_out() {
        let block = build_block(0x3C00, 0x0000, [0; 16], [0; 64]);
        let mut out = [0.0f32; BLOCK_SIZE - 1];
        dequant_block(&block, &mut out);
    }

    #[test]
    #[should_panic(expected = "not multiple of 84")]
    fn dequant_rejects_partial_block() {
        let bytes = vec![0u8; BYTES_PER_BLOCK - 1];
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequant(&bytes, &mut out);
    }
}
