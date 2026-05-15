//! Q4_K block dequantisation + Q8_K dot kernel.
//!
//! Block layout (144 bytes, 256 elements):
//! * `d`      : f16 super-block scale  (2 bytes, offset 0)
//! * `dmin`   : f16 super-block min    (2 bytes, offset 2)
//! * `scales` : 12 × u8                (12 bytes, offset 4) Eight 6-bit scales and eight 6-bit mins
//!   packed using the standard ggml `get_scale_min_k4` layout.
//! * `qs`     : 128 × u8               (128 bytes, offset 16) Two 4-bit quants per byte (low nibble
//!   first, then high nibble).
//!
//! Each block has 8 sub-blocks of 32 elements. For sub-block `j`:
//!   `x = d * scale[j] * q - dmin * min[j]`
//! where `q` is the 4-bit quant value (0..15).
//!
//! The Q8_K dot kernel mirrors the antirez/ds4 scalar reference:
//!   `dot = dall * isum - dmin * summs`
//! `isum` accumulates per-sub-block scale × dot(q4 nibble, Q8_K i8s); `summs`
//! folds Q8_K bsums against the 6-bit min offsets.
//!
//! Reference: `block_q4_K` and `ds4_vec_dot_q4_K_q8_K` in antirez/ds4 ds4.c.

use crate::quant::q8_0;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 144;

/// Byte offsets inside one Q4_K block.
pub mod offset {
    pub const D: usize = 0; // f16
    pub const DMIN: usize = 2; // f16
    pub const SCALES: usize = 4; // u8 × 12
    pub const QS: usize = 16; // u8 × 128
}

/// Decode the 6-bit scale and 6-bit min for sub-block `j` (0..8) from the
/// 12-byte packed scales array. Mirrors ggml's `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    debug_assert!(j < 8);
    debug_assert_eq!(q.len(), 12);
    if j < 4 {
        (q[j] & 0x3F, q[j + 4] & 0x3F)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantise one 144-byte Q4_K block into 256 f32s.
pub fn dequant_block(block: &[u8; BYTES_PER_BLOCK], out: &mut [f32]) {
    assert_eq!(
        out.len(),
        BLOCK_SIZE,
        "q4_k::dequant_block: out len {} != {BLOCK_SIZE}",
        out.len(),
    );

    let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = q8_0::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[offset::SCALES..offset::SCALES + 12];
    let qs = &block[offset::QS..offset::QS + 128];

    // 4 outer iterations, each processing 64 elements (one byte slab of qs).
    for j in 0..4 {
        let (sc_lo, mn_lo) = get_scale_min_k4(2 * j, scales);
        let (sc_hi, mn_hi) = get_scale_min_k4(2 * j + 1, scales);
        let d1 = d * sc_lo as f32;
        let m1 = dmin * mn_lo as f32;
        let d2 = d * sc_hi as f32;
        let m2 = dmin * mn_hi as f32;
        let q_chunk = &qs[j * 32..(j + 1) * 32];
        let out_chunk = &mut out[j * 64..(j + 1) * 64];
        for l in 0..32 {
            let q_lo = (q_chunk[l] & 0x0F) as f32;
            let q_hi = (q_chunk[l] >> 4) as f32;
            out_chunk[l] = d1 * q_lo - m1;
            out_chunk[32 + l] = d2 * q_hi - m2;
        }
    }
}

/// Dequantise a contiguous sequence of Q4_K blocks.
pub fn dequant(bytes: &[u8], out: &mut [f32]) {
    assert!(
        bytes.len().is_multiple_of(BYTES_PER_BLOCK),
        "q4_k::dequant: bytes len {} not multiple of {BYTES_PER_BLOCK}",
        bytes.len(),
    );
    let n_blocks = bytes.len() / BYTES_PER_BLOCK;
    assert_eq!(
        out.len(),
        n_blocks * BLOCK_SIZE,
        "q4_k::dequant: out len {} != {n_blocks} * {BLOCK_SIZE}",
        out.len(),
    );
    for (chunk, out_block) in bytes
        .chunks_exact(BYTES_PER_BLOCK)
        .zip(out.chunks_exact_mut(BLOCK_SIZE))
    {
        let block: &[u8; BYTES_PER_BLOCK] = chunk.try_into().unwrap();
        dequant_block(block, out_block);
    }
}

/// Dot product of one Q4_K weight block against one Q8_K activation block.
///
/// Both blocks cover exactly 256 elements. Returns `sum_k weight[k] * act[k]`.
pub fn dot_q4k_q8k_block(q4_block: &[u8; BYTES_PER_BLOCK], q8_block: &[u8]) -> f32 {
    use crate::quant::q8_k;

    debug_assert_eq!(q8_block.len(), q8_k::BYTES_PER_BLOCK);

    let q4_d = q8_0::f16_to_f32(u16::from_le_bytes([q4_block[0], q4_block[1]]));
    let q4_dmin = q8_0::f16_to_f32(u16::from_le_bytes([q4_block[2], q4_block[3]]));

    let scales_packed = &q4_block[offset::SCALES..offset::SCALES + 12];
    let qs = &q4_block[offset::QS..offset::QS + 128];

    let q8_d = f32::from_le_bytes(
        q8_block[q8_k::offset::D..q8_k::offset::D + 4]
            .try_into()
            .unwrap(),
    );
    let q8_qs = &q8_block[q8_k::offset::QS..q8_k::offset::QS + 256];
    let q8_bsums = &q8_block[q8_k::offset::BSUMS..q8_k::offset::BSUMS + 32];

    let dall = q8_d * q4_d;
    let dmin = q8_d * q4_dmin;

    // summs: each Q4_K min covers 32 elements, which spans two adjacent Q8_K
    // bsum groups (each bsum group is 16 elements wide).
    let mut summs = 0i32;
    for j in 0..8 {
        let (_, mn) = get_scale_min_k4(j, scales_packed);
        let bs0 = i16::from_le_bytes([q8_bsums[j * 4], q8_bsums[j * 4 + 1]]) as i32;
        let bs1 = i16::from_le_bytes([q8_bsums[j * 4 + 2], q8_bsums[j * 4 + 3]]) as i32;
        summs += (bs0 + bs1) * mn as i32;
    }

    // isum: 8 sub-blocks of 32 elements. Sub-blocks 2j and 2j+1 share the same
    // 32-byte qs slab — low nibble for the even sub-block, high nibble for
    // the odd one.
    let mut isum = 0i32;
    for j in 0..4 {
        let (sc_lo, _) = get_scale_min_k4(2 * j, scales_packed);
        let (sc_hi, _) = get_scale_min_k4(2 * j + 1, scales_packed);
        let q_chunk = &qs[j * 32..(j + 1) * 32];
        let q8_lo = &q8_qs[j * 64..j * 64 + 32];
        let q8_hi = &q8_qs[j * 64 + 32..(j + 1) * 64];

        let mut acc_lo = 0i32;
        let mut acc_hi = 0i32;
        for l in 0..32 {
            let q_l = (q_chunk[l] & 0x0F) as i32;
            let q_h = (q_chunk[l] >> 4) as i32;
            acc_lo += q8_lo[l] as i8 as i32 * q_l;
            acc_hi += q8_hi[l] as i8 as i32 * q_h;
        }
        isum += sc_lo as i32 * acc_lo + sc_hi as i32 * acc_hi;
    }

    dall * isum as f32 - dmin * summs as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(
        d_bits: u16,
        dmin_bits: u16,
        scales: [u8; 12],
        qs: [u8; 128],
    ) -> [u8; BYTES_PER_BLOCK] {
        let mut block = [0u8; BYTES_PER_BLOCK];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
        block[offset::SCALES..offset::SCALES + 12].copy_from_slice(&scales);
        block[offset::QS..offset::QS + 128].copy_from_slice(&qs);
        block
    }

    #[test]
    fn dequant_block_all_zero() {
        let block = build_block(0, 0, [0; 12], [0; 128]);
        let mut out = [123.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_block_unit_scale_no_min() {
        // d=1.0, dmin=0, all scales encode (sc=1, min=0) for sub-blocks 0..4
        // and decoding rules for 4..8 still recover sc=1, min=0 if we set
        // the relevant bits accordingly. Easier: rely on first 4 sub-blocks
        // and set the encoding for sub-blocks 4..8 to also produce sc=1,m=0.
        //
        // For j<4: scales[j] & 63 = 1, scales[j+4] & 63 = 0
        // For j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4) = 1
        //          m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4) = 0
        //
        // Set scales[0..4] = 0x01 (low 6 bits = 1, high 2 bits = 0)
        //     scales[4..8] = 0x00 (low 6 bits = 0, high 2 bits = 0)
        //     scales[8..12] = 0x01 (low nibble = 1 -> sc bits 0..3, high nibble = 0 -> m bits 0..3)
        let mut scales = [0u8; 12];
        for s in scales.iter_mut().take(4) {
            *s = 0x01;
        }
        for s in scales.iter_mut().skip(8) {
            *s = 0x01;
        }
        // qs all 0x11 -> low nibble = 1, high nibble = 1 -> all quants = 1
        let qs = [0x11u8; 128];
        let block = build_block(0x3C00, 0x0000, scales, qs);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn dequant_block_with_min() {
        // d=1.0, dmin=1.0, all sub-blocks have sc=0, m=1, all qs=0
        // -> x = 1*0*0 - 1*1 = -1.0
        // For j<4: scales[j] & 63 = 0, scales[j+4] & 63 = 1
        // For j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4]>>6)<<4) = 0
        //          m  = (scales[j+4] >> 4)   | ((scales[j]>>6)<<4)   = 1
        // -> scales[0..4] = 0x00, scales[4..8] = 0x01,
        //    scales[8..12] high nibble = 1, low nibble = 0 -> 0x10
        let mut scales = [0u8; 12];
        for s in scales.iter_mut().skip(4).take(4) {
            *s = 0x01;
        }
        for s in scales.iter_mut().skip(8) {
            *s = 0x10;
        }
        let block = build_block(0x3C00, 0x3C00, scales, [0u8; 128]);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, -1.0);
        }
    }

    #[test]
    fn dot_block_against_zeros_is_zero() {
        let mut scales = [0u8; 12];
        for s in scales.iter_mut().take(4) {
            *s = 0x01;
        }
        for s in scales.iter_mut().skip(8) {
            *s = 0x01;
        }
        let q4 = build_block(0x3C00, 0x0000, scales, [0x11u8; 128]);
        let q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        assert_eq!(dot_q4k_q8k_block(&q4, &q8), 0.0);
    }

    #[test]
    fn dot_block_all_ones_against_constant_q8() {
        // Q4_K: all weights = 1.0 (see dequant_block_unit_scale_no_min setup).
        // Q8_K: d=1.0, all qs=2 -> dot = 256 * 1.0 * 2 = 512
        let mut scales = [0u8; 12];
        for s in scales.iter_mut().take(4) {
            *s = 0x01;
        }
        for s in scales.iter_mut().skip(8) {
            *s = 0x01;
        }
        let q4 = build_block(0x3C00, 0x0000, scales, [0x11u8; 128]);

        let mut q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        q8[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..256 {
            q8[4 + i] = 2i8 as u8;
        }
        for j in 0..16 {
            let bsum = (16i16 * 2i16).to_le_bytes();
            q8[260 + j * 2] = bsum[0];
            q8[260 + j * 2 + 1] = bsum[1];
        }

        assert_eq!(dot_q4k_q8k_block(&q4, &q8), 512.0);
    }

    #[test]
    fn dot_block_min_term_only() {
        // Q4_K: all weights = -1.0. Q8_K: d=1, qs=1 -> dot = -256.
        let mut scales = [0u8; 12];
        for s in scales.iter_mut().skip(4).take(4) {
            *s = 0x01;
        }
        for s in scales.iter_mut().skip(8) {
            *s = 0x10;
        }
        let q4 = build_block(0x3C00, 0x3C00, scales, [0u8; 128]);

        let mut q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        q8[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..256 {
            q8[4 + i] = 1i8 as u8;
        }
        for j in 0..16 {
            let bsum = 16i16.to_le_bytes();
            q8[260 + j * 2] = bsum[0];
            q8[260 + j * 2 + 1] = bsum[1];
        }

        assert_eq!(dot_q4k_q8k_block(&q4, &q8), -256.0);
    }

    #[test]
    fn get_scale_min_k4_round_trip() {
        // Hand-encode (sc=37, m=22) for sub-block 5 and verify the decoder
        // pulls it back out. sc 6-bit = 0b100101, m 6-bit = 0b010110.
        // For j=5: low 4 of sc go into scales[9] low nibble -> 0x05
        //          high 2 of sc go into scales[1] top 2 bits -> bits 6..7 = 0b10
        //          low 4 of m  go into scales[9] high nibble -> 0x60
        //          high 2 of m go into scales[5] top 2 bits -> bits 6..7 = 0b01
        let mut scales = [0u8; 12];
        scales[9] = 0x65; // m_lo=6, sc_lo=5
        scales[1] = 0b10_000000; // sc_hi=2 (bits 4..5 of decoded sc)
        scales[5] = 0b01_000000; // m_hi=1
        let (sc, m) = get_scale_min_k4(5, &scales);
        assert_eq!(sc, 0b10_0101); // 37
        assert_eq!(m, 0b01_0110); // 22
    }

    #[test]
    fn dequant_multi_block() {
        let mut scales = [0u8; 12];
        for s in scales.iter_mut().take(4) {
            *s = 0x01;
        }
        for s in scales.iter_mut().skip(8) {
            *s = 0x01;
        }
        let b0 = build_block(0x3C00, 0x0000, scales, [0x11u8; 128]); // 1.0
        let b1 = build_block(0x4000, 0x0000, scales, [0x11u8; 128]); // 2.0

        let mut bytes = Vec::with_capacity(BYTES_PER_BLOCK * 2);
        bytes.extend_from_slice(&b0);
        bytes.extend_from_slice(&b1);
        let mut out = vec![0.0f32; BLOCK_SIZE * 2];
        dequant(&bytes, &mut out);
        assert!(out[..BLOCK_SIZE].iter().all(|&v| v == 1.0));
        assert!(out[BLOCK_SIZE..].iter().all(|&v| v == 2.0));
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn dequant_block_rejects_short_out() {
        let block = build_block(0, 0, [0; 12], [0; 128]);
        let mut out = [0.0f32; BLOCK_SIZE - 1];
        dequant_block(&block, &mut out);
    }

    #[test]
    #[should_panic(expected = "not multiple of 144")]
    fn dequant_rejects_partial_block() {
        let bytes = vec![0u8; BYTES_PER_BLOCK - 1];
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequant(&bytes, &mut out);
    }
}
