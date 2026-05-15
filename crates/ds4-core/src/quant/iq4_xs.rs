//! IQ4_XS block dequantisation + Q8_K dot kernel.
//!
//! Block layout (136 bytes, 256 elements):
//! * `d`        : f16 super-block scale (2 bytes, offset 0)
//! * `scales_h` : u16                   (2 bytes, offset 2) Top 2 bits of each 6-bit signed scale
//!   for the 8 sub-blocks (2 × 8 = 16 bits).
//! * `scales_l` : 4 × u8                (4 bytes, offset 4) Low 4 bits of each 6-bit scale, packed
//!   two per byte.
//! * `qs`       : 128 × u8              (128 bytes, offset 8) Each sub-block of 32 elements uses 16
//!   bytes; element layout within a sub-block is "low nibbles first 16, then high nibbles".
//!
//! For sub-block `ib` (0..8) the signed 6-bit scale is reassembled from the
//! low nibble in `scales_l` and the 2-bit slice in `scales_h`, then offset by
//! `-32` to give a value in `-32..32`. The element value is then
//! `x = d * signed_scale * kvalues_iq4nl[q_4bit_index]`.
//!
//! IQ4_XS has no min term — scales are signed and the codebook is signed, so
//! the dot kernel reduces to `dall * isum`.
//!
//! Reference: `block_iq4_xs`, `dequantize_row_iq4_xs`, and
//! `ds4_vec_dot_iq4_xs_q8_K` in ggml-quants.c / antirez/ds4 ds4.c.

use crate::quant::{iq4_codebook::KVALUES_IQ4NL, q8_0};

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 136;
const SUB_BLOCKS: usize = 8;

pub mod offset {
    pub const D: usize = 0; // f16
    pub const SCALES_H: usize = 2; // u16
    pub const SCALES_L: usize = 4; // u8 × 4
    pub const QS: usize = 8; // u8 × 128
}

/// Decode the signed 6-bit scale for sub-block `ib` (0..8).
fn signed_scale(ib: usize, scales_l: &[u8], scales_h: u16) -> i32 {
    debug_assert!(ib < SUB_BLOCKS);
    debug_assert_eq!(scales_l.len(), 4);
    let ls_lo = (scales_l[ib / 2] >> ((ib % 2) * 4)) & 0x0F;
    let ls_hi = ((scales_h >> (2 * ib)) & 0x3) as u8;
    (ls_lo | (ls_hi << 4)) as i32 - 32
}

/// Dequantise one 136-byte IQ4_XS block into 256 f32s.
pub fn dequant_block(block: &[u8; BYTES_PER_BLOCK], out: &mut [f32]) {
    assert_eq!(
        out.len(),
        BLOCK_SIZE,
        "iq4_xs::dequant_block: out len {} != {BLOCK_SIZE}",
        out.len(),
    );

    let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[offset::SCALES_L..offset::SCALES_L + 4];
    let qs = &block[offset::QS..offset::QS + 128];

    for ib in 0..SUB_BLOCKS {
        let dl = d * signed_scale(ib, scales_l, scales_h) as f32;
        let q_chunk = &qs[ib * 16..(ib + 1) * 16];
        let out_chunk = &mut out[ib * 32..(ib + 1) * 32];
        for j in 0..16 {
            out_chunk[j] = dl * KVALUES_IQ4NL[(q_chunk[j] & 0x0F) as usize] as f32;
            out_chunk[j + 16] = dl * KVALUES_IQ4NL[(q_chunk[j] >> 4) as usize] as f32;
        }
    }
}

/// Dequantise a contiguous sequence of IQ4_XS blocks.
pub fn dequant(bytes: &[u8], out: &mut [f32]) {
    assert!(
        bytes.len().is_multiple_of(BYTES_PER_BLOCK),
        "iq4_xs::dequant: bytes len {} not multiple of {BYTES_PER_BLOCK}",
        bytes.len(),
    );
    let n_blocks = bytes.len() / BYTES_PER_BLOCK;
    assert_eq!(
        out.len(),
        n_blocks * BLOCK_SIZE,
        "iq4_xs::dequant: out len {} != {n_blocks} * {BLOCK_SIZE}",
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

/// Dot product of one IQ4_XS weight block against one Q8_K activation block.
pub fn dot_iq4xs_q8k_block(iq4_block: &[u8; BYTES_PER_BLOCK], q8_block: &[u8]) -> f32 {
    use crate::quant::q8_k;

    debug_assert_eq!(q8_block.len(), q8_k::BYTES_PER_BLOCK);

    let q4_d = q8_0::f16_to_f32(u16::from_le_bytes([iq4_block[0], iq4_block[1]]));
    let scales_h = u16::from_le_bytes([iq4_block[2], iq4_block[3]]);
    let scales_l = &iq4_block[offset::SCALES_L..offset::SCALES_L + 4];
    let qs = &iq4_block[offset::QS..offset::QS + 128];

    let q8_d = f32::from_le_bytes(
        q8_block[q8_k::offset::D..q8_k::offset::D + 4]
            .try_into()
            .unwrap(),
    );
    let q8_qs = &q8_block[q8_k::offset::QS..q8_k::offset::QS + 256];

    let dall = q8_d * q4_d;

    let mut isum = 0i32;
    for ib in 0..SUB_BLOCKS {
        let scale = signed_scale(ib, scales_l, scales_h);
        let q_chunk = &qs[ib * 16..(ib + 1) * 16];
        let q8_lo = &q8_qs[ib * 32..ib * 32 + 16];
        let q8_hi = &q8_qs[ib * 32 + 16..(ib + 1) * 32];

        let mut acc = 0i32;
        for j in 0..16 {
            acc += q8_lo[j] as i8 as i32 * KVALUES_IQ4NL[(q_chunk[j] & 0x0F) as usize] as i32;
            acc += q8_hi[j] as i8 as i32 * KVALUES_IQ4NL[(q_chunk[j] >> 4) as usize] as i32;
        }
        isum += scale * acc;
    }

    dall * isum as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(
        d_bits: u16,
        scales_h: u16,
        scales_l: [u8; 4],
        qs: [u8; 128],
    ) -> [u8; BYTES_PER_BLOCK] {
        let mut block = [0u8; BYTES_PER_BLOCK];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        block[2..4].copy_from_slice(&scales_h.to_le_bytes());
        block[offset::SCALES_L..offset::SCALES_L + 4].copy_from_slice(&scales_l);
        block[offset::QS..offset::QS + 128].copy_from_slice(&qs);
        block
    }

    #[test]
    fn dequant_block_all_zero_d() {
        // d = 0 -> output is all zeros regardless of qs / scales.
        let block = build_block(0x0000, 0xFFFF, [0xFF; 4], [0xFF; 128]);
        let mut out = [123.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_block_unit_scale_zero_index() {
        // d=1.0, all scales encode signed 1 (raw 33 = 0b100001),
        // qs all zero -> codebook index 0 = -127.
        // Result: every element = 1.0 * 1 * -127 = -127.0
        // raw 33 = 0x21: low 4 bits = 0x1, high 2 bits = 0b10 (=2)
        // scales_l: each nibble = 0x1 -> bytes = 0x11
        // scales_h: each pair = 0b10 -> 0xAAAA
        let block = build_block(0x3C00, 0xAAAA, [0x11; 4], [0x00; 128]);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, -127.0);
        }
    }

    #[test]
    fn dequant_block_zero_scale_yields_zero() {
        // signed scale 0 = raw 32 = 0b100000. low 4 = 0, high 2 = 0b10.
        // scales_l = 0x00, scales_h = 0xAAAA.
        let block = build_block(0x3C00, 0xAAAA, [0x00; 4], [0xFF; 128]);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn signed_scale_decodes_known_values() {
        // Sub-block 0: low4=5, high2=0b11 -> raw 0x35 = 53 -> signed = 21
        // Sub-block 1: low4=0xA, high2=0b00 -> raw 0x0A = 10 -> signed = -22
        // Sub-block 2: low4=0x0, high2=0b00 -> raw 0    -> signed = -32
        // Pack into scales_l[0] = 0xA5 (sub0 low nib, sub1 high nib),
        //          scales_h: sub0 bits = 0b11 at bits 0..1, sub1 bits = 0b00 at bits 2..3,
        //                    sub2 bits = 0b00 at bits 4..5 -> 0x0003
        let scales_l = [0xA5u8, 0, 0, 0];
        let scales_h = 0b00_00_00_00_00_00_00_11u16;
        assert_eq!(signed_scale(0, &scales_l, scales_h), 21);
        assert_eq!(signed_scale(1, &scales_l, scales_h), -22);
        assert_eq!(signed_scale(2, &scales_l, scales_h), -32);
    }

    #[test]
    fn dot_block_against_zero_q8_is_zero() {
        let block = build_block(0x3C00, 0xAAAA, [0x11; 4], [0x77; 128]);
        let q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        assert_eq!(dot_iq4xs_q8k_block(&block, &q8), 0.0);
    }

    #[test]
    fn dot_block_constant_codebook_against_constant_q8() {
        // d=1.0, signed scale=1 everywhere, qs all 0x00 -> codebook[0] = -127.
        // Q8_K: d=1.0, qs all 1 -> per element = -127.
        // Sum over 256 = -32512.
        let block = build_block(0x3C00, 0xAAAA, [0x11; 4], [0x00; 128]);
        let mut q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        q8[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..256 {
            q8[4 + i] = 1i8 as u8;
        }
        // bsums not used by IQ4_XS dot — leave at zero.
        assert_eq!(dot_iq4xs_q8k_block(&block, &q8), -32512.0);
    }

    #[test]
    fn dot_block_signed_scale_negative_flips_sign() {
        // signed scale = -1 = raw 31 = 0b011111. low4 = 0xF, high2 = 0b01.
        // scales_l = 0xFF (sub0 nib = 0xF, sub1 nib = 0xF, etc.)
        // scales_h: each pair = 0b01 -> 0x5555
        let block = build_block(0x3C00, 0x5555, [0xFF; 4], [0x00; 128]);
        let mut q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        q8[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..256 {
            q8[4 + i] = 1i8 as u8;
        }
        // Each weight = 1 * -1 * -127 = 127. Sum = 256 * 127 = 32512.
        assert_eq!(dot_iq4xs_q8k_block(&block, &q8), 32512.0);
    }

    #[test]
    fn dequant_multi_block() {
        let b0 = build_block(0x3C00, 0xAAAA, [0x11; 4], [0x00; 128]); // every elem = -127.0
        let b1 = build_block(0x4000, 0xAAAA, [0x11; 4], [0x00; 128]); // every elem = -254.0 (d=2)

        let mut bytes = Vec::with_capacity(BYTES_PER_BLOCK * 2);
        bytes.extend_from_slice(&b0);
        bytes.extend_from_slice(&b1);
        let mut out = vec![0.0f32; BLOCK_SIZE * 2];
        dequant(&bytes, &mut out);
        assert!(out[..BLOCK_SIZE].iter().all(|&v| v == -127.0));
        assert!(out[BLOCK_SIZE..].iter().all(|&v| v == -254.0));
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn dequant_block_rejects_short_out() {
        let block = build_block(0, 0, [0; 4], [0; 128]);
        let mut out = [0.0f32; BLOCK_SIZE - 1];
        dequant_block(&block, &mut out);
    }

    #[test]
    #[should_panic(expected = "not multiple of 136")]
    fn dequant_rejects_partial_block() {
        let bytes = vec![0u8; BYTES_PER_BLOCK - 1];
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequant(&bytes, &mut out);
    }
}
