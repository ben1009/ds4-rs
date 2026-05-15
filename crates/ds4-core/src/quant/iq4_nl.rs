//! IQ4_NL block dequantisation + f32 dot kernel.
//!
//! Block layout (18 bytes, 32 elements):
//! * `d`  : f16 scale       (2 bytes, offset 0)
//! * `qs` : 16 × u8         (16 bytes, offset 2) Sixteen bytes pack 32 codebook indices: byte `i`
//!   holds the low-nibble index for element `i` and the high-nibble index for element `i + 16`.
//!
//! Dequantised value:
//!   `x = d * kvalues_iq4nl[index]`
//!
//! Unlike the K-quant variants, IQ4_NL has 32-element blocks. The matmul path
//! does not pre-quantise activations to Q8_K; instead it dequantises one block
//! (32 f32s) at a time and dots against the f32 activation chunk directly.
//!
//! Reference: `block_iq4_nl`, `dequantize_row_iq4_nl` in ggml-quants.c.

use crate::quant::{iq4_codebook::KVALUES_IQ4NL, q8_0};

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 18;

pub mod offset {
    pub const D: usize = 0; // f16
    pub const QS: usize = 2; // u8 × 16
}

/// Dequantise one 18-byte IQ4_NL block into 32 f32s.
pub fn dequant_block(block: &[u8; BYTES_PER_BLOCK], out: &mut [f32]) {
    assert_eq!(
        out.len(),
        BLOCK_SIZE,
        "iq4_nl::dequant_block: out len {} != {BLOCK_SIZE}",
        out.len(),
    );
    let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[offset::QS..offset::QS + 16];
    for i in 0..16 {
        out[i] = d * KVALUES_IQ4NL[(qs[i] & 0x0F) as usize] as f32;
        out[i + 16] = d * KVALUES_IQ4NL[(qs[i] >> 4) as usize] as f32;
    }
}

/// Dequantise a contiguous sequence of IQ4_NL blocks.
pub fn dequant(bytes: &[u8], out: &mut [f32]) {
    assert!(
        bytes.len().is_multiple_of(BYTES_PER_BLOCK),
        "iq4_nl::dequant: bytes len {} not multiple of {BYTES_PER_BLOCK}",
        bytes.len(),
    );
    let n_blocks = bytes.len() / BYTES_PER_BLOCK;
    assert_eq!(
        out.len(),
        n_blocks * BLOCK_SIZE,
        "iq4_nl::dequant: out len {} != {n_blocks} * {BLOCK_SIZE}",
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

/// Dot product of one IQ4_NL block against a 32-wide f32 activation chunk.
///
/// `act` covers the 32 elements aligned with this block. Returns the scalar
/// `sum_k weight[k] * act[k]`.
pub fn dot_iq4nl_f32_block(block: &[u8; BYTES_PER_BLOCK], act: &[f32; BLOCK_SIZE]) -> f32 {
    let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[offset::QS..offset::QS + 16];
    let mut acc = 0.0f32;
    for i in 0..16 {
        acc += KVALUES_IQ4NL[(qs[i] & 0x0F) as usize] as f32 * act[i];
        acc += KVALUES_IQ4NL[(qs[i] >> 4) as usize] as f32 * act[i + 16];
    }
    d * acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(d_bits: u16, qs: [u8; 16]) -> [u8; BYTES_PER_BLOCK] {
        let mut block = [0u8; BYTES_PER_BLOCK];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        block[offset::QS..offset::QS + 16].copy_from_slice(&qs);
        block
    }

    #[test]
    fn dequant_block_zero_d_yields_zeros() {
        let block = build_block(0x0000, [0xFF; 16]);
        let mut out = [123.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_block_unit_scale_index_zero() {
        // d=1.0, qs all zero -> all weights = kvalues_iq4nl[0] = -127.
        let block = build_block(0x3C00, [0x00; 16]);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, -127.0);
        }
    }

    #[test]
    fn dequant_block_index_layout_low_high() {
        // Pack two distinct indices into byte 0: low nibble = 8, high nibble = 15.
        // kvalues_iq4nl[8] = 1, kvalues_iq4nl[15] = 113.
        // After dequant: out[0] = 1.0, out[16] = 113.0; the rest = -127.0.
        let mut qs = [0u8; 16];
        qs[0] = 0xF8; // high=0xF, low=0x8
        let block = build_block(0x3C00, qs);
        let mut out = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[16], 113.0);
        assert_eq!(out[1], -127.0);
        assert_eq!(out[17], -127.0);
    }

    #[test]
    fn dot_block_against_zero_act_is_zero() {
        let block = build_block(0x3C00, [0x77; 16]);
        let act = [0.0f32; BLOCK_SIZE];
        assert_eq!(dot_iq4nl_f32_block(&block, &act), 0.0);
    }

    #[test]
    fn dot_block_constant_index_zero() {
        // d=1.0, all indices = 0 -> codebook[0] = -127. act = 1 -> dot = -127 * 32.
        let block = build_block(0x3C00, [0x00; 16]);
        let act = [1.0f32; BLOCK_SIZE];
        assert_eq!(dot_iq4nl_f32_block(&block, &act), -127.0 * 32.0);
    }

    #[test]
    fn dot_block_matches_dequant_then_dot() {
        // Random-ish block + activations: kernel result must equal manual
        // dequant + f32 dot.
        let mut qs = [0u8; 16];
        for (i, b) in qs.iter_mut().enumerate() {
            *b = ((i as u8) * 17).wrapping_add(3);
        }
        // d = 0.5 (f16 0x3800)
        let block = build_block(0x3800, qs);

        let act: [f32; BLOCK_SIZE] = std::array::from_fn(|i| (i as f32 * 0.13 - 1.0).sin());

        let mut deq = [0.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut deq);
        let manual: f32 = deq.iter().zip(act.iter()).map(|(w, a)| w * a).sum();

        let kernel = dot_iq4nl_f32_block(&block, &act);
        let tol = manual.abs() * 1e-6 + 1e-5;
        assert!(
            (kernel - manual).abs() < tol,
            "kernel {kernel} vs manual {manual}",
        );
    }

    #[test]
    fn dequant_multi_block() {
        let b0 = build_block(0x3C00, [0x00; 16]); // -127.0 throughout
        let b1 = build_block(0x4000, [0x00; 16]); // -254.0 throughout
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
        let block = build_block(0, [0; 16]);
        let mut out = [0.0f32; BLOCK_SIZE - 1];
        dequant_block(&block, &mut out);
    }

    #[test]
    #[should_panic(expected = "not multiple of 18")]
    fn dequant_rejects_partial_block() {
        let bytes = vec![0u8; BYTES_PER_BLOCK - 1];
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequant(&bytes, &mut out);
    }
}
