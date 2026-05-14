//! Q8_0 block dequantisation.
//!
//! Block layout (34 bytes, 32 elements):
//! - `d`: f16 scale (little-endian, bytes 0..2)
//! - `qs`: 32 × i8 quants (bytes 2..34)
//!
//! Dequantised value: `d * qs[i]`.
//!
//! Reference: `block_q8_0` in ggml / antirez/ds4.

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 34;

/// Convert an IEEE 754 half-precision `u16` to `f32`.
///
/// Hand-rolled so we don't pull in the `half` crate. Matches
/// `__half2float` for all finite values, subnormals, and +/-inf/NaN.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            // +/- zero
            sign << 31
        } else {
            // subnormal: normalise into f32 representation
            let mut m = mant;
            let mut e: i32 = 1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (e + 127 - 15) as u32;
            (sign << 31) | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // infinity / NaN — propagate mantissa into the f32 slot
        (sign << 31) | (0xFFu32 << 23) | (mant << 13)
    } else {
        // normal
        let f32_exp = exp + 127 - 15;
        (sign << 31) | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Dequantise one 34-byte Q8_0 block into 32 f32s.
///
/// Panics if `out` is not exactly `BLOCK_SIZE` long.
pub fn dequant_block(block: &[u8; BYTES_PER_BLOCK], out: &mut [f32]) {
    assert_eq!(
        out.len(),
        BLOCK_SIZE,
        "dequant_block: out len {} != {BLOCK_SIZE}",
        out.len(),
    );
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..BLOCK_SIZE {
        let q = block[2 + i] as i8;
        out[i] = d * q as f32;
    }
}

/// Dequantise a contiguous sequence of Q8_0 blocks.
///
/// `bytes.len()` must be a multiple of `BYTES_PER_BLOCK` and `out.len()` must
/// be the corresponding multiple of `BLOCK_SIZE`.
pub fn dequant(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(
        bytes.len() % BYTES_PER_BLOCK,
        0,
        "dequant: bytes len {} not multiple of {BYTES_PER_BLOCK}",
        bytes.len(),
    );
    let n_blocks = bytes.len() / BYTES_PER_BLOCK;
    assert_eq!(
        out.len(),
        n_blocks * BLOCK_SIZE,
        "dequant: out len {} != {n_blocks} * {BLOCK_SIZE}",
        out.len(),
    );
    for (i, chunk) in bytes.chunks_exact(BYTES_PER_BLOCK).enumerate() {
        let block: &[u8; BYTES_PER_BLOCK] = chunk.try_into().unwrap();
        dequant_block(block, &mut out[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(scale: u16, quants: [i8; 32]) -> [u8; BYTES_PER_BLOCK] {
        let mut block = [0u8; BYTES_PER_BLOCK];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        for (i, &q) in quants.iter().enumerate() {
            block[2 + i] = q as u8;
        }
        block
    }

    #[test]
    fn f16_to_f32_zero_and_one() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        // 1.0 in f16 = 0x3C00 (exp 15, mant 0)
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // -1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        // 2.0
        assert_eq!(f16_to_f32(0x4000), 2.0);
        // 0.5
        assert_eq!(f16_to_f32(0x3800), 0.5);
    }

    #[test]
    fn f16_to_f32_inf_nan() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00).is_sign_positive());
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00).is_sign_negative());
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_to_f32_subnormal() {
        // Smallest positive subnormal f16: 2^-24 ≈ 5.96e-8
        let v = f16_to_f32(0x0001);
        assert!((v - 2.0f32.powi(-24)).abs() < 1e-12);
    }

    #[test]
    fn dequant_block_trivial_scale_one() {
        // d = 1.0 => output == quants as f32
        let quants: [i8; 32] = std::array::from_fn(|i| i as i8 - 16);
        let block = build_block(0x3C00, quants); // 1.0
        let mut out = [0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for i in 0..BLOCK_SIZE {
            assert_eq!(out[i], quants[i] as f32);
        }
    }

    #[test]
    fn dequant_block_applies_scale() {
        // d = 0.5 => output == quants * 0.5
        let quants: [i8; 32] = std::array::from_fn(|_| 4);
        let block = build_block(0x3800, quants); // 0.5
        let mut out = [0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for v in out {
            assert_eq!(v, 2.0);
        }
    }

    #[test]
    fn dequant_block_handles_negative_scale_and_quants() {
        // d = -1.0, quants all 127 => output all -127.0
        let quants = [127i8; 32];
        let block = build_block(0xBC00, quants);
        let mut out = [0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for v in out {
            assert_eq!(v, -127.0);
        }
    }

    #[test]
    fn dequant_multi_block() {
        let b0 = build_block(0x3C00, [1i8; 32]); // d=1.0
        let b1 = build_block(0x4000, [2i8; 32]); // d=2.0
        let mut bytes = Vec::with_capacity(BYTES_PER_BLOCK * 2);
        bytes.extend_from_slice(&b0);
        bytes.extend_from_slice(&b1);
        let mut out = vec![0f32; BLOCK_SIZE * 2];
        dequant(&bytes, &mut out);
        assert!(out[..BLOCK_SIZE].iter().all(|&v| v == 1.0));
        assert!(out[BLOCK_SIZE..].iter().all(|&v| v == 4.0));
    }

    #[test]
    #[should_panic(expected = "not multiple of 34")]
    fn dequant_rejects_partial_block() {
        let bytes = vec![0u8; 33];
        let mut out = vec![0f32; 32];
        dequant(&bytes, &mut out);
    }

    #[test]
    fn f16_to_f32_max_finite() {
        // 0x7BFF = 65504, the largest finite f16.
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        // -65504
        assert_eq!(f16_to_f32(0xFBFF), -65504.0);
    }

    #[test]
    fn f16_to_f32_negative_subnormal() {
        let v = f16_to_f32(0x8001);
        assert!((v + 2.0f32.powi(-24)).abs() < 1e-12);
    }

    #[test]
    fn dequant_block_zero_scale_produces_zeros() {
        let quants: [i8; 32] = std::array::from_fn(|i| i as i8 - 16);
        let block = build_block(0x0000, quants);
        let mut out = [0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_block_handles_min_i8_quant() {
        // d = 1.0, quant = -128 => output = -128.0
        let mut quants = [0i8; 32];
        quants[5] = -128;
        quants[10] = 127;
        let block = build_block(0x3C00, quants);
        let mut out = [0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        assert_eq!(out[5], -128.0);
        assert_eq!(out[10], 127.0);
    }

    #[test]
    fn dequant_block_alternating_signs() {
        // d = 0.25, quants alternate +/-1 -> output alternates +/-0.25
        let quants: [i8; 32] = std::array::from_fn(|i| if i % 2 == 0 { 1 } else { -1 });
        let block = build_block(0x3400, quants); // 0.25
        let mut out = [0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            let expected = if i % 2 == 0 { 0.25 } else { -0.25 };
            assert_eq!(*v, expected);
        }
    }

    #[test]
    fn dequant_empty_input_is_noop() {
        let bytes: Vec<u8> = Vec::new();
        let mut out: Vec<f32> = Vec::new();
        dequant(&bytes, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn dequant_three_blocks_varied_scales_and_quants() {
        let b0 = build_block(0x0000, [50i8; 32]); // d=0 -> all zero
        let b1 = build_block(0x3C00, std::array::from_fn(|i| i as i8 - 16)); // d=1
        let b2 = build_block(0xBC00, std::array::from_fn(|i| -(i as i8) + 8)); // d=-1
        let mut bytes = Vec::with_capacity(BYTES_PER_BLOCK * 3);
        bytes.extend_from_slice(&b0);
        bytes.extend_from_slice(&b1);
        bytes.extend_from_slice(&b2);
        let mut out = vec![0f32; BLOCK_SIZE * 3];
        dequant(&bytes, &mut out);

        for v in &out[..BLOCK_SIZE] {
            assert_eq!(*v, 0.0);
        }
        for (i, v) in out[BLOCK_SIZE..2 * BLOCK_SIZE].iter().enumerate() {
            assert_eq!(*v, (i as i8 - 16) as f32);
        }
        for (i, v) in out[2 * BLOCK_SIZE..].iter().enumerate() {
            assert_eq!(*v, -((-(i as i8) + 8) as f32));
        }
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn dequant_block_rejects_wrong_out_len() {
        let block = build_block(0x3C00, [0i8; 32]);
        let mut out = [0f32; 16];
        dequant_block(&block, &mut out);
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn dequant_rejects_mismatched_out_len() {
        let bytes = vec![0u8; BYTES_PER_BLOCK * 2];
        let mut out = vec![0f32; BLOCK_SIZE]; // should be 64
        dequant(&bytes, &mut out);
    }
}
