//! IQ2_XXS block dequantisation + Q8_K dot kernel.
//!
//! Block layout (66 bytes, 256 elements):
//! * `d`      : f16 scale           (2 bytes, offset 0)
//! * `qs`     : 32 × uint16_t       (64 bytes, offset 2)
//!
//! The 256 elements are divided into 8 sub-blocks of 32 elements each.
//! Each sub-block consumes 4 uint16_t (8 bytes):
//!   * The first uint32_t (low 4 bytes of the 8-byte chunk) holds four 8-bit
//!     grid indices.
//!   * The second uint32_t (high 4 bytes) holds four 7-bit sign indices in
//!     its low 28 bits and a 4-bit scale multiplier in bits 28–31.
//!
//! Dequantised value:
//!   `x = d * (0.5 + scale) * 0.25 * grid_value * sign`
//!
//! The dot kernel uses a precomputed `iq2xxs_signed_grid` table so the
//! per-element sign application happens once at init rather than every call.
//!
//! Reference: `block_iq2_xxs`, `dequantize_row_iq2_xxs`, and
//! `ds4_vec_dot_iq2_xxs_q8_K` in antirez/ds4 ds4.c.

use std::sync::LazyLock;


use crate::quant::q8_0;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 66;

/// Byte offsets inside one IQ2_XXS block.
pub mod offset {
    pub const D: usize = 0; // f16
    pub const QS: usize = 2; // uint16_t × 32
}

// ---------------------------------------------------------------------------
// Lookup tables (identical to ggml / antirez/ds4)
// ---------------------------------------------------------------------------

static IQ2_XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

static KSIGNS_IQ2XS: [u8; 128] = [
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

/// Precomputed signed grid: `[grid_index][sign_index][8]` → signed i8.
///
/// Initialised once on first access via `LazyLock`. The table is ~256 KiB.
static SIGNED_GRID: LazyLock<[[[i8; 8]; 128]; 256]> = LazyLock::new(|| {
    let mut table = [[[0i8; 8]; 128]; 256];
    for g in 0..256 {
        let grid_u64 = IQ2_XXS_GRID[g];
        for s in 0..128 {
            let signs = KSIGNS_IQ2XS[s];
            for j in 0..8 {
                let v = ((grid_u64 >> (8 * j)) & 0xFF) as i32;
                let neg = (signs & KMASK_IQ2XS[j]) != 0;
                table[g][s][j] = if neg { -(v as i8) } else { v as i8 };
            }
        }
    }
    table
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn read_qs_u16(block: &[u8; BYTES_PER_BLOCK]) -> [u16; 32] {
    let mut out = [0u16; 32];
    for i in 0..32 {
        out[i] = u16::from_le_bytes([
            block[offset::QS + i * 2],
            block[offset::QS + i * 2 + 1],
        ]);
    }
    out
}

// ---------------------------------------------------------------------------
// Dequantisation
// ---------------------------------------------------------------------------

/// Dequantise one 66-byte IQ2_XXS block into 256 f32s.
///
/// Panics if `out` is not exactly `BLOCK_SIZE` long.
pub fn dequant_block(block: &[u8; BYTES_PER_BLOCK], out: &mut [f32]) {
    assert_eq!(
        out.len(),
        BLOCK_SIZE,
        "iq2_xxs::dequant_block: out len {} != {BLOCK_SIZE}",
        out.len(),
    );

    let d = q8_0::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs_u16 = read_qs_u16(block);
    let mut out_off = 0usize;

    for ib32 in 0..(BLOCK_SIZE / 32) {
        let aux0 = u32::from_le_bytes([
            (qs_u16[ib32 * 4 + 0] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 0] >> 8) as u8,
            (qs_u16[ib32 * 4 + 1] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 1] >> 8) as u8,
        ]);
        let aux1 = u32::from_le_bytes([
            (qs_u16[ib32 * 4 + 2] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 2] >> 8) as u8,
            (qs_u16[ib32 * 4 + 3] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 3] >> 8) as u8,
        ]);

        let scale = (aux1 >> 28) as f32;
        let db = d * (0.5f32 + scale) * 0.25f32;

        let aux8_0 = (aux0 & 0xFF) as u8;
        let aux8_1 = ((aux0 >> 8) & 0xFF) as u8;
        let aux8_2 = ((aux0 >> 16) & 0xFF) as u8;
        let aux8_3 = ((aux0 >> 24) & 0xFF) as u8;

        let grid_vals = [
            IQ2_XXS_GRID[aux8_0 as usize],
            IQ2_XXS_GRID[aux8_1 as usize],
            IQ2_XXS_GRID[aux8_2 as usize],
            IQ2_XXS_GRID[aux8_3 as usize],
        ];

        for l in 0..4 {
            let sign_idx = ((aux1 >> (7 * l)) & 127) as usize;
            let signs = KSIGNS_IQ2XS[sign_idx];
            let grid_u64 = grid_vals[l];
            for j in 0..8 {
                let v = ((grid_u64 >> (8 * j)) & 0xFF) as f32;
                let neg = (signs & KMASK_IQ2XS[j]) != 0;
                out[out_off + j] = db * v * if neg { -1.0f32 } else { 1.0f32 };
            }
            out_off += 8;
        }
    }
}

/// Dequantise a contiguous sequence of IQ2_XXS blocks.
///
/// `bytes.len()` must be a multiple of `BYTES_PER_BLOCK` and `out.len()` must
/// be the corresponding multiple of `BLOCK_SIZE`.
pub fn dequant(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(
        bytes.len() % BYTES_PER_BLOCK,
        0,
        "iq2_xxs::dequant: bytes len {} not multiple of {BYTES_PER_BLOCK}",
        bytes.len(),
    );
    let n_blocks = bytes.len() / BYTES_PER_BLOCK;
    assert_eq!(
        out.len(),
        n_blocks * BLOCK_SIZE,
        "iq2_xxs::dequant: out len {} != {n_blocks} * {BLOCK_SIZE}",
        out.len(),
    );
    for (i, chunk) in bytes.chunks_exact(BYTES_PER_BLOCK).enumerate() {
        let block: &[u8; BYTES_PER_BLOCK] = chunk.try_into().unwrap();
        dequant_block(block, &mut out[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]);
    }
}

// ---------------------------------------------------------------------------
// Dot kernel (Q8_K activation × IQ2_XXS weight)
// ---------------------------------------------------------------------------

/// Dot product of one IQ2_XXS weight block against one Q8_K activation block.
///
/// Both blocks cover exactly 256 elements. Returns the scalar
/// `sum_k weight[k] * act[k]`.
pub fn dot_iq2xxs_q8k_block(iq2_block: &[u8; BYTES_PER_BLOCK], q8_block: &[u8]) -> f32 {
    use crate::quant::q8_k;

    debug_assert_eq!(q8_block.len(), q8_k::BYTES_PER_BLOCK);

    let iq2_d = q8_0::f16_to_f32(u16::from_le_bytes([iq2_block[0], iq2_block[1]]));
    let q8_d = f32::from_le_bytes(q8_block[q8_k::offset::D..q8_k::offset::D + 4].try_into().unwrap());
    let d = iq2_d * q8_d;

    let qs_u16 = read_qs_u16(iq2_block);
    let q8_qs = &q8_block[q8_k::offset::QS..q8_k::offset::QS + 256];

    let signed_grid = &*SIGNED_GRID;

    let mut bsum = 0i32;
    let mut q8_off = 0usize;

    for ib32 in 0..(BLOCK_SIZE / 32) {
        let aux0 = u32::from_le_bytes([
            (qs_u16[ib32 * 4 + 0] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 0] >> 8) as u8,
            (qs_u16[ib32 * 4 + 1] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 1] >> 8) as u8,
        ]);
        let aux1 = u32::from_le_bytes([
            (qs_u16[ib32 * 4 + 2] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 2] >> 8) as u8,
            (qs_u16[ib32 * 4 + 3] & 0xFF) as u8,
            (qs_u16[ib32 * 4 + 3] >> 8) as u8,
        ]);

        let ls = (2 * (aux1 >> 28) + 1) as i32;
        let aux8 = [
            (aux0 & 0xFF) as usize,
            ((aux0 >> 8) & 0xFF) as usize,
            ((aux0 >> 16) & 0xFF) as usize,
            ((aux0 >> 24) & 0xFF) as usize,
        ];

        let mut sumi = 0i32;
        for l in (0..4).step_by(2) {
            let sign0 = ((aux1 >> (7 * l)) & 127) as usize;
            let sign1 = ((aux1 >> (7 * (l + 1))) & 127) as usize;
            sumi += dot_iq2_pair_16(&signed_grid[aux8[l]][sign0],
                                    &signed_grid[aux8[l + 1]][sign1],
                                    &q8_qs[q8_off..q8_off + 16]);
            q8_off += 16;
        }
        bsum += sumi * ls;
    }

    0.125f32 * d * (bsum as f32)
}

/// 16-wide dot product: two signed 8-element IQ2_XXS grids against 16 Q8_K qs.
fn dot_iq2_pair_16(grid0: &[i8; 8], grid1: &[i8; 8], q8: &[u8]) -> i32 {
    debug_assert_eq!(q8.len(), 16);
    let mut sum = 0i32;
    for i in 0..8 {
        sum += (grid0[i] as i32) * (q8[i] as i8 as i32);
    }
    for i in 0..8 {
        sum += (grid1[i] as i32) * (q8[8 + i] as i8 as i32);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(d_bits: u16, qs: [u16; 32]) -> [u8; BYTES_PER_BLOCK] {
        let mut block = [0u8; BYTES_PER_BLOCK];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for (i, &q) in qs.iter().enumerate() {
            block[offset::QS + i * 2..offset::QS + i * 2 + 2].copy_from_slice(&q.to_le_bytes());
        }
        block
    }

    #[test]
    fn dequant_block_all_zero() {
        let block = build_block(0x0000, [0u16; 32]);
        let mut out = vec![123.0f32; BLOCK_SIZE];
        dequant_block(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn signed_grid_init_sanity() {
        // Grid 0 is all 0x08 (8). Sign 0 has no bits set -> all positive.
        assert_eq!(SIGNED_GRID[0][0][0], 8);
        assert_eq!(SIGNED_GRID[0][0][7], 8);

        // Sign 255 (index 127 in KSIGNS_IQ2XS) = 255 -> all 8 bits set -> all negative.
        assert_eq!(SIGNED_GRID[0][127][0], -8);
        assert_eq!(SIGNED_GRID[0][127][7], -8);
    }

    #[test]
    fn dot_block_against_zeros_is_zero() {
        let block = build_block(0x3C00, [0u16; 32]);
        let q8 = vec![0u8; crate::quant::q8_k::BYTES_PER_BLOCK];
        assert_eq!(dot_iq2xxs_q8k_block(&block, &q8), 0.0);
    }

    #[test]
    fn dot_block_matches_dequant_dot_simple() {
        // Build an IQ2_XXS block where every sub-block uses grid 0 (all 8s)
        // and sign 0 (all positive), with scale = 0.
        // dequant = d * (0.5 + 0) * 0.25 * 8 = d * 1.0
        // If d = 1.0, every weight = 1.0.
        let d_bits = 0x3C00; // 1.0

        // Each sub-block: aux32[1] >> 28 = 0 (scale), sign indices all 0.
        // aux32[0] = 0x0000_0000 (grid indices all 0)
        // aux32[1] = 0x0000_0000 (sign indices all 0, scale=0)
        // In u16 layout: qs[4*ib32+0] = 0, qs[4*ib32+1] = 0,
        //                qs[4*ib32+2] = 0, qs[4*ib32+3] = 0
        let qs = [0u16; 32];
        let iq2 = build_block(d_bits, qs);

        // Q8_K: d=1.0, all qs=1 -> dot = 256 * 1.0 * 1 = 256
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

        let dot_kernel = dot_iq2xxs_q8k_block(&iq2, &q8);

        // Also compute via dequant for cross-check.
        let mut w = vec![0.0f32; BLOCK_SIZE];
        dequant_block(&iq2, &mut w);
        let dot_ref: f32 = w.iter().zip(q8[4..260].iter().map(|&b| b as i8 as f32)).map(|(&a, b)| a * b).sum();

        assert!(
            (dot_kernel - dot_ref).abs() < 1e-3,
            "kernel {dot_kernel} != ref {dot_ref}"
        );
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn dequant_block_rejects_short_out() {
        let block = build_block(0x3C00, [0; 32]);
        let mut out = [0.0f32; BLOCK_SIZE - 1];
        dequant_block(&block, &mut out);
    }

    #[test]
    #[should_panic(expected = "not multiple of 66")]
    fn dequant_rejects_partial_block() {
        let bytes = vec![0u8; BYTES_PER_BLOCK - 1];
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequant(&bytes, &mut out);
    }
}
