//! Generate reference vectors for the routed-expert quant kernels.
//!
//! Run via `scripts/regen_vectors.sh routed_quants` (preferred) or:
//!
//! ```text
//! cargo run --quiet -p ds4-core --example gen_vectors_routed_quants
//! ```
//!
//! Five formats, each landing two `.bin` files under `tests/vectors/`:
//!
//! | Format   | Dequant vector              | Dot vector                 |
//! |----------|-----------------------------|----------------------------|
//! | IQ2_XXS  | `iq2_xxs_dequant.bin` (256) | `iq2_xxs_dot.bin`  (1 f32) |
//! | Q2_K     | `q2_k_dequant.bin`    (256) | `q2_k_dot.bin`     (1 f32) |
//! | Q4_K     | `q4_k_dequant.bin`    (256) | `q4_k_dot.bin`     (1 f32) |
//! | IQ4_XS   | `iq4_xs_dequant.bin`  (256) | `iq4_xs_dot.bin`   (1 f32) |
//! | IQ4_NL   | `iq4_nl_dequant.bin`  (32)  | `iq4_nl_dot.bin`   (1 f32) |
//!
//! Q2_K / IQ2_XXS / Q4_K / IQ4_XS dot kernels score a Q8_K-quantised
//! activation against the weight block; IQ4_NL takes f32 activations
//! directly because its 32-element block is too small for Q8_K's 256-wide
//! framing. All inputs are deterministic so regeneration is bit-exact.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use ds4_core::quant::{iq2_xxs, iq4_nl, iq4_xs, q2_k, q4_k, q8_k};

fn main() {
    let dir = find_vectors_dir();

    write_iq2_xxs(&dir);
    write_q2_k(&dir);
    write_q4_k(&dir);
    write_iq4_xs(&dir);
    write_iq4_nl(&dir);

    eprintln!("Wrote routed-expert reference vectors to {}", dir.display());
}

// =========================================================================
// IQ2_XXS — gate/up dtype
// =========================================================================

fn write_iq2_xxs(dir: &Path) {
    // Block layout: f16 d at offset 0, then 32 × u16 qs at offset 2.
    // Pick a non-trivial scale and a varied qs pattern (rotating across
    // grid / sign / scale fields) to exercise more than one branch of the
    // dequant loop.
    let d = 0x3800u16; // 0.5
    let mut qs = [0u16; 32];
    for (i, q) in qs.iter_mut().enumerate() {
        // Mix bits across the four 16-bit halves of each sub-block:
        //   * grid index (low 8 bits of qs[4*ib32 + 0..2])
        //   * sign + scale (low 16 of aux32[1] = qs[4*ib32 + 2..4])
        // Vary by index to surface scale/grid interactions.
        *q = ((i as u16).wrapping_mul(0x37) ^ 0x5A93).wrapping_add(i as u16 * 11);
    }
    let block = pack_iq2_xxs(d, qs);

    // Dequant.
    let mut deq = vec![0f32; iq2_xxs::BLOCK_SIZE];
    iq2_xxs::dequant_block(&block, &mut deq);
    write_f32_le(&dir.join("iq2_xxs_dequant.bin"), &deq);

    // Dot against a Q8_K activation block.
    let q8 = build_synthetic_q8k(0.125, 0);
    let dot = iq2_xxs::dot_iq2xxs_q8k_block(&block, &q8);
    write_f32_le(&dir.join("iq2_xxs_dot.bin"), &[dot]);
}

fn pack_iq2_xxs(d_bits: u16, qs: [u16; 32]) -> [u8; iq2_xxs::BYTES_PER_BLOCK] {
    let mut block = [0u8; iq2_xxs::BYTES_PER_BLOCK];
    block[..2].copy_from_slice(&d_bits.to_le_bytes());
    for (i, &q) in qs.iter().enumerate() {
        let off = iq2_xxs::offset::QS + i * 2;
        block[off..off + 2].copy_from_slice(&q.to_le_bytes());
    }
    block
}

// =========================================================================
// Q2_K — routed-expert down dtype (variant)
// =========================================================================

fn write_q2_k(dir: &Path) {
    // Layout: d (f16, off 0), dmin (f16, off 2), scales (16 × u8, off 4),
    // qs (64 × u8, off 20).
    let d = 0x3C00u16; // 1.0
    let dmin = 0x3800u16; // 0.5
    let mut scales = [0u8; 16];
    for (i, s) in scales.iter_mut().enumerate() {
        // High nibble = min index, low nibble = scale index. Vary both.
        *s = ((i as u8).wrapping_mul(3) & 0x0F) | (((i as u8).wrapping_mul(5) << 4) & 0xF0);
    }
    let mut qs = [0u8; 64];
    for (i, q) in qs.iter_mut().enumerate() {
        // Two-bit quants packed 4 per byte; vary so different rows pull
        // different (q, scale) combinations.
        *q = (i as u8).wrapping_mul(0x37).wrapping_add(0x11);
    }
    let mut block = [0u8; q2_k::BYTES_PER_BLOCK];
    block[..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&dmin.to_le_bytes());
    block[q2_k::offset::SCALES..q2_k::offset::SCALES + 16].copy_from_slice(&scales);
    block[q2_k::offset::QS..q2_k::offset::QS + 64].copy_from_slice(&qs);

    let mut deq = vec![0f32; q2_k::BLOCK_SIZE];
    q2_k::dequant_block(&block, &mut deq);
    write_f32_le(&dir.join("q2_k_dequant.bin"), &deq);

    let q8 = build_synthetic_q8k(0.0625, 1);
    let dot = q2_k::dot_q2k_q8k_block(&block, &q8);
    write_f32_le(&dir.join("q2_k_dot.bin"), &[dot]);
}

// =========================================================================
// Q4_K — routed-expert down dtype (variant)
// =========================================================================

fn write_q4_k(dir: &Path) {
    // Layout: d (f16, off 0), dmin (f16, off 2), scales (12 × u8, off 4),
    // qs (128 × u8, off 16). The 12-byte scales pack 8 sub-block scale/min
    // pairs via `get_scale_min_k4`. We just write deterministic bytes —
    // dequant + dot are pure transforms of those inputs.
    let d = 0x3C00u16; // 1.0
    let dmin = 0x3400u16; // 0.25
    let mut scales = [0u8; 12];
    for (i, s) in scales.iter_mut().enumerate() {
        *s = ((i as u8).wrapping_mul(7) ^ 0x29).wrapping_add(0x10);
    }
    let mut qs = [0u8; 128];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = (i as u8).wrapping_mul(0x1B).wrapping_add(0x05);
    }
    let mut block = [0u8; q4_k::BYTES_PER_BLOCK];
    block[..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&dmin.to_le_bytes());
    block[q4_k::offset::SCALES..q4_k::offset::SCALES + 12].copy_from_slice(&scales);
    block[q4_k::offset::QS..q4_k::offset::QS + 128].copy_from_slice(&qs);

    let mut deq = vec![0f32; q4_k::BLOCK_SIZE];
    q4_k::dequant_block(&block, &mut deq);
    write_f32_le(&dir.join("q4_k_dequant.bin"), &deq);

    let q8 = build_synthetic_q8k(0.03125, 2);
    let dot = q4_k::dot_q4k_q8k_block(&block, &q8);
    write_f32_le(&dir.join("q4_k_dot.bin"), &[dot]);
}

// =========================================================================
// IQ4_XS — gate/up dtype (variant)
// =========================================================================

fn write_iq4_xs(dir: &Path) {
    // Layout: d (f16, off 0), scales_h (u16, off 2), scales_l (4 × u8, off 4),
    // qs (128 × u8, off 8). Eight 6-bit signed scales (raw value − 32).
    let d = 0x3800u16; // 0.5
    let scales_h = 0x9C5Au16;
    let scales_l = [0xA1u8, 0x37, 0xC4, 0x6Du8];
    let mut qs = [0u8; 128];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = (i as u8).wrapping_mul(0x4D).wrapping_add(0x12);
    }
    let mut block = [0u8; iq4_xs::BYTES_PER_BLOCK];
    block[..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&scales_h.to_le_bytes());
    block[iq4_xs::offset::SCALES_L..iq4_xs::offset::SCALES_L + 4].copy_from_slice(&scales_l);
    block[iq4_xs::offset::QS..iq4_xs::offset::QS + 128].copy_from_slice(&qs);

    let mut deq = vec![0f32; iq4_xs::BLOCK_SIZE];
    iq4_xs::dequant_block(&block, &mut deq);
    write_f32_le(&dir.join("iq4_xs_dequant.bin"), &deq);

    let q8 = build_synthetic_q8k(0.0125, 3);
    let dot = iq4_xs::dot_iq4xs_q8k_block(&block, &q8);
    write_f32_le(&dir.join("iq4_xs_dot.bin"), &[dot]);
}

// =========================================================================
// IQ4_NL — gate/up dtype (variant), 32-element block
// =========================================================================

fn write_iq4_nl(dir: &Path) {
    let d = 0x3C00u16; // 1.0
    let mut qs = [0u8; 16];
    for (i, q) in qs.iter_mut().enumerate() {
        // Pack two distinct codebook indices per byte; vary them.
        let lo = (i as u8).wrapping_mul(3) & 0x0F;
        let hi = ((i as u8).wrapping_mul(5).wrapping_add(7)) & 0x0F;
        *q = lo | (hi << 4);
    }
    let mut block = [0u8; iq4_nl::BYTES_PER_BLOCK];
    block[..2].copy_from_slice(&d.to_le_bytes());
    block[iq4_nl::offset::QS..iq4_nl::offset::QS + 16].copy_from_slice(&qs);

    let mut deq = vec![0f32; iq4_nl::BLOCK_SIZE];
    iq4_nl::dequant_block(&block, &mut deq);
    write_f32_le(&dir.join("iq4_nl_dequant.bin"), &deq);

    // Deterministic 32-wide f32 activations: sawtooth in [-1, 1].
    let mut act = [0f32; iq4_nl::BLOCK_SIZE];
    for (i, v) in act.iter_mut().enumerate() {
        let phase = (i % 16) as i32;
        let s = if phase <= 8 { phase - 4 } else { 12 - phase };
        *v = s as f32 / 4.0;
    }
    let dot = iq4_nl::dot_iq4nl_f32_block(&block, &act);
    write_f32_le(&dir.join("iq4_nl_dot.bin"), &[dot]);
}

// =========================================================================
// Q8_K activation builder shared by IQ2_XXS / Q2_K / Q4_K / IQ4_XS
// =========================================================================

/// Build a Q8_K activation block by quantising a deterministic 256-wide
/// f32 vector. `amp` scales the triangular wave; `salt` perturbs the phase
/// so different formats see different (but reproducible) activations.
fn build_synthetic_q8k(amp: f32, salt: usize) -> Vec<u8> {
    let mut x = vec![0f32; q8_k::BLOCK_SIZE];
    for (i, v) in x.iter_mut().enumerate() {
        let phase = ((i + salt * 17) as i32).rem_euclid(64);
        let tri = if phase <= 32 { phase - 16 } else { 48 - phase };
        *v = tri as f32 * amp;
    }
    let mut out = vec![0u8; q8_k::BYTES_PER_BLOCK];
    q8_k::quantize_block(&x, &mut out);
    out
}

// =========================================================================
// I/O helpers
// =========================================================================

fn find_vectors_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR is always set under cargo run");
    let dir = PathBuf::from(manifest).join("tests/vectors");
    std::fs::create_dir_all(&dir).expect("create tests/vectors");
    dir
}

fn write_f32_le(path: &Path, values: &[f32]) {
    let mut f = File::create(path).expect("open vector file");
    for v in values {
        f.write_all(&v.to_le_bytes()).expect("write f32");
    }
}
