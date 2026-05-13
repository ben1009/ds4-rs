//! Generate the Q8_0 reference vectors committed under `tests/vectors/`.
//!
//! Run via `scripts/regen_vectors.sh q8_0` (preferred) or directly:
//!
//! ```text
//! cargo run --quiet -p ds4-core --example gen_vectors_q8_0
//! ```
//!
//! Q8_0 dequant is pure block arithmetic (f16 scale × i8 quants) with a
//! fixed byte layout defined by ggml. That makes a Rust-generated reference
//! vector valid: any implementation that agrees with the formula will
//! produce the same bytes, regardless of floating-point accumulation order.
//!
//! Later PRs that introduce non-associative computations (attention,
//! softmax, MoE sums) will have their subroutines dump from antirez/ds4 C
//! builds instead, via the same `scripts/regen_vectors.sh` dispatcher.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use ds4_core::{
    ops::matmul::{WeightView, matmul_row},
    quant::q8_0,
};

const BLOCK_SIZE: usize = q8_0::BLOCK_SIZE;
const BYTES_PER_BLOCK: usize = q8_0::BYTES_PER_BLOCK;

fn main() {
    let vectors_dir = find_vectors_dir();
    write_dequant_vector(&vectors_dir);
    write_matmul_vector(&vectors_dir);
    eprintln!("Wrote Q8_0 reference vectors to {}", vectors_dir.display());
}

fn find_vectors_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR points at crates/ds4-core regardless of where cargo
    // is invoked. Keeps the script portable from repo root or elsewhere.
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR is always set under cargo run");
    let dir = PathBuf::from(manifest).join("tests/vectors");
    std::fs::create_dir_all(&dir).expect("create tests/vectors");
    dir
}

/// Dequantised bytes for 8 hand-constructed Q8_0 blocks (256 elements).
fn write_dequant_vector(dir: &Path) {
    let blocks = synthetic_blocks(8);
    let mut bytes = Vec::with_capacity(blocks.len() * BYTES_PER_BLOCK);
    for b in &blocks {
        bytes.extend_from_slice(b);
    }
    let mut out = vec![0f32; blocks.len() * BLOCK_SIZE];
    q8_0::dequant(&bytes, &mut out);

    let path = dir.join("q8_0_dequant.bin");
    write_f32_le(&path, &out);
}

/// Q8_0 weight [N=4, K=64] × f32 act [K=64] → f32 out [N=4].
fn write_matmul_vector(dir: &Path) {
    let n = 4;
    let k = 64;
    assert_eq!(k % BLOCK_SIZE, 0);

    // Build deterministic Q8_0 weight rows.
    let blocks_per_row = k / BLOCK_SIZE;
    let mut bytes = Vec::with_capacity(n * blocks_per_row * BYTES_PER_BLOCK);
    for row in 0..n {
        for block_idx in 0..blocks_per_row {
            bytes.extend_from_slice(&synthetic_block_rowcol(row, block_idx));
        }
    }

    // Deterministic activations: triangular wave in [-1, 1].
    let act: Vec<f32> = (0..k).map(triangular_wave).collect();

    let w = WeightView::Q8_0 {
        bytes: &bytes,
        out_features: n,
        in_features: k,
    };
    let mut out = vec![0f32; n];
    matmul_row(w, &act, &mut out);

    let path = dir.join("q8_0_matmul.bin");
    write_f32_le(&path, &out);
}

fn write_f32_le(path: &Path, values: &[f32]) {
    let mut f = File::create(path).expect("open vector file");
    for v in values {
        f.write_all(&v.to_le_bytes()).expect("write f32");
    }
}

/// Deterministic Q8_0 blocks indexed 0..n. Scale varies across blocks;
/// quants sweep the full i8 range so we exercise both signs + saturation
/// edges.
fn synthetic_blocks(n: usize) -> Vec<[u8; BYTES_PER_BLOCK]> {
    (0..n)
        .map(|i| {
            // Scales: 1.0, 0.5, 0.25, ... + a negative scale block to catch
            // sign-handling regressions.
            let f16_scales: [u16; 8] = [
                0x3C00, // 1.0
                0x3800, // 0.5
                0x3400, // 0.25
                0x3000, // 0.125
                0xBC00, // -1.0
                0xBA00, // -0.75
                0x3C00, // 1.0
                0x4400, // 4.0
            ];
            let scale = f16_scales[i % f16_scales.len()];
            let mut quants = [0i8; BLOCK_SIZE];
            for (j, q) in quants.iter_mut().enumerate() {
                // Integer arithmetic keeps quants bit-exact across platforms.
                let v = (i as i32 * 7 + j as i32 * 3 - 48).rem_euclid(256) - 128;
                *q = v as i8;
            }
            pack_block(scale, quants)
        })
        .collect()
}

/// Per-row per-block synthetic weight generator for the matmul vector.
fn synthetic_block_rowcol(row: usize, block_idx: usize) -> [u8; BYTES_PER_BLOCK] {
    // Positive and negative scales by (row + block_idx) parity.
    let scale: u16 = match (row + block_idx) % 4 {
        0 => 0x3C00, // 1.0
        1 => 0x3800, // 0.5
        2 => 0xBC00, // -1.0
        _ => 0x4000, // 2.0
    };
    let mut quants = [0i8; BLOCK_SIZE];
    for (j, q) in quants.iter_mut().enumerate() {
        let v = (row as i32 * 11 + block_idx as i32 * 5 + j as i32 * 2).rem_euclid(256) - 128;
        *q = v as i8;
    }
    pack_block(scale, quants)
}

fn pack_block(scale_f16: u16, quants: [i8; BLOCK_SIZE]) -> [u8; BYTES_PER_BLOCK] {
    let mut b = [0u8; BYTES_PER_BLOCK];
    b[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    for (i, q) in quants.iter().enumerate() {
        b[2 + i] = *q as u8;
    }
    b
}

fn triangular_wave(i: usize) -> f32 {
    // Period 16: goes -1 → 1 → -1 over 16 samples. Integer arithmetic so
    // regeneration stays bit-exact.
    let phase = (i % 16) as i32;
    let n = if phase <= 8 { phase - 4 } else { 12 - phase };
    n as f32 / 4.0
}
