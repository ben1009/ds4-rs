//! Generate the Q8_K reference vectors committed under `tests/vectors/`.
//!
//! Run via `scripts/regen_vectors.sh q8_k` (preferred) or:
//!
//! ```text
//! cargo run --quiet -p ds4-core --example gen_vectors_q8_k
//! ```
//!
//! Q8_K quantisation is a pure block transform — no f32 reductions with
//! order-dependent rounding — so a Rust-generated vector is reproducible
//! across platforms for the same input.
//!
//! Vectors dumped:
//! * `q8_k_block.bin`      — a single 292-byte block quantised from a deterministic 256-element
//!   input that exercises both signs, the max-abs saturation path, and a few near-zero samples.
//! * `q8_k_multi_block.bin` — three consecutive blocks with different scales, to guard the chunked
//!   `quantize` entry point.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use ds4_core::quant::q8_k;

fn main() {
    let vectors_dir = find_vectors_dir();

    // Single-block vector.
    let x = single_block_input();
    let mut out = vec![0u8; q8_k::BYTES_PER_BLOCK];
    q8_k::quantize_block(&x, &mut out);
    write_bytes(&vectors_dir.join("q8_k_block.bin"), &out);

    // Three-block vector — each block has a different dynamic range.
    let n = 3;
    let mut x = vec![0f32; n * q8_k::BLOCK_SIZE];
    for b in 0..n {
        for i in 0..q8_k::BLOCK_SIZE {
            x[b * q8_k::BLOCK_SIZE + i] = per_block_input(b, i);
        }
    }
    let mut out = vec![0u8; n * q8_k::BYTES_PER_BLOCK];
    q8_k::quantize(&x, &mut out);
    write_bytes(&vectors_dir.join("q8_k_multi_block.bin"), &out);

    eprintln!("Wrote Q8_K reference vectors to {}", vectors_dir.display());
}

fn find_vectors_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR is always set under cargo run");
    let dir = PathBuf::from(manifest).join("tests/vectors");
    std::fs::create_dir_all(&dir).expect("create tests/vectors");
    dir
}

fn write_bytes(path: &Path, bytes: &[u8]) {
    let mut f = File::create(path).expect("open vector file");
    f.write_all(bytes).expect("write bytes");
}

/// Deterministic 256-element input with:
/// * a single +max at index 42 (tests positive-max saturation)
/// * a single -max at index 170 (confirms sign handling)
/// * a triangular wave across the rest
/// * a band of exact zeros around index 128 (confirms zero pass-through)
fn single_block_input() -> Vec<f32> {
    let mut x = vec![0f32; q8_k::BLOCK_SIZE];
    for (i, v) in x.iter_mut().enumerate() {
        let phase = (i as i32 * 3 + 7).rem_euclid(128);
        let tri = if phase <= 64 { phase - 32 } else { 96 - phase };
        *v = tri as f32 * 0.03125; // [-1, 1] band
    }
    x[42] = 5.0; // positive max
    x[170] = -4.0; // negative
    for v in x[120..136].iter_mut() {
        *v = 0.0;
    }
    x
}

/// Deterministic per-block generator — block `b` uses a band of magnitude
/// `(b + 1)` so scales differ monotonically.
fn per_block_input(b: usize, i: usize) -> f32 {
    let amp = (b as f32 + 1.0) * 0.5;
    let phase = (i as i32 + b as i32 * 11).rem_euclid(64);
    let tri = if phase <= 32 { phase - 16 } else { 48 - phase };
    tri as f32 * (amp / 16.0)
}
