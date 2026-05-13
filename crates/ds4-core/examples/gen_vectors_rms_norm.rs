//! Generate the RMSNorm reference vector committed under `tests/vectors/`.
//!
//! Run via `scripts/regen_vectors.sh rms_norm` (preferred) or:
//!
//! ```text
//! cargo run --quiet -p ds4-core --example gen_vectors_rms_norm
//! ```
//!
//! RMSNorm uses a fixed left-to-right reduction order in our implementation,
//! so a Rust-generated vector is reproducible across platforms for the same
//! inputs. Later PRs whose ops involve ordering-sensitive reductions
//! (softmax, attention, MoE sums) will dump from antirez/ds4 instead via
//! the same `scripts/regen_vectors.sh` dispatcher.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use ds4_core::ops::norm::{rms_norm, rms_norm_no_weight};

const N: usize = 4096;
const EPS: f32 = 1e-6;

fn main() {
    let vectors_dir = find_vectors_dir();

    // Deterministic inputs. Use integer-arithmetic "random" so the harness
    // produces bit-identical outputs on every run; f32 constants like sin()
    // are also deterministic but we prefer to keep generators visibly
    // reproducible.
    let x: Vec<f32> = (0..N).map(deterministic_input).collect();
    let w: Vec<f32> = (0..N).map(deterministic_weight).collect();

    // 1) Weighted RMSNorm — the `attn_norm` / `ffn_norm` / `output_norm` workhorse.
    let mut y = vec![0f32; N];
    rms_norm(&x, &w, EPS, &mut y);
    write_f32_le(&vectors_dir.join("rms_norm.bin"), &y);

    // 2) No-weight RMSNorm — HC pre/post + the output-head pre-matmul path.
    let mut y_nw = vec![0f32; N];
    rms_norm_no_weight(&x, EPS, &mut y_nw);
    write_f32_le(&vectors_dir.join("rms_norm_no_weight.bin"), &y_nw);

    eprintln!(
        "Wrote RMSNorm reference vectors to {}",
        vectors_dir.display()
    );
}

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

/// Inputs that span positive + negative, small + large, with a few near-zero
/// samples to exercise the denominator.
fn deterministic_input(i: usize) -> f32 {
    // Period-64 sawtooth mapped to [-2.0, +2.0].
    let phase = (i as i32 * 7 + 3).rem_euclid(64);
    let saw = (phase as f32 - 32.0) * (1.0 / 16.0);
    // Slow triangular envelope in [0.5, 1.5] with period 256.
    // Integer arithmetic keeps the whole generator cross-platform bit-exact —
    // libm's f32 transcendentals are not guaranteed identical across
    // Linux / Windows / macOS.
    let tri_phase = (i as i32).rem_euclid(256);
    let tri = if tri_phase <= 128 {
        tri_phase as f32
    } else {
        (256 - tri_phase) as f32
    };
    let env = 0.5 + tri * (1.0 / 128.0);
    saw * env
}

/// Weights with mixed sign and a few unit-scale entries.
fn deterministic_weight(i: usize) -> f32 {
    let phase = (i as i32 * 13 - 5).rem_euclid(32);
    (phase as f32 - 16.0) * (1.0 / 8.0)
}
