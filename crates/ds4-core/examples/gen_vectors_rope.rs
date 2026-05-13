//! Generate the RoPE reference vectors committed under `tests/vectors/`.
//!
//! Run via `scripts/regen_vectors.sh rope` (preferred) or:
//!
//! ```text
//! cargo run --quiet -p ds4-core --example gen_vectors_rope
//! ```
//!
//! RoPE uses per-element rotations derived from deterministic integer-indexed
//! frequencies. cos/sin are unavoidable here but they are the implementation
//! itself — any consistent Rust version will produce the same bytes
//! cross-platform because `f32::cos` / `f32::sin` on the same bit-exact
//! input round identically on IEEE-754 hardware the way libm's rustc-shipped
//! impl rounds them.
//!
//! Vectors dumped:
//! * `rope_plain_pos127.bin`   — plain RoPE (no YaRN) at position 127
//! * `rope_yarn_pos127.bin`    — DS4 YaRN-scaled RoPE at position 127
//! * `rope_yarn_pos0.bin`      — identity check (pos = 0 must equal input)
//! * `rope_yarn_inverse.bin`   — inverse at position 127

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use ds4_core::ops::rope::{RopeFreqs, RopeParams, YarnParams, apply_rope, apply_rope_inverse};

const HEAD_DIM: usize = 512;
const N_ROT: usize = 64;

fn main() {
    let vectors_dir = find_vectors_dir();
    let head = synthetic_head(HEAD_DIM);

    // Plain RoPE (no YaRN).
    let plain = RopeFreqs::new(&RopeParams {
        n_rot: N_ROT,
        base: 10_000.0,
        yarn: None,
    });
    let mut buf = head.clone();
    apply_rope(&mut buf, 127, &plain);
    write_f32_le(&vectors_dir.join("rope_plain_pos127.bin"), &buf);

    // DS4 Flash YaRN params — see rfcs/0002-forward-pass.md §3.4.
    let ds4 = RopeFreqs::new(&RopeParams {
        n_rot: N_ROT,
        base: 10_000.0,
        yarn: Some(YarnParams {
            scale_factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx: 65_536.0,
            attn_factor: None,
        }),
    });

    // Position-0 identity check (no-op).
    let mut buf = head.clone();
    apply_rope(&mut buf, 0, &ds4);
    write_f32_le(&vectors_dir.join("rope_yarn_pos0.bin"), &buf);

    // Position 127 with YaRN.
    let mut buf = head.clone();
    apply_rope(&mut buf, 127, &ds4);
    write_f32_le(&vectors_dir.join("rope_yarn_pos127.bin"), &buf);

    // Inverse rotation — must round-trip to `head` when followed by forward.
    let mut buf = head.clone();
    apply_rope_inverse(&mut buf, 127, &ds4);
    write_f32_le(&vectors_dir.join("rope_yarn_inverse.bin"), &buf);

    eprintln!("Wrote RoPE reference vectors to {}", vectors_dir.display());
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

/// Deterministic integer-only head. First `HEAD_DIM - N_ROT = 448` dims are
/// the untouched nope prefix; last 64 dims are the rotation tail.
fn synthetic_head(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            // Period-64 triangular wave in [-2.0, 2.0] — integer arithmetic
            // only, so the generator is cross-platform reproducible.
            let phase = (i as i32 * 5 + 11).rem_euclid(64);
            let tri = if phase <= 32 {
                phase as f32
            } else {
                (64 - phase) as f32
            };
            (tri - 16.0) * (1.0 / 8.0)
        })
        .collect()
}
