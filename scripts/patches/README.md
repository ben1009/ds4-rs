# C-side patches for antirez/ds4

Cross-reference vectors in `crates/ds4-core/tests/vectors/` that need to
match the upstream C reference (attention outputs, softmax sums, MoE
routing, etc.) are produced by cloning antirez/ds4 at a pinned SHA,
applying a patch from this directory, building, and running a dump
binary.

Patches land here as each reference-vector-requiring PR needs them:

- `0001-disable-fp8-kv.patch` — *(future PR)* disables the FP8 E4M3 KV
  round-trip in `ds4.c` so KV values are stored as raw f16, matching
  the Rust implementation's f32 cache. Needed before any attention-
  output vector can cross-ref at 1e-4 tolerance.

RFC 0002 §4.2 specifies this directory structure so the regen harness
stays auditable.

## Conventions

- Each patch targets a specific upstream SHA, recorded in its header.
- Patches are applied by `scripts/regen_vectors.sh`, not by CI.
- One patch per behavioural change, not per file.
- Keep each patch surgical — a few lines that toggle a compile-time
  path or replace a single call site, not a rewrite.
