# ds4-rs Agent Guide

## Project Overview

`ds4-rs` is a Rust port of [antirez/ds4](https://github.com/antirez/ds4), a focused, single-model local inference engine for DeepSeek V4 Flash. The goal is to load GGUF model weights, run a CPU reference forward pass, and generate tokens greedily. Future work includes an OpenAI/Anthropic-compatible HTTP server, on-disk KV cache, and speculative decoding.

* **Target platform:** Linux only. GPU compute backends (Metal, Vulkan, CUDA, etc.) are deferred.
* **License:** MIT
* **Language:** English for all comments, docs, and commit messages.

## Technology Stack

* **Language:** Rust, Edition 2024.
* **Toolchain:** A specific nightly is pinned in `rust-toolchain` (`nightly-2026-05-12`). CI and local development must use this exact toolchain.
* **Build system:** Cargo workspace + [`cargo-make`](https://github.com/sagiegurari/cargo-make) (see `Makefile.toml`).
* **Test runner:** [`cargo-nextest`](https://nexte.st/) is the preferred test runner in CI and locally.
* **Coverage:** [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) + `nextest` for HTML/LCOV reports.
* **Task runner wrapper:** The `./dev` bash script installs `cargo-binstall` and `cargo-make` if missing, then delegates to `cargo make`.

## Workspace Layout

```
dev                     # Task runner wrapper
Cargo.toml              # Workspace root
rust-toolchain          # Pinned nightly channel
Makefile.toml           # cargo-make task definitions
 crates/
 ├── ds4-core/          # Core inference library
 │   ├── src/
 │   │   ├── lib.rs          # Re-exports all public modules
 │   │   ├── engine.rs       # Engine::open(model_path) — loads GGUF + weight map
 │   │   ├── session.rs      # Session::new, prefill, eval_token, argmax
 │   │   ├── tokenizer.rs    # BPE encode/decode from GGUF vocab
 │   │   ├── config.rs       # ModelConfig extracted from GGUF metadata
 │   │   ├── gguf.rs         # GGUF v3 parser, mmap-based zero-copy
 │   │   ├── tensor.rs       # Minimal borrowed `Tensor<'a>` and `OwnedTensor` views
 │   │   ├── model/
 │   │   │   ├── mod.rs      # WeightMap, layer views
 │   │   │   ├── weights.rs  # Typed GGUF weight accessors (q8_0, f16, q2_k, ...)
 │   │   │   ├── layer.rs    # Per-layer borrowed weight structs
 │   │   │   ├── forward.rs  # End-to-end forward pass orchestration
 │   │   │   └── kv_cache.rs # MLA latent KV cache
 │   │   ├── ops/
 │   │   │   ├── mod.rs
 │   │   │   ├── matmul.rs   # Per-dtype dot kernels + matmul_row / matmul_batch
 │   │   │   ├── norm.rs     # RMSNorm
 │   │   │   ├── rope.rs     # Partial RoPE + YaRN
 │   │   │   ├── softmax.rs  # Softmax + sqrt(softplus) router gating
 │   │   │   ├── swiglu.rs   # SwiGLU clamped activation
 │   │   │   └── hc.rs       # Sinkhorn split / HC mix / HC expand
 │   │   └── quant/
 │   │       ├── mod.rs
 │   │       ├── q8_0.rs     # Q8_0 dequant
 │   │       ├── q8_k.rs     # f32 → Q8_K activation pre-quant
 │   │       ├── q2_k.rs     # Q2_K dequant + dot
 │   │       ├── q4_k.rs     # Q4_K dequant + dot
 │   │       ├── iq2_xxs.rs  # IQ2_XXS dequant + dot
 │   │       ├── iq4_xs.rs   # IQ4_XS dequant + dot
 │   │       ├── iq4_nl.rs   # IQ4_NL dequant + dot
 │   │       └── iq4_codebook.rs
 │   ├── Cargo.toml
 │   └── tests/
 │       ├── manifest.rs         # SHA-256 manifest check for binary test vectors
 │       └── vectors/
 │           ├── manifest.toml
 │           └── *.bin             # Frozen cross-reference vectors
 └── ds4-cli/             # CLI binary (`ds4`)
     ├── src/main.rs      # clap args, one-shot generation loop
     └── Cargo.toml
rfcs/
 ├── 0001-port-overview.md
 └── 0002-forward-pass.md
scripts/
 ├── patches/
 └── regen_vectors.sh    # Manual vector regeneration harness
```

## Build System & Common Commands

The project uses `cargo-make` via `Makefile.toml`. Use `./dev <task>` or `cargo make <task>`.

| Task | Command | What it does |
|------|---------|--------------|
| List tasks | `./dev` | Shows all available cargo-make steps |
| Build | `cargo build --workspace` | Standard debug build |
| Check all | `cargo make check` | Runs fmt, dep-sort, clippy, machete, test, typos in sequence |
| Test | `cargo make test` | `cargo nextest run --workspace --all-features --all-targets` |
| Coverage | `cargo make test-cov` | `cargo llvm-cov nextest ... --html` |
| Format check | `cargo make check-fmt` | `cargo fmt --all` (writes) |
| Clippy | `cargo make check-clippy` | `cargo clippy --workspace --all-features --all-targets -- -D warnings` |
| Typos | `cargo make check-typos` | `typos` (installs `typos-cli` if missing) |
| Unused deps | `cargo make check-machete` | `cargo machete` |
| Dep sort | `cargo make check-dep-sort` | `cargo sort -w -c` then auto-fix |
| Hakari | `cargo make check-hakari` | `cargo hakari verify` |

Standard cargo commands also work:

```bash
cargo build --workspace
cargo test --workspace          # plain cargo test, but nextest is preferred
cargo run --bin ds4 -- -p "hello"
```

## Code Style & Conventions

* **Formatter:** `rustfmt` with `rustfmt.toml`.
  * `edition = "2024"`, `style_edition = "2024"`
  * `imports_granularity = "Crate"`
  * `group_imports = "StdExternalCrate"`
  * `comment_width = 120`, `wrap_comments = true`
  * `normalize_comments = true`, `format_code_in_doc_comments = true`
* **Naming:** Stick to the `antirez/ds4` C naming conventions (e.g., `iq2_xxs`, `q2_k`, `ffn_gate_inp`) so cross-repo grepping works.
* **Quant dtype enums:** Use `#[allow(non_camel_case_types)]` for weight dtype names (e.g., `IQ2_XXS`, `Q4_K`).
* **Module docs:** Use `//!` at the top of each module file. Item docs use `///`.
* **Error handling:** `anyhow::Result` at API boundaries; `thiserror` for structured errors where needed.
* **Logging:** Use `tracing` macros (`tracing::info!`, `tracing::debug!`). The CLI initializes `tracing_subscriber::fmt` with an `EnvFilter` defaulting to `INFO`.
* **Safety:** `unsafe` is allowed for memory reinterpretation (mmap, raw pointer casting) but is kept minimal and well-audited. No `unsafe` in hot numerical kernels unless required for SIMD (not yet present).
* **Dependencies:** Prefer workspace dependencies declared in the root `Cargo.toml`. Crate-level `Cargo.toml` references them with `{ workspace = true }`.

## Testing Strategy

Three layers of tests are used:

1. **In-module unit tests:** Each op and dtype kernel has small hand-computed tests at the bottom of its source file. Run with `cargo test` or `cargo make test`.
2. **Cross-reference binary vectors:** Committed `.bin` files under `crates/ds4-core/tests/vectors/` represent known-good outputs. `tests/manifest.rs` checks their SHA-256 against `tests/vectors/manifest.toml` on every CI run.
   * Regeneration is **manual** via `scripts/regen_vectors.sh <op>`.
   * After regeneration, update `manifest.toml` with the new SHAs and run `cargo test -p ds4-core --test manifest`.
3. **End-to-end smoke:** The CLI `ds4 -p "..."` exercises tokenizer → prefill → eval → argmax → decode. A real-model integration test lives at `crates/ds4-core/tests/forward_smoke.rs`, gated behind `DS4_TEST_MODEL` (path to a real DS4 GGUF). Run it with `DS4_TEST_MODEL=/path/to/ds4flash.gguf cargo test -p ds4-core --test forward_smoke -- --ignored`.

### CI Safety Checks

* **AddressSanitizer & LeakSanitizer:** Run on `ubuntu-latest` with `+nightly` and `-Z sanitizer=address/leak`.
* **Miri:** Run `cargo +nightly miri test` for undefined behavior detection.
* **Codecov:** Uploads LCOV from `cargo llvm-cov nextest` on every PR/push.

## CI / CD & Security

GitHub Actions workflows (all use pinned action hashes and `step-security/harden-runner`):

* **`test.yml`** — Required test matrix (ubuntu-latest + macos-latest + windows-latest) using `cargo nextest --locked`. Coverage job uploads to Codecov.
* **`check.yml`** — `cargo fmt --check`, `cargo clippy --workspace --all-features --all-targets -- -D warnings`, and `typos` spell check.
* **`safety.yml`** — AddressSanitizer, LeakSanitizer, and Miri.
* **`dependency-review.yml`** — Scans PR dependency manifests for known vulnerabilities.
* **`scorecards.yml`** — OpenSSF Scorecard supply-chain security analysis.
* **`scheduled.yml`** — Rolling `cargo +nightly nextest --locked` run on a cron schedule.

Additional security measures:
* `CODECOV_TOKEN` secret required for coverage uploads.
* `.github/dependabot.yml` keeps GitHub Actions up to date.

## Documentation & Planning Artifacts

| File | Purpose |
|------|---------|
| `PLAN.md` | High-level implementation plan (phases 1–4) and dependency table. |
| `todo.md` | Current backlog and next moves. Updated alongside code changes. |
| `rfcs/0001-port-overview.md` | Architecture RFC: motivation, design decisions, phases, dependencies. |
| `rfcs/0002-forward-pass.md` | Detailed forward-pass design: module breakdown, op specs, testing strategy, commit roadmap. |
| `scripts/regen_vectors.sh` | Manual harness for regenerating cross-reference binary vectors. |

## Current Status & Known Limitations (as of latest commit)

* **Phase 1 forward pass is numerically complete.** The workspace scaffold, GGUF loader, tokenizer, tensor views, all Phase 1 quant kernels (Q8_0, Q8_K, Q2_K, IQ2_XXS, Q4_K, IQ4_XS, IQ4_NL), RMSNorm, partial RoPE + YaRN, softmax, SwiGLU, HC helpers, typed weight accessors, MLA latent KV cache, real attention (the cached 512-dim row plays K and V across all 64 query heads — no separate per-head up-projection, matching antirez/ds4), routed MoE assembly (hash routing for layers 0–2, biased top-k for layers 3+), learned output HC reduction, and CLI wiring have landed. The `DS4_TEST_MODEL` smoke run now reaches the full forward graph end-to-end against a real DS4 GGUF; full numerical validation still needs a healthy GGUF (see Testing Strategy).
* **GGUF metadata keys.** Config reads `deepseek4.*` keys (no longer placeholder `llama.*`), including `deepseek4.attention.q_lora_rank` for the Q-LoRA rank.
* **Tokenizer.** JoyAI BPE (`tokenizer.ggml.pre = "joyai-llm"`): GPT-2 byte encoding, JoyAI pre-tokenizer cascade, BPE with single-byte fallback. No LLaMA-style `<0xNN>` byte-fallback tokens.
* **GGUF dim ordering.** The loader maps 2-D weight tensor dims `(ne0, ne1)` to `(in_features, out_features)` (matching `ggml_new_tensor_2d`). The previous flipped axes are gone; all `WeightView` constructors share the `dims_in_out` helper.
* **Topk NaN fallback.** `topk_indices_desc` falls back to lowest unselected index when all values are NaN/-inf instead of panicking, matching the C reference's behaviour on model files with NaN F16 weights.
* **Session lifecycle and REPL.** `Session::rewind`, `Session::invalidate`, `Session::pos`, `Session::tokens`, and `Session::ctx_size` ship in `crates/ds4-core/src/session.rs`. The CLI exposes an interactive REPL (default when no `-p` is supplied) with `/reset` (`/clear`), `/rewind <n>`, `/ctx`, `/help` (`/?`), and `/exit` (`/quit`). Reasoning-mode toggles (`/think`, `/nothink`) and file-read (`/read`) are not yet implemented.
* **Effective context:** Sliding-window attention only (`sliding_window = 128`). Long-range context via raw KV ring + compressor/indexer, on-disk KVC, and prefix matching are Phase 2.
* **Performance:** CPU reference, single-threaded, seconds-per-token on commodity hardware. SIMD/threading/GPU are deferred.
* **Binary name:** `ds4` (from `ds4-cli/src/main.rs`).
* **Default model path:** `./ds4flash.gguf` (override with `--model`).

## Quick Start for Agents

```bash
# 1. Ensure the pinned nightly is installed (rustup will auto-install it)
rustup show

# 2. Run the full check suite
cargo make check

# 3. Build and run the CLI
cargo build --release --bin ds4
./target/release/ds4 -p "hello world" -n 16

# 4. Regenerate vectors after an op change (manual, not in CI)
scripts/regen_vectors.sh q8_0
# Then update crates/ds4-core/tests/vectors/manifest.toml
```
