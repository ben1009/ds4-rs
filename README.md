# ds4-rs

[![Test](https://github.com/ben1009/ds4-rs/actions/workflows/test.yml/badge.svg)](https://github.com/ben1009/ds4-rs/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ben1009/ds4-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/ben1009/ds4-rs)

A Rust port of [antirez/ds4](https://github.com/antirez/ds4): a single-model inference engine for **DeepSeek V4 Flash**.

> **Status:** Phase 1 forward pass is numerically complete; the
> `DS4_TEST_MODEL` smoke run now completes end-to-end against a real
> DS4 GGUF (prefill plus decode steps) without panicking. Full
> numerical validation still needs a healthy GGUF — the q2-imatrix
> DS4-Flash file in hand has corrupted F16 weights and produces the
> same gibberish on the antirez/ds4 C reference. Interactive REPL and
> session lifecycle (`rewind`, `invalidate`) are wired up. See
> [`todo.md`](todo.md) for the current backlog.

## Goals

- Load DeepSeek V4 Flash GGUF weights and run greedy token generation on Linux.
- Keep the architecture simple, modular, and easy to hack on.
- Defer GPU backends (Metal, CUDA, Vulkan) to future work.

## Workspace

| Crate | Description |
|-------|-------------|
| [`ds4-core`](crates/ds4-core) | GGUF parser, tokenizer, model config, tensor ops, quant kernels, session/KV cache, and forward pass. |
| [`ds4-cli`](crates/ds4-cli) | Command-line binary `ds4` for one-shot text generation. |

## Quick Start

### Build

```bash
cargo build --release
```

### Run

```bash
# One-shot generation (use -p)
ds4 -p "What is Rust?" -n 128

# Interactive REPL (default when no -p is supplied)
ds4

# With explicit model path and context size
ds4 --model ./ds4flash.gguf -p "Hello" -n 64 --ctx 8192
```

In the REPL, `/help` (or `/?`) lists the slash commands: `/reset`
(`/clear`), `/rewind <n>`, `/ctx`, `/exit` (`/quit`).

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `./ds4flash.gguf` | Path to the GGUF model file |
| `-p, --prompt` | — | Prompt for one-shot generation |
| `-n, --max-tokens` | `256` | Maximum tokens to generate |
| `--ctx` | `2048` | Context size (Phase 1 limit: 4096) |

### Test

```bash
cargo test --workspace
```

## Architecture

- **GGUF v3** — memory-mapped for zero-copy weight access.
- **BPE tokenizer** — loaded from GGUF metadata.
- **Quantization** — mixed-dtype kernels (Q8_K activations, IQ2_XXS / Q2_K / Q4_K / IQ4_XS / IQ4_NL weights) with matmul dispatch.
- **MLA latent KV cache** — caches compressed key/value state.
- **MoE routing** — shared expert + routed experts (top-k gating with hash routing for early layers).

See [`PLAN.md`](PLAN.md) and [`rfcs/`](rfcs/) for design details.

## Requirements

- Linux
- Rust toolchain (see [`rust-toolchain`](rust-toolchain))

## License

MIT — see [`LICENSE`](LICENSE).
