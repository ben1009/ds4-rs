# ds4-rs Implementation Plan

## Overview

Port [antirez/ds4](https://github.com/antirez/ds4) from C/Objective-C/Metal to Rust. Single-model inference engine for DeepSeek V4 Flash. Linux-only target; Metal GPU backend deferred to future work.

## Phase 1: Core Engine + CLI

**Goal:** Load model weights, run forward pass, generate tokens greedily via `ds4 -p "hello"`.

**Status:** In progress. GGUF/config/tokenizer plumbing, core op helpers,
partial decode forward orchestration, MLA latent KV cache, routed MoE
assembly, and session/CLI wiring have landed. Phase 1 logits are not
numerically complete yet: real MLA K/V up-projection, learned output HC
reduction, and end-to-end smoke coverage remain tracked in `todo.md`.

### Step 1 — Workspace scaffold

Status: **done**.

- `Cargo.toml` workspace root with dependencies
- Create `crates/ds4-core/`, `crates/ds4-cli/` with Cargo.toml stubs

### Step 2 — `ds4-core`: GGUF parser + config + model

Status: **partially done**. The loader and typed weight accessors exist; routed
expert quant typed accessors are still blocked on the remaining quant kernels.

| File | Responsibility |
|------|---------------|
| `gguf.rs` | GGUF v3 parser — magic, version, metadata `HashMap<String, Value>`, tensor info `HashMap<String, TensorInfo>`, data offset. Memory-map file for zero-copy. |
| `config.rs` | `ModelConfig` — extract from GGUF metadata: `n_layer`, `n_embd`, `n_head`, `n_kv_head`, `head_dim`, `n_expert`, `n_expert_used`, `n_ff`, `n_hc`, `vocab_size`, `rope_theta` |
| `model.rs` | `WeightMap` — mmap + tensor info lookup. `tensor(name) -> &[u8]`, `tensor_f32(name) -> &[f32]` |

### Step 3 — `ds4-core`: tokenizer

Status: **done for the current Phase 1 path**.

| File | Responsibility |
|------|---------------|
| `tokenizer.rs` | BPE tokenizer from GGUF vocab (`tokenizer.ggml.tokens`, `.scores`, `.merges`). `encode(text) -> Vec<u32>`, `decode(token_id) -> &str` |

### Step 4 — `ds4-core`: engine + session

Status: **partial**. `Session::prefill` and `Session::eval_token` now call the
decode forward path and preserve session/KV state on errors. The forward graph
still contains the numerical stubs listed in `todo.md`.

| File | Responsibility |
|------|---------------|
| `engine.rs` | `Engine::open(model_path)` — load GGUF, build WeightMap |
| `session.rs` | `Session::new(engine, ctx_size)` allocates KV state. `eval_token(token) -> logits`, `prefill(tokens) -> logits`, `argmax(logits) -> token_id` |

Forward pass per layer (43 layers):
```
HC split → norm → QKV proj → RoPE → KV store → attention → HC expand
→ norm → router → shared expert + routed MoE → HC mix
```
Output head: HC reduction → norm → LM head → logits.

### Step 5 — `ds4-cli`: one-shot generation

Status: **wired, pending credible logits**. The CLI reaches session prefill and
generation, but real model smoke validation waits for the remaining Phase 1
forward-pass work.

| File | Responsibility |
|------|---------------|
| `main.rs` | clap args (`--model`, `-p`, `-n`, `--ctx`), load engine, tokenize, prefill, generate loop, print |

## Phase 2: Session + KV Cache

- Multi-turn session lifecycle (create, eval, rewind, invalidate)
- Raw KV ring buffer (sliding window in memory)
- Compressed KV store + ratio-4 indexer
- On-disk KVC cache (48-byte header + session payload, binary-compatible with ds4)
- Prefix matching for cache reuse
- CLI interactive REPL (`/think`, `/nothink`, `/ctx`, `/read`)

## Phase 3: Server

- axum + tokio HTTP server
- `POST /v1/chat/completions` (OpenAI-compatible, SSE streaming)
- `POST /v1/messages` (Anthropic-compatible, SSE streaming)
- `POST /v1/completions` (raw text)
- `GET /v1/models`
- DSML tool schema rendering + generated output parsing
- Streaming state machine: DSML → OpenAI tool call translation
- KV disk cache: cold/continued/evict/shutdown saves

## Phase 4: Speculative Decoding + Polish

- MTP draft model loading + draft-then-verify loop
- Test vector validation against official DeepSeek API outputs
- Full CLI arg parity (`--kv-disk-dir`, `--mtp`, `--kv-disk-space-mb`, etc.)
- Documentation

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `anyhow` | 1 | Error handling |
| `thiserror` | 2 | Error derive |
| `tracing` | 0.1 | Logging |
| `clap` | 4 (derive) | CLI args |
| `memmap2` | 0.9 | File mmap |
| `zerocopy` | 0.8 (derive) | Safe transmute |
| `parking_lot` | 0.12 | Fast RwLock |

## Architecture Notes

- **Linux-only**: No macOS/Metal support. GPU compute backend (Vulkan, CUDA, etc.) deferred to future work.
- **mmap weights**: GGUF file memory-mapped for zero-copy weight access.
- **Thread safety**: `Engine` is `Arc`-shared (read-only after init). `Session` is single-owner (inference worker only).
