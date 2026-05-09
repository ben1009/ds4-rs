# ds4-rs Implementation Plan

## Overview

Port [antirez/ds4](https://github.com/antirez/ds4) from C/Objective-C/Metal to Rust. Single-model inference engine for DeepSeek V4 Flash on Apple Metal.

## Phase 1: Core Engine + CLI

**Goal:** Load model weights, run forward pass, generate tokens greedily via `ds4 -p "hello"`.

### Step 1 — Workspace scaffold + MSL shaders

- `Cargo.toml` workspace root with dependencies
- Create `crates/ds4-metal/`, `crates/ds4-core/`, `crates/ds4-cli/` with Cargo.toml stubs
- Copy all 19 `.metal` shader files from ds4 into `metal/`

### Step 2 — `ds4-metal` crate

| File | Responsibility |
|------|---------------|
| `context.rs` | `MetalContext::new()` — default device, command queue, compile shaders into one library, pre-compile all required pipelines at init (read-only `HashMap<String, Pipeline>` after init, no locking in hot path) |
| `buffer.rs` | `MetalBuffer` — `alloc()`, `from_slice()`, `view()` (zero-copy sub-range), `from_mmap()` (no-copy wrap of mmap'd weights), `contents_mut()`, `read()`, `write()` |
| `encoder.rs` | `ComputeEncoder` — `set_buffer()`, `set_bytes()`, `dispatch()`, `dispatch_2d()`, `linear_split()`, `set_params!` macro |
| `tensor.rs` | `GpuTensor { buffer, shape, dtype }`, `DType` enum (F32, F16, I32, Q8_0, Q2_K, Q4_K, IQ2_XXS) |
| `shaders.rs` | `include_str!` all 19 `.metal` files, `combined_shader_source()` |

### Step 3 — `ds4-core`: GGUF parser + config + model

| File | Responsibility |
|------|---------------|
| `gguf.rs` | GGUF v3 parser — magic, version, metadata `HashMap<String, Value>`, tensor info `HashMap<String, TensorInfo>`, data offset. Memory-map file for zero-copy. |
| `config.rs` | `ModelConfig` — extract from GGUF metadata: `n_layer`, `n_embd`, `n_head`, `n_kv_head`, `head_dim`, `n_expert`, `n_expert_used`, `n_ff`, `n_hc`, `vocab_size`, `rope_theta` |
| `model.rs` | `WeightMap` — mmap + tensor info lookup. `tensor(name) -> &[u8]`, `tensor_f32(name) -> &[f32]` |

### Step 4 — `ds4-core`: tokenizer

| File | Responsibility |
|------|---------------|
| `tokenizer.rs` | BPE tokenizer from GGUF vocab (`tokenizer.ggml.tokens`, `.scores`, `.merges`). `encode(text) -> Vec<u32>`, `decode(token_id) -> &str` |

### Step 5 — `ds4-core`: engine + session

| File | Responsibility |
|------|---------------|
| `engine.rs` | `Engine::open(model_path)` — load GGUF, init MetalContext, build WeightMap |
| `session.rs` | `Session::new(engine, ctx_size)` allocates GPU buffers. `eval_token(token) -> logits`, `prefill(tokens) -> logits`, `argmax(logits) -> token_id` |

Forward pass per layer (43 layers):
```
HC split → norm → QKV proj → RoPE → KV store → attention → HC expand
→ norm → router → shared expert + routed MoE → HC mix
```
Output head: HC reduction → norm → LM head → logits.

### Step 6 — `ds4-cli`: one-shot generation

| File | Responsibility |
|------|---------------|
| `main.rs` | clap args (`--model`, `-p`, `-n`, `--ctx`), load engine, tokenize, prefill, generate loop, print |

## Phase 2: Session + KV Cache

- Multi-turn session lifecycle (create, eval, rewind, invalidate)
- Raw KV ring buffer (sliding window in GPU memory)
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
| `objc2` | 0.6.3 | ObjC runtime |
| `objc2-metal` | 0.3.2 | Metal GPU bindings |
| `objc2-foundation` | 0.3.2 | Foundation types |
| `anyhow` | 1 | Error handling |
| `thiserror` | 2 | Error derive |
| `tracing` | 0.1 | Logging |
| `clap` | 4 (derive) | CLI args |
| `memmap2` | 0.9 | File mmap |
| `zerocopy` | 0.8 (derive) | Safe transmute |
| `parking_lot` | 0.12 | Fast RwLock |

## Architecture Notes

- **Metal bindings**: `objc2-metal` (NOT the deprecated `metal-rs`). This is what candle uses.
- **Shader loading**: `include_str!` at compile time, compile all into one Metal library, pre-compile all pipelines during `Engine::open()` init. Pipeline map is read-only after init — no locking overhead during inference.
- **Buffer binding**: `set_params!` macro + `EncoderParam` trait (following candle pattern).
- **mmap weights**: GGUF file memory-mapped, Metal buffers wrap mmap regions with `newBufferWithBytesNoCopy` (zero-copy). **Alignment fallback**: `newBufferWithBytesNoCopy` requires page alignment (4096 bytes) but GGUF tensors are typically 32-byte aligned. Strategy: attempt no-copy first; if alignment fails, fall back to `newBufferWithBytes_length_options` (which copies). Log a warning on fallback so users can detect copy overhead.
- **Thread safety**: `Engine` is `Arc`-shared (read-only after init). `Session` is single-owner (Metal worker only).
