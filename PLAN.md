# ds4-rs Implementation Plan

## Overview

Port [antirez/ds4](https://github.com/antirez/ds4) from C/Objective-C/Metal to Rust. Single-model inference engine for DeepSeek V4 Flash. Linux-only target; Metal GPU backend deferred to future work.

## Phase 1: Core Engine + CLI

**Goal:** Load model weights, run forward pass, generate tokens greedily via `ds4 -p "hello"`.

**Status:** Forward pass numerically complete; the `DS4_TEST_MODEL`
smoke run now completes end-to-end against a real DS4 GGUF (prefill
plus decode steps) without panicking. Numerical validation still
needs a healthy GGUF — the q2-imatrix DS4-Flash file currently in
hand has corrupted F16 weights, and the antirez/ds4 C reference
produces the same gibberish on it. Landed: GGUF / config / tokenizer
plumbing (`deepseek4.*` metadata keys, JoyAI BPE tokenizer with the
GPT-2 byte map, `q_lora_rank` read from metadata), the GGUF loader
fix that maps `(ne0, ne1)` to `(in_features, out_features)` for all
weight views, all Phase 1 quant kernels (Q8_0, Q8_K, Q2_K, IQ2_XXS,
Q4_K, IQ4_XS, IQ4_NL), core op helpers (RMSNorm, partial RoPE + YaRN,
softmax, SwiGLU, HC Sinkhorn), the MLA latent KV cache, the full
attention path (single 512-dim cached row used as both K and V — no
separate per-head up-projection), the routed MoE assembly (hash
routing for layers 0–2 / biased top-k for layers 3+), the learned
output HC reduction, per-group slicing of `attn_output_a` in
`grouped_out_decode`, and a topk fallback for all-NaN router probs.
`Session::prefill` and `Session::eval_token` reach the full forward
graph; the CLI `ds4 -p "..." -n ...` exercises tokenizer → prefill →
eval → argmax → decode end-to-end. `crates/ds4-core/tests/forward_smoke.rs`
(gated behind `DS4_TEST_MODEL`) drives the same path against a real
model.

### Step 1 — Workspace scaffold

Status: **done**.

- `Cargo.toml` workspace root with dependencies
- Create `crates/ds4-core/`, `crates/ds4-cli/` with Cargo.toml stubs

### Step 2 — `ds4-core`: GGUF parser + config + model

Status: **done.** The loader, typed weight accessors, and all Phase 1
quant kernels (Q8_0, Q8_K, Q2_K, IQ2_XXS, Q4_K, IQ4_XS, IQ4_NL) are in
place; routed-expert tensors auto-dispatch via `WeightMap::quant_weight`.

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

Status: **done for Phase 1.** `Session::prefill` and `Session::eval_token`
run the full decode forward path (real MLA attention + learned output HC
reduction), preserve session / KV state on errors, and reject context
overflow before mutating state.

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

Status: **wired.** The CLI exercises tokenizer → session prefill →
eval_token → argmax → decode end-to-end. Real-model smoke validation runs
via the `DS4_TEST_MODEL`-gated integration test in
`crates/ds4-core/tests/forward_smoke.rs`.

| File | Responsibility |
|------|---------------|
| `main.rs` | clap args (`--model`, `-p`, `-n`, `--ctx`), load engine, tokenize, prefill, generate loop, print |

## Phase 2: Session + KV Cache

- Multi-turn session lifecycle (create, eval, rewind, invalidate) — **done**
  (#28). In-memory `Session::rewind` / `Session::invalidate` ship; the
  REPL exercises multi-turn prefix preservation across turns.
- Raw KV ring buffer (sliding window in memory)
- Compressed KV store + ratio-4 indexer
- On-disk KVC cache (48-byte header + session payload, binary-compatible with ds4)
- Prefix matching for cache reuse
- CLI interactive REPL — **done** (#29). Shipped commands: `/reset`
  (`/clear`), `/rewind <n>`, `/ctx`, `/help` (`/?`), `/exit` (`/quit`).
  Reasoning-mode toggles (`/think`, `/nothink`) and file-read (`/read`)
  are still pending.

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
