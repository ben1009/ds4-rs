# RFC: ds4-rs — Rust Port of antirez/ds4

## Summary

Port [antirez/ds4](https://github.com/antirez/ds4), a focused DeepSeek V4 Flash local inference engine, from C/Objective-C/Metal to Rust. The goal is a single-model inference engine that runs DeepSeek V4 Flash on Apple Metal hardware with on-disk KV cache, OpenAI/Anthropic-compatible server API, and speculative decoding support.

## Motivation

ds4 is a compelling inference engine: it treats KV cache as a first-class disk citizen, supports 2-bit/4-bit asymmetric quantization, and provides a production-ready server with OpenAI and Anthropic-compatible endpoints. However, it is written in C with manual memory management, hand-rolled JSON parsing, and Objective-C Metal bindings.

A Rust port would provide:

- **Memory safety** without sacrificing performance — critical for a long-running inference server
- **Better concurrency story** — Rust's ownership model makes the single-writer/Metal-worker pattern safer to reason about
- **Ecosystem access** — `serde`, `tokio`, `axum`, `wgpu`/`metal-rs` for cleaner server, serialization, and GPU code
- **Easier cross-platform groundwork** — while Metal is the primary target, Rust's `wgpu` could enable Vulkan/DX12 backends in the future
- **Community contribution** — Rust attracts more open-source contributors for systems-level ML tooling

## Non-Goals

- Generic model support (staying single-model focused like the original)
- Linux/Windows Metal support (Metal-only, same as ds4)
- Replacing the Metal shaders (MSL shaders are kept as-is; we bind them from Rust)
- CPU inference path (exists in original for debugging only, low priority)

## Architecture

```
ds4-rs/
├── Cargo.toml               # Workspace root
├── crates/
│   ├── ds4-core/             # Core inference engine, model loading, KV cache
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs     # Engine lifecycle, options, memory estimation
│   │   │   ├── session.rs    # Session state, eval, sampling, argmax
│   │   │   ├── tokenizer.rs  # Tokenization, chat template rendering
│   │   │   ├── kv_cache.rs   # In-memory KV ring buffer + compressed KV
│   │   │   ├── kv_disk.rs    # On-disk KV cache (KVC format read/write)
│   │   │   ├── model.rs      # GGUF weight loading (ds4-specific)
│   │   │   ├── quant.rs      # IQ2_XXS, Q2_K, Q4_K dequantization
│   │   │   └── mtp.rs        # Multi-token prediction speculative decoding
│   │   └── Cargo.toml
│   ├── ds4-metal/            # Metal GPU backend
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── device.rs     # Metal device, command queue, pipeline state
│   │   │   ├── tensor.rs     # Device tensor allocation, views, transfers
│   │   │   ├── graph.rs      # Forward pass graph builder (per-layer)
│   │   │   ├── kernels.rs    # MSL kernel dispatch (matmul, norm, RoPE, attn, MoE, HC)
│   │   │   └── shaders/      # MSL shader source files (copied from ds4)
│   │   └── Cargo.toml
│   ├── ds4-server/           # HTTP server (OpenAI + Anthropic compatible)
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── api/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── chat.rs       # POST /v1/chat/completions
│   │   │   │   ├── completions.rs # POST /v1/completions
│   │   │   │   ├── anthropic.rs   # POST /v1/messages
│   │   │   │   └── models.rs      # GET /v1/models
│   │   │   ├── dsml.rs       # DeepSeek Markup Language rendering + parsing
│   │   │   ├── streaming.rs  # SSE streaming + DSML→OpenAI tool call translation
│   │   │   └── types.rs      # Shared request/response types
│   │   └── Cargo.toml
│   └── ds4-cli/              # CLI (interactive + one-shot)
│       ├── src/
│       │   ├── main.rs
│       │   └── repl.rs       # Interactive REPL with linenoise-style editing
│       └── Cargo.toml
├── metal/                    # MSL shader files (verbatim from ds4)
│   ├── argsort.metal
│   ├── bin.metal
│   ├── ... (all .metal files)
│   └── unary.metal
├── tests/                    # Test vectors (verbatim from ds4)
│   └── test-vectors/
└── scripts/
    └── download_model.sh
```

## Key Design Decisions

### 1. Metal Bindings: `metal-rs` crate

Use the [`metal`](https://crates.io/crates/metal) crate (Mozilla's `gfx-rs` team) for Objective-C Metal bindings. This provides safe Rust wrappers around `MTLDevice`, `MTLCommandQueue`, `MTLBuffer`, `MTLComputePipelineState`, etc.

For MSL shader compilation at runtime, use `metal::Device::new_library_with_source()` — same approach as the C code which compiles shaders at init time.

### 2. Async Server: `axum` + `tokio`

Replace the blocking-thread-per-connection model with:

```
tokio runtime
├── TCP listener (axum)
├── Request parser tasks (spawned per connection)
└── Single Metal worker task (owns the ds4_session, processes jobs from a channel)
```

The Metal worker remains single-threaded (serialized GPU work), receiving jobs via `tokio::sync::mpsc`. Request tasks await responses via `oneshot` channels. This matches ds4's architecture but with async I/O for connection handling.

### 3. KV Cache Binary Format: Bit-for-bit Compatible

The on-disk KVC format (48-byte header + session payload) must be byte-identical to the C implementation to enable cache file interoperability. Use `#[repr(C, packed)]` structs and `byteorder`/`zerocopy` for serialization.

```rust
#[repr(C, packed)]
struct KvcHeader {
    magic: [u8; 3],           // "KVC"
    version: u8,              // 1
    routed_quant_bits: u8,    // 2 or 4
    save_reason: u8,          // 0-4
    reserved: [u8; 2],
    cached_token_count: u32,
    hit_count: u32,
    context_size: u32,
    reserved2: [u8; 4],
    creation_time: u64,
    last_used_time: u64,
    payload_bytes: u64,
}
```

### 4. JSON Parsing: `serde_json`

Replace hand-rolled JSON parsing with `serde` for request parsing. The original's JSON parser is selective (only extracts needed fields), but `serde_json` with `#[serde(deny_unknown_fields)]` and `Value`-based partial parsing can achieve the same effect with less maintenance burden.

For DSML rendering/parsing (which is XML-like, not JSON), implement a dedicated module.

### 5. Tokenizer: Embedded GGUF Vocabulary

Load vocabulary from the GGUF model file at engine init. The tokenizer implementation is deterministic (BPE) and model-specific, so no external dependency needed. Store vocab as `Vec<String>` with a hash map for text→id lookup.

### 6. Thread Safety

```rust
// Engine is Send + Sync (immutable after init)
struct Engine { ... } // weights, config, tokenizer — read-only after load

// Session is NOT Send — owned exclusively by the Metal worker
struct Session { ... } // mutable KV state, GPU buffers — single owner

// The worker pattern:
struct MetalWorker {
    engine: Arc<Engine>,
    session: Session,
    rx: mpsc::Receiver<Job>,
}
```

## Implementation Phases

### Phase 1: Core Engine (`ds4-core` + `ds4-metal`)

**Goal:** Load model weights, run a forward pass, generate tokens greedily.

1. **GGUF loader** — Parse ds4-specific GGUF files, extract tensors and metadata
2. **Metal device init** — Create device, command queue, compile shaders
3. **Tensor management** — Allocate/free Metal buffers, host↔device transfers
4. **Weight mapping** — Map GGUF tensors to Metal buffer offsets (model_map pattern)
5. **Forward pass** — Implement the per-layer graph:
   - Token embedding → HC state
   - Per-layer: norm → QKV → RoPE → KV store → attention → norm → router → MoE → HC mix
   - Final norm → LM head → logits
6. **Sampling** — Greedy argmax + temperature/top-k/top-p/min-p
7. **Tokenizer** — BPE encode/decode, chat template rendering
8. **CLI one-shot** — `ds4 -p "hello"` generates tokens

### Phase 2: Session & KV Cache

**Goal:** Multi-turn conversations with KV cache persistence.

1. **Session lifecycle** — Create, eval, rewind, invalidate
2. **KV ring buffer** — Sliding-window raw KV in GPU memory
3. **Compressed KV** — Compressor + indexer for long-context
4. **Disk cache** — KVC format read/write, prefix matching, eviction
5. **CLI interactive** — REPL with `/think`, `/nothink`, `/ctx`, `/read`

### Phase 3: Server

**Goal:** Drop-in replacement for `ds4-server`.

1. **axum HTTP server** — Listening, routing, error handling
2. **OpenAI endpoint** — `/v1/chat/completions` with streaming
3. **Anthropic endpoint** — `/v1/messages` with streaming
4. **Completions endpoint** — `/v1/completions`
5. **DSML** — Tool schema rendering, generated output parsing
6. **Streaming state machine** — DSML→OpenAI tool call translation
7. **KV disk cache integration** — Cold/continued/evict/shutdown saves

### Phase 4: Speculative Decoding & Polish

**Goal:** Feature parity with ds4.

1. **MTP** — Load MTP draft model, draft-then-verify loop
2. **Test vectors** — Validate against official DeepSeek API outputs
3. **Configuration** — CLI args parity (`--ctx`, `--kv-disk-dir`, `--mtp`, etc.)
4. **Documentation** — Usage guide, architecture docs

## Dependencies

| Crate | Purpose |
|-------|---------|
| `metal` | Apple Metal GPU bindings |
| `objc2` | Objective-C runtime (transitive from metal) |
| `tokio` | Async runtime |
| `axum` | HTTP framework |
| `serde` / `serde_json` | JSON serialization |
| `sha1` | KV cache key hashing |
| `memmap2` | GGUF file memory mapping |
| `clap` | CLI argument parsing |
| `tracing` / `tracing-subscriber` | Logging |
| `anyhow` / `thiserror` | Error handling |
| `zerocopy` | Safe transmute for binary format parsing |

## Risks & Open Questions

1. **Metal shader compatibility** — MSL shaders from ds4 may need minor tweaks for `metal-rs`'s API. The kernel dispatch pattern (compute command encoder with thread groups) should map 1:1.

2. **Performance parity** — C → Rust overhead is negligible for compute-bound GPU workloads, but host-side logic (JSON parsing, streaming state machine) needs benchmarking. The original achieves ~468 t/s prefill on M3 Ultra.

3. **GGUF format stability** — ds4 uses a custom/ds4-specific GGUF layout. We depend on this format being stable or versionable.

4. **Memory layout** — The original uses raw pointer arithmetic for model weight access. Rust's type system adds safety but we need `unsafe` blocks for GPU buffer reinterpretation, kept minimal and well-audited.

5. **MTP speculative decoding** — Marked "experimental" in the original. Implement last, after core inference is validated.

## Success Criteria

- [ ] Loads the same GGUF model files as ds4
- [ ] Generates identical output for greedy decoding (test vector validation)
- [ ] KV cache files are binary-compatible with ds4
- [ ] Server serves OpenAI and Anthropic-compatible endpoints
- [ ] Streaming with tool call translation works correctly
- [ ] Performance within 10% of ds4 on same hardware
- [ ] Runs as a single binary with no runtime dependencies beyond macOS/Metal
