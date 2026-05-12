# Design: Forward Pass (Phase 1 Step 2)

Status: **Draft — design review only, no code yet.**
Replaces the schematic paragraph in `PLAN.md §Step 4`.

## 1. Scope

Implement enough of the DeepSeek V4 Flash forward pass to make the existing
`Session::prefill` / `Session::eval_token` stubs produce *real* logits and
unblock greedy generation via `ds4 -p "..."`.

**In scope (this phase):**

- CPU reference forward pass, single-threaded, `f32` activations end-to-end.
- All 43 layers including hash-routed layers 0–2 and top-k-routed layers 3+.
- MLA attention with partial RoPE, per-head Q RMSNorm, and sink logit.
- MoE with 1 shared + 6/256 routed experts, SwiGLU-clamped, sqrt(softplus)
  router gating, biased-top-k selection.
- HC (hyper-connection) split/mix using 20 Sinkhorn iterations.
- Dequantisation for the weight dtypes actually used: F16, Q8_0, Q2_K, Q4_K,
  IQ2_XXS, IQ4_K. (One dequant path per dtype, no shortcuts.)
- In-memory MLA latent KV cache (512-dim latent + 64-dim decoupled RoPE key
  per token, per layer). Up-projection happens inside the attention kernel.
  No on-disk cache (that's Phase 2), no FP8 round-trip on KV writes yet
  (see §6).
- Golden-vector tests against the C reference for selected ops and a short
  greedy prompt.

**Out of scope (explicitly deferred):**

- Ratio-4 / ratio-128 compressor, indexer, mixed-KV sliding-window attention
  — the full ds4 attention is gated behind these. Phase 1 uses *raw* attention
  over the full context (no sliding window, no compressor). This produces
  correct logits below `sliding_window = 128` tokens and an approximation
  beyond; acceptable for a greedy smoke test, not for production quality.
- FP8 E4M3 KV storage round-trip. KV is stored as plain `f32`.
- Speculative decoding / MTP draft model (Phase 4).
- SIMD, threading, or any GPU backend.
- Disk KV cache, server, REPL.

The deferrals above are load-bearing: they let Phase 1 ship in a few hundred
lines of numerical Rust instead of a few thousand, and they're the pieces
most likely to drift from the C reference. Phase 2 revisits attention end-
to-end with the full compressor + indexer path.

## 2. Module breakdown

All new code under `crates/ds4-core/src/`.

| Module | Responsibility |
|--------|---------------|
| `tensor.rs` | Minimal `Tensor<'a>` — borrowed `&[f32]` view with shape + strides. Owned `OwnedTensor` for scratch buffers. No autograd, no broadcasting framework. Just shape-carrying slices. |
| `quant/mod.rs` | Dequant entry point: `dequant(dtype, bytes, out: &mut [f32])`. |
| `quant/q8_0.rs` | Q8_0 block (34 B/32 elem) → f32. |
| `quant/q2_k.rs` / `q4_k.rs` / `iq2_xxs.rs` / `iq4_k.rs` | K-quant and I-quant paths. Each is a direct port of the ggml reference with a test vector per block format. |
| `quant/q8_k.rs` | f32 → Q8_K (activation pre-quant for IQ2_XXS/Q2_K matmuls). |
| `ops/matmul.rs` | `matmul_f32(a: [M,K], b: [N,K], out: [M,N])`. Plus `matmul_mixed(a_q8k, b_iq2xxs, ...)` for routed experts. Naive triple loop, no blocking yet. |
| `ops/norm.rs` | `rms_norm(x, weight, eps, out)` and `rms_norm_no_weight` used by HC pre/post and output head. |
| `ops/rope.rs` | Partial RoPE on the last 64 dims of each 512-dim head, with YaRN scale. Two frequency bases: 10000 (attention) and 160000 (compressor — unused in Phase 1). |
| `ops/softmax.rs` | Standard softmax + the `sqrt(softplus(logit))` router activation. |
| `ops/swiglu.rs` | `silu(clamp(gate, ±10)) * clamp(up, ±10)`. |
| `ops/hc.rs` | Sinkhorn split (20 iters), HC weighted sum, HC expand. |
| `model/weights.rs` | Typed accessors: `WeightMap::q8_0("attn_q_a.weight")` → `Q8_0View<'a>`; one method per dtype. Wraps the existing `tensor_bytes(name)` API. |
| `model/layer.rs` | `Layer { attn, moe, hc_attn, hc_ffn }` structs — just borrowed views, no owned data. Built once in `Engine::open`. |
| `model/forward.rs` | The orchestration: `forward(engine, state, tokens) -> logits`. Linear, no trait abstractions. |
| `model/kv_cache.rs` | `KvCache { latent: Vec<f32>, k_pe: Vec<f32>, pos: usize }` — MLA latent storage. Shapes `[n_layer, ctx, 512]` and `[n_layer, ctx, 64]`, appended in place. Up-projection runs inside the attention kernel. |

Naming sticks to the antirez/ds4 conventions so grepping cross-repo works.

## 3. Key decisions

### 3.1 Activations are `f32` everywhere

The C reference is also f32 end-to-end. Mixed precision is a perf
optimisation for Phase 3+, and adding it now would make golden-vector tests
harder to reproduce. Cost is ~2× memory pressure on intermediates, which is
fine for a CPU reference.

### 3.2 Dequantise lazily, per-op, no pre-materialised f32 weights

Materialising a Q8_0 tensor to f32 up front for a 4096→1024 projection is
33 MB per call site. Instead, `matmul` is specialised per weight dtype and
walks blocks directly — same pattern as `ggml_vec_dot_*`. This keeps memory
use close to the on-disk footprint.

For routed-expert matmuls the *activation* is quantised to Q8_K first (one
row at a time, stack-scratch), matching `ds4_quantize_row_q8_K` in the C
reference.

### 3.3 No tensor framework, no `ndarray`

The forward graph is ~50 ops and entirely static. Adding `ndarray` or a
custom `Tensor` abstraction now costs complexity and buys us nothing —
everything operates on `&[f32]` with hand-computed strides. If we later want
SIMD or threading, it goes *inside* each op, not in a shared type.

### 3.4 Raw attention only in Phase 1

Full ds4 attention routes every layer through a ratio-4 or ratio-128
compressor plus a top-k indexer that produces a boolean mask. That's a
separate mini-attention module plus two extra projections per layer and is
the single largest deviation from "standard transformer". For this phase we:

1. Compute Q, K, V normally per token.
2. Apply partial RoPE on the last 64 dims of each head.
3. Add the per-head sink logit into the softmax denominator.
4. Dot-product over the full context (no sliding window, no compressor).

This is correct for `pos < sliding_window = 128` and increasingly
approximate past that. Default `ctx_size = 2048` (see §3.5) keeps prompts
well inside that budget for a smoke-test run. Phase 2 lands the real
attention.

### 3.5 KV cache layout — MLA latent caching

MLA's whole point is that K/V don't need to live in the cache as
`[n_head × head_dim]` per token. The compressor projects each token down to
a 512-dim latent (`attn_kv_a_norm` output) plus a 64-dim decoupled RoPE key
(`k_pe`), and attention up-projects on the fly per query-head. So we cache
the *latent*, not the materialised K/V.

Per-token cache footprint (f32):

| Tensor | Dim | Bytes/token |
|--------|-----|-------------|
| `kv_latent` (pre-expand) | 512 | 2 KiB |
| `k_pe` (decoupled RoPE key) | 64 | 256 B |
| **Total** | 576 | **2.25 KiB** |

For `ctx_size = 4096` and 43 layers: `4096 × 43 × 2.25 KiB ≈ 396 MiB`.
vs ~23 GiB for materialised K/V per-head — ~60× smaller.

Layout: two flat `Vec<f32>`s per session, indexed `[layer][pos][dim]`.
Up-projection runs inside the attention kernel using the same `attn_kv` /
`attn_k_nope` / `attn_v` weights as prefill — there is no second copy.

Default `ctx_size = 2048` (~200 MiB KV) for the Phase 1 smoke test;
configurable up to 4096. We surface a clear error past that until Phase 2
lands the compressor + indexer.

### 3.6 Non-tied output head

`output.weight` (Q8_0, 4096→129280) is *separate* from `token_embd.weight`
(F16, 129280→4096). Two distinct tensors, two distinct matmuls.

### 3.7 HC state

Residual stream is 4 parallel `f32` vectors of width `n_embd = 4096`. Sub-
layer output is reduced into the stream via a Sinkhorn-normalised 4×4
combine matrix produced per layer from the `hc_*_fn / _scale / _base`
weights. 20 Sinkhorn iterations, unrolled, element-wise normalise rows then
columns.

### 3.8 Matmul: `matrixmultiply` on the hot path

A naive triple-loop f32×f32 matmul against a 4096→129280 output head is
~1 GFLOPs/token at best and takes ~10+ seconds per token on commodity CPUs —
too slow even for a "correctness-only" smoke test. We pull in
[`matrixmultiply`](https://crates.io/crates/matrixmultiply) (pure Rust,
SIMD-microkernel, no BLAS dep) for the f32×f32 hot path:

- f32 activations × f32 (dequant result) — handled by `matrixmultiply::sgemm`.
- Quantised weight matmuls still walk blocks directly (no pre-dequant), but
  accumulate into f32 with the same microkernel-shaped inner loop so the
  hot path stays vectorised.

No `rayon` / threading in Phase 1 — parallelism is a Phase 3 concern and
would fragment the golden-vector tolerances. Single-threaded
`matrixmultiply` is the sweet spot: tens of lines of plumbing, ~5–10×
speedup, no behavioural change.

Add `matrixmultiply = "0.3"` to `workspace.dependencies` in PR #1.

## 4. Testing strategy

Three layers of tests. Each new op lands with a test in its own PR — none
of this "big bang" at the end.

### 4.1 Unit tests

One per op, per dtype. Known input → known output from a small hand-computed
case. The Q-format dequants can lean on the block header math; we don't need
a reference .gguf for these.

### 4.2 Cross-reference vectors against the C implementation

**Vectors land alongside the op PR that produces them, not in a final
"big bang" PR.** A single bit-exact vector per op guards that PR's
correctness forever, and debugging a regression means staring at one op's
diff, not the whole forward pass.

Per-PR vector ownership (maps to the commit roadmap in §5):

| PR | Vector |
|----|--------|
| 1 (Q8_0 matmul) | Dequant output for one Q8_0 block + one 32×K×M matmul result. |
| 2 (RMSNorm) | `rms_norm(x, w, 1e-6)` for a 4096-wide slice. |
| 3 (RoPE) | Rotated Q for a single head at positions 0, 1, 127. |
| 4 (Q8_K pre-quant) | `ds4_quantize_row_q8_K` output for a 256-wide row. |
| 5 (IQ2_XXS / Q2_K) | One full dot-product of a Q8_K row against an IQ2_XXS and a Q2_K block. |
| 6 (Q4_K / IQ4_K) | Same shape as (5) for the Q4 variants. |
| 7 (softmax / SwiGLU / HC Sinkhorn) | 20-iter Sinkhorn on a 4×4 matrix; `sqrt(softplus(x))` for 8 values. |
| 9 (attention block) | Full layer-0 attention output on a 4-token prefill. |
| 10 (MoE block) | Router probs + expert outputs for layer 3 on a single token. |
| 11 (end-to-end forward) | Final logits for the 4th token of the fixed 4-token prompt. |

Dumping these from antirez/ds4 needs a small patch that disables FP8 KV
rounding (so our tolerances stay at `1e-4`) and writes each intermediate
to a `.bin`. The patch + script live in `scripts/regen_vectors.sh`, run
manually, not in CI.

### 4.3 End-to-end smoke test

`ds4 -p "hello" -n 16` against a tiny test GGUF (if one exists upstream),
or against the full model behind `#[ignore]` + an env var. Greedy output
must be deterministic.

## 5. Commit roadmap

Each bullet = one PR. Ordered for incremental landability.

1. **`quant/q8_0` + `ops/matmul` (f32×Q8_0)** — smallest useful slice, tests
   on known block.
2. **`tensor.rs` + `ops/rms_norm`** — trivial but unblocks the norm sites.
3. **`ops/rope` (partial + YaRN)** — isolated, tested with a hand-computed
   rotation.
4. **`quant/q8_k` + Q8_K activation pre-quant path** — needed before the
   I-quant matmuls.
5. **`quant/iq2_xxs` + `quant/q2_k`** — one PR, pair up because expert
   gate/up and down share a block stride.
6. **`quant/q4_k` + `quant/iq4_k`** — Q4 variant of (5).
7. **`ops/softmax` + `ops/swiglu` + `ops/hc` (Sinkhorn)** — glue ops.
8. **`model/weights.rs` typed accessors** — tightens the existing
   `tensor_bytes` API.
9. **`model/kv_cache` + attention assembly (single layer, no compressor)**
   — now we have a working attention block.
10. **MoE assembly (hash-layer + top-k variants) + full layer** — now a full
    layer runs.
11. **`model/forward.rs` + CLI wiring** — `ds4 -p` produces real logits;
    lands the final end-to-end reference vector.
12. **End-to-end smoke test + PLAN.md update.**

Roughly 12 PRs. The early ones are small (200–400 lines); the MoE and
attention ones are larger (600–800). Each one builds and passes
`cargo test --workspace` on its own.

## 6. Known gaps / risks

- **Attention approximation.** Raw attention over full context diverges from
  the C reference once `pos ≥ 128`. Documented in §3.4. Phase 2 fixes it.
- **FP8 KV round-trip.** Not implemented. This changes KV values slightly
  even on first write. The cross-reference vectors in §4.2 must be
  generated with FP8 *disabled* in the C build, or the tolerances raised
  past `1e-4`. Decision: patch the C build to skip FP8 for the reference
  run. That patch lives in `scripts/regen_vectors.sh`, not in CI.
- **Hash-layer routing (layers 0–2).** Requires loading `ffn_gate_tid2eid`
  (I32, 6 × n_vocab = 3 MB). Confirm the tensor exists in the GGUF we
  target before starting PR #10.
- **Memory at high ctx_size.** MLA latent caching brings the default
  2048-token KV to ~200 MiB and 4096 tokens to ~400 MiB (§3.5) —
  manageable on a dev box. Anything past 4096 errors out until Phase 2
  lands compressor + indexer.
- **Performance.** Even with `matrixmultiply` the Phase 1 path is
  correctness-focused, not production-fast. Expect seconds-per-token on
  commodity CPUs; Phase 3 revisits hot loops.

## 7. What this PR contains

This PR is design only — it adds this `rfcs/0002-forward-pass.md` file and
nothing else. Approval means "the plan is sound, start implementing".
Actual code lands in the 12 PRs above, each reviewed independently.
