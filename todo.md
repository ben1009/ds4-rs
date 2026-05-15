# ds4-rs TODO

This file tracks the next implementation moves after checking the current
repository against `PLAN.md` and `rfcs/0002-forward-pass.md`.

## Current Baseline

- `cargo test --workspace` passes.
- `rfcs/0002-forward-pass.md` is partially implemented, but its status line is stale.
- `Session::prefill` and `Session::eval_token` now call the forward path, but full numerical correctness is still blocked by the Phase 1 stubs below.
- `model::forward` exists, but it still contains known Phase 1 stubs:
  - routed expert MoE is disabled until I-quant / K-quant matmul support lands;
  - attention scores against the cached MLA latent directly instead of doing per-head MLA K/V up-projection;
  - output HC reduction sums streams instead of using learned output combine weights.

## Next Move

1. [x] Wire `Session` into the existing decode forward path.
   - Make `prefill(tokens)` evaluate prompt tokens in order and leave logits for the last prompt token.
   - Make `eval_token(token)` append one token, run `forward_decode`, and return real logits.
   - Enforce `ctx_size` overflow before mutating session state.
   - Preserve empty-prefill behavior or document and test any behavior change.
   - Add tests that fail if these methods silently return all-zero logits without invoking forward.
   - Validate with `cargo test --workspace`.
   - Done: session now calls `forward_decode`, rejects context overflow before mutation, and rolls token state back when forward fails.

2. [x] Update RFC / plan status after session wiring.
   - Change RFC 0002 status from "Draft -- design review only, no code yet" to an implementation-progress status.
   - Mark already-landed roadmap items clearly.
   - Keep the known stubs explicit so they do not look like completed forward-pass behavior.
   - Validate with a documentation grep for stale "no code yet" / obsolete PR numbering.
   - Done: RFC 0002, `PLAN.md`, and stale code comments now describe the partial implementation state.

## Forward Pass Correctness

3. [x] Implement `quant/iq2_xxs` and `quant/q2_k`.
   - Port the block layout and dot-product behavior from the ggml / ds4 reference.
   - Add `WeightView` variants and matmul dispatch for Q8_K activation rows against IQ2_XXS and Q2_K weights.
   - Add unit tests for block decoding and dot products.
   - Add or regenerate committed vectors and update `tests/vectors/manifest.toml`.
   - Validate with `cargo test --workspace`.
   - Done in `dedf3f8`: kernels, mixed-dtype dot products, `WeightView` variants, and matmul dispatch landed with in-module unit tests. Committed reference vectors / `manifest.toml` entries for IQ2_XXS and Q2_K are still outstanding -- rolled into item 9.

4. [x] Implement `quant/iq4_k` and `quant/q4_k`.
   - Mirror the IQ2_XXS / Q2_K implementation pattern for the Q4 expert variants.
   - Extend matmul dispatch and typed weight handling.
   - Add reference-vector coverage.
   - Validate with `cargo test --workspace`.
   - Done: Q4_K, IQ4_XS, and IQ4_NL kernels (dequant + Q8_K-activation dot product, plus a
     direct f32 dot path for IQ4_NL's 32-element blocks) are wired through `WeightView`,
     `quant_weight` auto-dispatch, and `matmul_row` / `matmul_batch`. The previous
     `WeightView::Unsupported` placeholders for routed Q4 experts are gone. In-module
     unit tests cover decoding and dot products; reference vectors carry over to item 9.

5. [ ] Replace the routed MoE stub.
   - Use `ffn_gate_inp` router logits.
   - Implement hash routing for layers 0-2 via `ffn_gate_tid2eid`.
   - Implement biased top-k routing for layers 3+.
   - Apply `sqrt(softplus(logit))` gating and routed expert accumulation.
   - Keep shared expert contribution intact.
   - Validate with focused router / expert tests and `cargo test --workspace`.

6. [ ] Replace the attention MLA stub with real per-head K/V up-projection.
   - Cache only `kv_latent` and `k_pe`, as currently designed.
   - During attention, up-project latent rows into per-head K/V using the correct attention weights.
   - Keep sliding-window + sink behavior from RFC 0002.
   - Add a small deterministic attention test before end-to-end validation.
   - Validate with `cargo test --workspace`.

7. [ ] Implement learned output HC reduction.
   - Identify and load the GGUF tensor for output HC combine weights.
   - Replace the current stream-sum fallback in `output_head`.
   - Add tests for the reduction behavior.
   - Validate with `cargo test --workspace`.

## End-to-End Enablement

8. [ ] Add a real forward smoke path for the CLI.
   - Ensure `ds4 -p "..." -n ...` exercises tokenizer, session prefill, eval, argmax, and decode.
   - Prefer an ignored full-model test behind an environment variable if no tiny GGUF fixture exists.
   - Document the expected limitations: CPU reference speed, sliding-window-only context, no compressor/indexer.

9. [ ] Regenerate and expand reference vectors.
   - Keep `scripts/regen_vectors.sh` as the source of truth for vector reproduction.
   - Add vectors alongside each op or block implementation, not as one large final batch.
   - Keep `crates/ds4-core/tests/vectors/manifest.toml` in sync.
   - Backfill committed vectors for the routed-expert quants (IQ2_XXS, Q2_K, Q4_K,
     IQ4_XS, IQ4_NL) carried over from items 3 and 4: block dequant + Q8_K-activation
     (or f32-activation, for IQ4_NL) dot product reference vectors.

10. [ ] Revisit Phase 2 only after Phase 1 logits are credible.
    - Session lifecycle operations: rewind, invalidate, and multi-turn flow.
    - Raw KV ring behavior and long-context compressor / indexer.
    - On-disk KVC compatibility.
    - CLI interactive REPL.

## Hygiene

11. [ ] Keep docs aligned with implementation status.
    - Update `PLAN.md` and RFC 0002 when a TODO item changes architectural assumptions.
    - Avoid leaving stale comments that say a PR is future work after code lands.

12. [ ] Keep every completed item independently testable.
    - Each item should pass `cargo test --workspace`.
    - Prefer focused regression tests near the changed module.
    - If a behavior requires a real model file, gate it with an explicit env var and document it.
