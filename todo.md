# ds4-rs TODO

This file tracks the next implementation moves after checking the current
repository against `PLAN.md` and `rfcs/0002-forward-pass.md`.

## Current Baseline

- `cargo test --workspace` passes (408 ds4-core unit tests + 31 ds4-cli
  unit tests + manifest check; one ignored doctest and one ignored
  end-to-end smoke test gated behind `DS4_TEST_MODEL`).
- Phase 1 forward pass is numerically complete: real MLA attention (the
  cached 512-dim row plays K and V across all 64 query heads), routed
  MoE assembly, and learned output HC reduction are all in place.
- `Session::prefill` and `Session::eval_token` reach the full forward
  graph; the CLI `ds4 -p "..." -n ...` exercises the same code path.
- Session lifecycle ops (`rewind`, `invalidate`) and an interactive REPL
  (`/reset`, `/rewind <n>`, `/ctx`, `/help`, `/exit`) have landed
  (PRs #28, #29). The REPL is the default when no `-p` is supplied.
- Reference vectors for Q8_0, Q8_K, RMSNorm, RoPE, and the routed-expert
  quants (IQ2_XXS, Q2_K, Q4_K, IQ4_XS, IQ4_NL) are committed under
  `crates/ds4-core/tests/vectors/` and SHA-checked by
  `tests/manifest.rs`.
- `forward_smoke` now runs end-to-end against a real DS4 GGUF without
  panicking: prefill plus several decode steps complete the full
  forward graph. PRs #33 (deepseek4.* metadata keys), #34 (JoyAI BPE
  tokenizer rewrite), #35 (`q_lora_rank` from metadata), #36 (correct
  GGUF `(ne0, ne1) → (in_features, out_features)` mapping), #37
  (per-group slicing of `attn_output_a` in `grouped_out_decode`), and
  #38 (topk fallback for all-NaN router probs) closed out the
  immediate blockers. Caveat: the q2-imatrix DS4-Flash GGUF used for
  this run has corrupted F16 weights (~2.8% NaN bit-patterns), so the
  smoke test's "decode logits diverge across steps" assertion still
  fails on that file. The antirez/ds4 C reference produces the same
  gibberish on the same file, confirming this is a model-file issue,
  not a Phase 1 implementation issue. Numerical validation just needs
  a healthy DS4 GGUF.
- Phase 2 architecture work (compressor / indexer, on-disk KVC) is the
  new gate for the remaining backlog and can begin in parallel with
  end-to-end numerical validation (item 10).

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

5. [x] Replace the routed MoE stub.
   - Use `ffn_gate_inp` router logits.
   - Implement hash routing for layers 0-2 via `ffn_gate_tid2eid`.
   - Implement biased top-k routing for layers 3+.
   - Apply `sqrt(softplus(logit))` gating and routed expert accumulation.
   - Keep shared expert contribution intact.
   - Validate with focused router / expert tests and `cargo test --workspace`.
   - Done: `routed_moe_decode` runs the F16 router, picks 6 experts via
     `tid2eid` on hash layers or biased top-k (with optional
     `exp_probs_b.bias`) on top-k layers, applies `sqrt(softplus)` gating with
     the `1/16384` sum floor + 1.5× rescale, runs each routed expert through
     SwiGLU with the ±10 clamp, and sums into the FFN output. New
     `expert_subview` slices per-expert bytes from 3D routed-expert tensors.
     Tests cover top-k, expert byte slicing, and the gating floor.

6. [x] Replace the attention MLA stub with real per-head K/V up-projection.
   - Cache only `kv_latent` and `k_pe`, as currently designed.
   - During attention, up-project latent rows into per-head K/V using the correct attention weights.
   - Keep sliding-window + sink behavior from RFC 0002.
   - Add a small deterministic attention test before end-to-end validation.
   - Validate with `cargo test --workspace`.
   - Done: matched antirez/ds4 ds4.c — there is no separate per-head K/V
     up-projection weight in DS4 MLA. The cached row is logically the
     full DS4_N_HEAD_DIM = 512 vector (`kv_latent` || `k_pe`) and is used
     unchanged as both K (for scoring) and V (for the weighted sum).
     Fixes: `attn_kv_a_norm` is now sized to the full 512 dims (matching
     `rms_norm_weight(kv, raw, attn_kv_a_norm, DS4_N_HEAD_DIM)`); the
     full row is normed before splitting into latent/k_pe; attention
     accumulates the value sum across both halves into the head output.
     Refactored the math into `attention_rows_inner` and added four
     deterministic tests (single-row identity, sink-dominated collapse,
     uniform-row averaging, per-head independence).

7. [x] Implement learned output HC reduction.
   - Identify and load the GGUF tensor for output HC combine weights.
   - Replace the current stream-sum fallback in `output_head`.
   - Add tests for the reduction behavior.
   - Validate with `cargo test --workspace`.
   - Done: matches `output_hc_head_one` in antirez/ds4 ds4.c.
     `output_head` now does `rms_norm_no_weight` over the full HC state,
     `output_hc_fn` (F16) matvec to a per-stream pre vector, then
     `weights[i] = sigmoid_stable(pre[i] * output_hc_scale + output_hc_base[i]) + 1e-6`,
     and finally `hc_weighted_sum` to collapse to a plain n_embd vector
     before the existing RMSNorm + Q8_0 vocab projection. The pure
     weight math is factored into `output_hc_weights` and covered by
     three deterministic tests (formula, saturation/floor, zero-input).

## End-to-End Enablement

8. [x] Add a real forward smoke path for the CLI.
   - Ensure `ds4 -p "..." -n ...` exercises tokenizer, session prefill, eval, argmax, and decode.
   - Prefer an ignored full-model test behind an environment variable if no tiny GGUF fixture exists.
   - Document the expected limitations: CPU reference speed, sliding-window-only context, no compressor/indexer.
   - Done: `crates/ds4-core/tests/forward_smoke.rs` exercises tokenizer →
     prefill → eval_token → argmax end-to-end and is gated behind
     `DS4_TEST_MODEL` (path to a real DS4 GGUF). The test focuses on
     forward-path *health* (logits are finite, in-range, and decode-step
     output diverges from prefill) rather than text quality, so it is
     robust to the Phase 1 limitations called out in its module doc:
     CPU reference speed, sliding-window-only context, no compressor /
     indexer, no FP8 KV round-trip. Run it with
     `DS4_TEST_MODEL=/path/to/ds4flash.gguf cargo test -p ds4-core
     --test forward_smoke -- --ignored`. The CLI binary
     (`ds4 -p "..." -n ...`) reaches the same code path.

9. [x] Regenerate and expand reference vectors.
   - Keep `scripts/regen_vectors.sh` as the source of truth for vector reproduction.
   - Add vectors alongside each op or block implementation, not as one large final batch.
   - Keep `crates/ds4-core/tests/vectors/manifest.toml` in sync.
   - Backfill committed vectors for the routed-expert quants (IQ2_XXS, Q2_K, Q4_K,
     IQ4_XS, IQ4_NL) carried over from items 3 and 4: block dequant + Q8_K-activation
     (or f32-activation, for IQ4_NL) dot product reference vectors.
   - Done: `crates/ds4-core/examples/gen_vectors_routed_quants.rs` produces
     two `.bin` files per format (block dequant + per-block dot) for
     IQ2_XXS, Q2_K, Q4_K, IQ4_XS, and IQ4_NL. The manifest carries SHA
     entries so any byte drift surfaces as a CI failure (`cargo test
     -p ds4-core --test manifest`). The `scripts/regen_vectors.sh
     routed_quants` op invokes the example and prints the SHAs for a
     manual review when regeneration is needed. Existing op-level vectors
     for Q8_0, Q8_K, RMSNorm, and RoPE are unchanged.

10. Phase 2 work can proceed in parallel with end-to-end numerical
    validation. The `DS4_TEST_MODEL` run now reaches the full forward
    graph end-to-end against a real DS4 GGUF, so the previous hard
    gate is lifted; the unchecked sub-items just need a healthy GGUF
    for numerical sign-off (the q2-imatrix file currently in hand has
    corrupted F16 weights, see Current Baseline).
    - [x] Session lifecycle operations: rewind, invalidate, and multi-turn flow.
      - Done in #28: `Session::invalidate`, `Session::rewind`,
        `Session::pos`, `Session::tokens`, and `Session::ctx_size` cover
        in-memory rewind / invalidate; the REPL exercises multi-turn
        prefix preservation across turns.
    - [ ] Raw KV ring behavior and long-context compressor / indexer.
    - [ ] On-disk KVC compatibility.
    - [x] CLI interactive REPL.
      - Done in #29: `repl()` in `crates/ds4-cli/src/main.rs` ships
        `/reset` (`/clear`), `/rewind <n>`, `/ctx`, `/help` (`/?`), and
        `/exit` (`/quit`). The REPL is the default when no `-p` is
        supplied. Reasoning-mode toggles (`/think`, `/nothink`) and
        file-read (`/read`) are still future work.

## Hygiene (ongoing)

11. Keep docs aligned with implementation status.
    - Update `PLAN.md` and RFC 0002 when a TODO item changes architectural assumptions.
    - Avoid leaving stale comments that say a PR is future work after code lands.

12. Keep every completed item independently testable.
    - Each item should pass `cargo test --workspace`.
    - Prefer focused regression tests near the changed module.
    - If a behavior requires a real model file, gate it with an explicit env var and document it.
