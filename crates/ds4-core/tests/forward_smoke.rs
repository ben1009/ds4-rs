//! End-to-end forward-pass smoke test against a real DS4 GGUF.
//!
//! Phase 1's full forward path now runs (real MLA attention + learned
//! output HC reduction). This test exercises tokenizer → prefill → eval →
//! argmax → decode end-to-end. It is `#[ignore]` because it needs a real
//! GGUF on disk; run it manually with:
//!
//! ```text
//! DS4_TEST_MODEL=/path/to/ds4flash.gguf \
//!     cargo test -p ds4-core --test forward_smoke -- --ignored --nocapture
//! ```
//!
//! Phase 1 limitations the test deliberately tolerates:
//! * CPU reference speed (seconds-per-token; we cap generation at a few tokens so this finishes in
//!   reasonable wall time).
//! * Sliding-window attention only — context older than 128 tokens is masked out (no compressor /
//!   indexer). Long-range coherence is *not* expected.
//! * No FP8 KV round-trip — the cached KV is plain f32.
//!
//! The asserts target *forward-pass health*, not text quality:
//! * Prefill returns logits the right shape, all finite.
//! * eval_token produces logits that are non-trivially different from prefill's logits (catches an
//!   all-zero or stuck forward path).
//! * argmax always picks a valid vocab id.

#![cfg(not(miri))]

use std::path::PathBuf;

use ds4_core::{engine::Engine, session::Session};

fn model_path() -> Option<PathBuf> {
    std::env::var_os("DS4_TEST_MODEL").map(PathBuf::from)
}

#[test]
#[ignore = "needs DS4_TEST_MODEL pointing at a real DS4 GGUF"]
fn smoke_prefill_then_decode_few_tokens() {
    let path = match model_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "DS4_TEST_MODEL not set — skipping. Set it to a real DS4 GGUF \
                 to exercise the full forward path."
            );
            return;
        }
    };
    if !path.exists() {
        panic!(
            "DS4_TEST_MODEL points at {} which does not exist",
            path.display()
        );
    }

    let engine = Engine::open(&path).expect("Engine::open failed");
    let n_vocab = engine.config.n_vocab as usize;

    // Small ctx so the test is fast even on the CPU reference path.
    let ctx_size = 256u32;
    let mut session = Session::new(engine.clone(), ctx_size).expect("Session::new failed");

    // A tiny prompt is enough — we are exercising the wiring, not generation
    // quality. The tokenizer drops the BOS prepend control flag through, but
    // we let it default to true to mirror the CLI.
    let prompt = "Hello";
    let prompt_tokens: Vec<u32> = engine.tokenizer.encode(prompt, true);
    assert!(
        !prompt_tokens.is_empty(),
        "tokenizer returned empty tokens for {prompt:?}"
    );
    assert!(
        prompt_tokens.iter().all(|&t| (t as usize) < n_vocab),
        "tokenizer emitted out-of-range token id",
    );

    let prefill_logits = session.prefill(&prompt_tokens).expect("prefill failed");
    assert_eq!(
        prefill_logits.len(),
        n_vocab,
        "prefill logits length mismatch"
    );
    assert!(
        prefill_logits.iter().all(|v| v.is_finite()),
        "prefill produced non-finite logits — forward path is unhealthy",
    );

    // Snapshot the prefill logits so we can verify decode actually changed
    // state. (An all-zero or stuck forward path would return the same vector.)
    let prefill_snapshot: Vec<f32> = prefill_logits.to_vec();
    let first_tok = Session::argmax(&prefill_snapshot)
        .expect("prefill argmax returned None — all-NaN or all-neg-inf logits?");
    assert!((first_tok as usize) < n_vocab);

    // Decode a few tokens. We don't care what they say — only that the
    // forward path keeps producing finite, in-range logits.
    let mut prev = first_tok;
    let mut diff_seen = false;
    for step in 0..3 {
        let logits = session
            .eval_token(prev)
            .unwrap_or_else(|e| panic!("eval_token step {step} failed: {e}"));
        assert_eq!(logits.len(), n_vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "step {step}: non-finite logits",
        );
        // Detect a stuck forward path: at least one decode step should
        // produce logits that differ from the prefill snapshot somewhere.
        if !diff_seen
            && logits
                .iter()
                .zip(prefill_snapshot.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        {
            diff_seen = true;
        }
        let next =
            Session::argmax(logits).unwrap_or_else(|| panic!("step {step}: argmax returned None"));
        assert!((next as usize) < n_vocab);
        prev = next;
    }
    assert!(
        diff_seen,
        "decode logits never diverged from prefill — forward path may be stuck",
    );
}
