//! End-to-end validation against DeepSeek API reference outputs.
//!
//! Compares ds4-rs greedy decoding against reference text captured from the
//! official DeepSeek API (temperature=0). Proves the Rust port produces
//! correct outputs for a set of representative prompts.
//!
//! Prerequisites:
//! 1. Generate reference files: `scripts/capture_api_vectors.sh`
//! 2. Set DS4_TEST_MODEL to a DS4 GGUF file
//!
//! Run:
//! ```text
//! DS4_TEST_MODEL=/path/to/ds4flash.gguf \
//!     cargo test -p ds4-core --test api_validation -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::{path::PathBuf, sync::Arc};

use ds4_core::{engine::Engine, session::Session};

/// DeepSeek V4 chat template prefix.
const TEMPLATE_PREFIX: &str = "<｜begin▁of▁sentence｜><｜User｜>";
const TEMPLATE_INFIX: &str = "<｜Assistant｜>";

fn model_path() -> Option<PathBuf> {
    std::env::var_os("DS4_TEST_MODEL").map(PathBuf::from)
}

fn vectors_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo");
    PathBuf::from(manifest).join("tests/vectors/api")
}

/// One reference test case loaded from JSON.
struct Reference {
    name: String,
    prompt: String,
    expected_text: String,
    max_tokens: u32,
}

fn load_references() -> Vec<Reference> {
    let dir = vectors_dir();
    if !dir.exists() {
        panic!(
            "Reference directory {} does not exist. \
             Run scripts/capture_api_vectors.sh first.",
            dir.display()
        );
    }

    let mut refs = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("read api vectors dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map_or(false, |ft| ft.is_file())
                && e.path().extension().is_some_and(|ext| ext == "json")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let data = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("failed to read {}: {e}", path.display());
        });
        let v: serde_json::Value = serde_json::from_str(&data).unwrap_or_else(|e| {
            panic!("failed to parse {}: {e}", path.display());
        });
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let prompt = v["prompt"].as_str().expect("prompt field").to_string();
        let expected_text = v["expected_text"]
            .as_str()
            .expect("expected_text field")
            .to_string();
        let max_tokens = v["max_tokens"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(64);
        refs.push(Reference {
            name,
            prompt,
            expected_text,
            max_tokens,
        });
    }

    refs
}

/// Apply the DeepSeek chat template to a user prompt.
fn apply_chat_template(prompt: &str) -> String {
    format!("{TEMPLATE_PREFIX}{prompt}{TEMPLATE_INFIX}")
}

/// Generate text from ds4-rs using greedy decoding, up to `max_bytes` of output.
fn generate_greedy(engine: &Arc<Engine>, prompt: &str, max_tokens: u32) -> String {
    let full_prompt = apply_chat_template(prompt);
    let tokens = engine.tokenizer.encode(&full_prompt, true);
    assert!(
        !tokens.is_empty(),
        "tokenizer returned empty tokens for prompt: {prompt:?}"
    );

    let prompt_len = u32::try_from(tokens.len()).expect("prompt too long for u32");
    let ctx_size = prompt_len
        .checked_add(max_tokens)
        .and_then(|n| n.checked_add(64))
        .and_then(|n| n.checked_next_power_of_two())
        .expect("requested context size exceeds u32 limits");
    let mut session = Session::new(engine.clone(), ctx_size).expect("Session::new failed");

    session.prefill(&tokens).expect("prefill failed");

    let eos = engine.tokenizer.eos_token();
    let mut output_bytes: Vec<u8> = Vec::new();
    let mut token = Session::argmax(session.logits()).expect("argmax on prefill logits");

    for _ in 0..max_tokens {
        if token == eos {
            break;
        }
        engine
            .tokenizer
            .append_token_bytes(token, &mut output_bytes);
        let logits = session.eval_token(token).expect("eval_token failed");
        token = Session::argmax(logits).expect("argmax returned None");
    }

    String::from_utf8_lossy(&output_bytes).into_owned()
}

fn compare_outputs(name: &str, expected: &str, actual: &str) {
    if expected == actual {
        eprintln!("  {name}: PASS ({} bytes)", actual.len());
        return;
    }

    // Find divergence at char level (safe for multi-byte UTF-8).
    let diverge_char = expected
        .chars()
        .zip(actual.chars())
        .position(|(a, b)| a != b)
        .unwrap_or(expected.chars().count().min(actual.chars().count()));

    let expected_preview: String = expected.chars().skip(diverge_char).take(20).collect();
    let actual_preview: String = actual.chars().skip(diverge_char).take(20).collect();

    panic!(
        "{name}: MISMATCH at char {diverge_char}\n\
         expected: ...{expected_preview:?}...\n\
         actual:   ...{actual_preview:?}...\n\
         expected len: {} bytes, actual len: {} bytes",
        expected.len(),
        actual.len(),
    );
}

#[test]
#[ignore = "needs DS4_TEST_MODEL pointing at a real DS4 GGUF and reference files from capture_api_vectors.sh"]
fn api_validation_greedy() {
    let path = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("DS4_TEST_MODEL not set — skipping.");
            return;
        }
    };
    if !path.exists() {
        panic!(
            "DS4_TEST_MODEL points at {} which does not exist",
            path.display()
        );
    }

    let refs = load_references();
    assert!(
        !refs.is_empty(),
        "no reference files found in {}. Run scripts/capture_api_vectors.sh.",
        vectors_dir().display()
    );

    let engine = Engine::open(&path).expect("Engine::open failed");

    eprintln!(
        "Running {} API validation tests against {}",
        refs.len(),
        path.display()
    );

    for reference in &refs {
        eprintln!("  generating: {} ({})", reference.name, reference.prompt);
        let actual = generate_greedy(&engine, &reference.prompt, reference.max_tokens);
        compare_outputs(&reference.name, &reference.expected_text, &actual);
    }

    eprintln!("All {} API validation tests passed.", refs.len());
}
