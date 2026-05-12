use anyhow::Result;
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;

use ds4_core::engine::Engine;
use ds4_core::session::Session;

/// Length of the longest valid UTF-8 prefix of `bytes`. Any trailing partial
/// multi-byte sequence is left for the next call so streaming output doesn't
/// split characters across flushes.
fn utf8_valid_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) => {
            if e.error_len().is_some() {
                // Real invalid byte — flush everything up to and including it;
                // from_utf8_lossy on the next flush will emit U+FFFD for it.
                bytes.len()
            } else {
                e.valid_up_to()
            }
        }
    }
}

/// ds4-rs: DeepSeek V4 Flash inference engine for Linux
#[derive(Parser)]
#[command(name = "ds4", version)]
struct Args {
    /// Path to the GGUF model file
    #[arg(long, default_value = "./ds4flash.gguf")]
    model: PathBuf,

    /// Prompt for one-shot generation
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value = "256")]
    max_tokens: u32,

    /// Context size
    #[arg(long, default_value = "32768")]
    ctx: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    let engine = Engine::open(&args.model)?;

    match args.prompt {
        Some(prompt) => {
            let mut session = Session::new(engine.clone(), args.ctx)?;
            let tokens = engine.tokenizer.encode(&prompt, true);
            tracing::info!("Prompt: {} tokens", tokens.len());

            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            let eos = engine.tokenizer.eos_token();

            let logits = session.prefill(&tokens)?;
            let mut token = Session::argmax(logits);
            let mut pending: Vec<u8> = Vec::new();

            for _ in 0..args.max_tokens {
                if token == eos {
                    break;
                }
                engine.tokenizer.append_token_bytes(token, &mut pending);
                let split = utf8_valid_prefix_len(&pending);
                if split > 0 {
                    handle.write_all(&pending[..split])?;
                    handle.flush()?;
                    pending.drain(..split);
                }
                let logits = session.eval_token(token)?;
                token = Session::argmax(logits);
            }
            if !pending.is_empty() {
                // Any trailing invalid bytes: replace with U+FFFD once.
                let tail = String::from_utf8_lossy(&pending);
                handle.write_all(tail.as_bytes())?;
            }
            writeln!(handle)?;
        }
        None => {
            println!("Interactive mode not yet implemented. Use -p for one-shot generation.");
        }
    }

    Ok(())
}
