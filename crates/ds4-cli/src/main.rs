use anyhow::Result;
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;

use ds4_core::engine::Engine;
use ds4_core::session::Session;

/// Split `bytes` into (valid UTF-8 prefix length, invalid-sequence length).
/// If invalid_len > 0, those bytes should be replaced with U+FFFD and drained
/// together with the valid prefix. If invalid_len == 0, any bytes beyond the
/// valid prefix are a trailing partial multi-byte sequence and should be held.
fn utf8_split(bytes: &[u8]) -> (usize, usize) {
    match std::str::from_utf8(bytes) {
        Ok(_) => (bytes.len(), 0),
        Err(e) => match e.error_len() {
            Some(n) => (e.valid_up_to(), n),
            None => (e.valid_up_to(), 0),
        },
    }
}

/// Drain valid UTF-8 chunks from `pending`, replacing any invalid sequences
/// with U+FFFD. Trailing partial multi-byte sequences are left in `pending`
/// unless `flush_partial` is true (end-of-stream).
fn write_utf8<W: Write>(w: &mut W, pending: &mut Vec<u8>, flush_partial: bool) -> std::io::Result<()> {
    loop {
        let (valid, invalid) = utf8_split(pending);
        if valid > 0 {
            w.write_all(&pending[..valid])?;
        }
        if invalid > 0 {
            w.write_all("\u{FFFD}".as_bytes())?;
            pending.drain(..valid + invalid);
        } else {
            pending.drain(..valid);
            break;
        }
    }
    if flush_partial && !pending.is_empty() {
        w.write_all("\u{FFFD}".as_bytes())?;
        pending.clear();
    }
    Ok(())
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
                write_utf8(&mut handle, &mut pending, false)?;
                handle.flush()?;
                let logits = session.eval_token(token)?;
                token = Session::argmax(logits);
            }
            write_utf8(&mut handle, &mut pending, true)?;
            writeln!(handle)?;
        }
        None => {
            println!("Interactive mode not yet implemented. Use -p for one-shot generation.");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{utf8_split, write_utf8};

    #[test]
    fn holds_partial_multibyte() {
        assert_eq!(utf8_split(&[0xC3]), (0, 0));
        assert_eq!(utf8_split(&[0xC3, 0xA9]), (2, 0));
    }

    #[test]
    fn replaces_invalid_with_fffd() {
        let mut pending = vec![b'A', 0xFF, b'B'];
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, "A\u{FFFD}B".as_bytes());
        assert!(pending.is_empty());
    }

    #[test]
    fn end_of_stream_replaces_dangling_partial() {
        let mut pending = vec![b'A', 0xC3];
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, b"A");
        assert_eq!(pending, vec![0xC3]);
        write_utf8(&mut out, &mut pending, true).unwrap();
        assert_eq!(out, "A\u{FFFD}".as_bytes());
        assert!(pending.is_empty());
    }
}
