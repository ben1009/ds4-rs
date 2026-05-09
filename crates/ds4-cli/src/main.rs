use anyhow::Result;
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;

use ds4_core::engine::Engine;
use ds4_core::session::Session;

/// ds4-rs: DeepSeek V4 Flash inference engine for Apple Metal
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
            let tokens = engine.tokenizer.encode(&prompt);
            tracing::info!("Prompt: {} tokens", tokens.len());

            let logits = session.prefill(&tokens)?;

            let mut token = Session::argmax(&logits);
            let mut generated = Vec::new();

            for _ in 0..args.max_tokens {
                if token == engine.tokenizer.eos_token() {
                    break;
                }
                generated.push(token);
                let logits = session.eval_token(token)?;
                token = Session::argmax(&logits);
            }

            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            write!(handle, "{}", engine.tokenizer.decode_tokens(&generated))?;
            writeln!(handle)?;
        }
        None => {
            println!("Interactive mode not yet implemented. Use -p for one-shot generation.");
        }
    }

    Ok(())
}
