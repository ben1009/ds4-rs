mod chat_template;
mod generation;
mod handlers;
mod sse;
mod types;

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use clap::Parser;
use ds4_core::{engine::Engine, speculative::SpecConfig};
use generation::InferenceHandle;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "ds4-server", about = "DS4 inference server")]
struct Args {
    /// Path to the model file
    #[arg(long)]
    model: PathBuf,

    /// Server port
    #[arg(long, default_value = "8080")]
    port: u16,

    /// KV cache directory for session persistence
    #[arg(long)]
    kv_cache_dir: Option<PathBuf>,

    /// Path to MTP draft model GGUF (enables speculative decoding)
    #[arg(long)]
    mtp: Option<PathBuf>,

    /// Maximum number of MTP draft tokens per step
    #[arg(long, default_value = "1")]
    mtp_draft: usize,

    /// Logit margin threshold for MTP confidence gating
    #[arg(long, default_value = "3.0")]
    mtp_margin: f32,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
    inference: InferenceHandle,
    model_id: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    tracing::info!("loading model from {}", args.model.display());
    let engine = match &args.mtp {
        Some(mtp_path) => Engine::open_with_mtp(&args.model, mtp_path)?,
        None => Engine::open(&args.model)?,
    };

    if let Some(ref dir) = args.kv_cache_dir {
        std::fs::create_dir_all(dir)?;
    }

    let spec_config = args.mtp.as_ref().map(|_| SpecConfig {
        mtp_draft_tokens: args.mtp_draft,
        mtp_margin: args.mtp_margin,
    });

    let (inference, _worker) =
        generation::spawn_worker(engine.clone(), args.kv_cache_dir.clone(), spec_config)?;

    let model_id = args
        .model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("ds4")
        .to_string();

    let state = AppState {
        engine,
        inference,
        model_id,
    };

    let app = axum::Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(handlers::openai_chat_completions),
        )
        .route(
            "/v1/completions",
            axum::routing::post(handlers::openai_completions),
        )
        .route("/v1/models", axum::routing::get(handlers::openai_models))
        .route(
            "/v1/messages",
            axum::routing::post(handlers::anthropic_messages),
        )
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    tracing::info!("listening on {addr}");
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
