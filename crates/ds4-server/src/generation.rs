use std::{
    path::PathBuf,
    sync::Arc,
    thread::{self, JoinHandle},
};

use anyhow::Result;
use ds4_core::{engine::Engine, model::kv_disk, session::Session};
use tokio::sync::mpsc;

use crate::types::GenerationEvent;

pub struct InferenceRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub response_tx: mpsc::Sender<GenerationEvent>,
}

#[derive(Clone)]
pub struct InferenceHandle {
    tx: mpsc::Sender<InferenceRequest>,
}

impl InferenceHandle {
    pub async fn submit(&self, request: InferenceRequest) -> Result<()> {
        self.tx
            .send(request)
            .await
            .map_err(|_| anyhow::anyhow!("inference worker shut down"))
    }
}

pub fn spawn_worker(
    engine: Arc<Engine>,
    kv_cache_dir: Option<PathBuf>,
) -> Result<(InferenceHandle, JoinHandle<()>)> {
    let (tx, mut rx) = mpsc::channel::<InferenceRequest>(4);

    let handle = thread::spawn(move || {
        let ctx_size = engine.config.ctx_size;
        let mut session =
            Session::new(engine.clone(), ctx_size).expect("failed to create initial session");

        while let Some(request) = rx.blocking_recv() {
            let result = process_request(&engine, &mut session, &request, &kv_cache_dir);

            if let Err(e) = result {
                let _ = request
                    .response_tx
                    .blocking_send(GenerationEvent::Error(e.to_string()));
            }
        }
    });

    Ok((InferenceHandle { tx }, handle))
}

fn process_request(
    engine: &Arc<Engine>,
    session: &mut Session,
    request: &InferenceRequest,
    kv_cache_dir: &Option<PathBuf>,
) -> Result<()> {
    let eos_token = engine.tokenizer.eos_token();
    let ctx_size = engine.config.ctx_size;

    // Try prefix match if cache dir is set
    let suffix = if let Some(dir) = kv_cache_dir {
        if let Some((mut loaded, suffix)) =
            kv_disk::load_prefix_match(dir, &request.prompt_tokens, engine)?
        {
            std::mem::swap(session, &mut loaded);
            suffix
        } else {
            // No match — reset session and prefill everything
            *session = Session::new(engine.clone(), ctx_size)?;
            request.prompt_tokens.clone()
        }
    } else {
        *session = Session::new(engine.clone(), ctx_size)?;
        request.prompt_tokens.clone()
    };

    // Prefill
    if suffix.is_empty() {
        session.recompute_last_logits()?;
    } else {
        session.prefill(&suffix)?;
    }

    // Save after prefill
    if let Some(dir) = kv_cache_dir {
        let _ = kv_disk::save_session(&dir.join("session.kvc"), session, kv_disk::SaveReason::Cold);
    }

    // Generate tokens
    let mut completion_tokens = 0u32;
    let logits = session.logits();
    let mut token =
        Session::argmax(logits).ok_or_else(|| anyhow::anyhow!("no logits available"))?;

    loop {
        if completion_tokens >= request.max_tokens {
            break;
        }

        let text = engine.tokenizer.decode(token).to_string();

        if request
            .response_tx
            .blocking_send(GenerationEvent::Token(text))
            .is_err()
        {
            return Ok(()); // client disconnected
        }

        completion_tokens += 1;

        if token == eos_token {
            break;
        }

        session.eval_token(token)?;
        token = Session::argmax(session.logits())
            .ok_or_else(|| anyhow::anyhow!("no logits after eval"))?;
    }

    // Save after generation
    if let Some(dir) = kv_cache_dir {
        let _ = kv_disk::save_session(
            &dir.join("session.kvc"),
            session,
            kv_disk::SaveReason::Continued,
        );
    }

    let _ = request.response_tx.blocking_send(GenerationEvent::Done {
        prompt_tokens: request.prompt_tokens.len() as u32,
        completion_tokens,
    });

    Ok(())
}
