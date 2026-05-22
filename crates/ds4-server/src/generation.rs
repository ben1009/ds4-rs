use std::{
    path::{Path, PathBuf},
    sync::Arc,
    thread::{self, JoinHandle},
};

use anyhow::Result;
use ds4_core::{engine::Engine, model::kv_disk, session::Session, speculative::SpecConfig};
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
    spec_config: Option<SpecConfig>,
) -> Result<(InferenceHandle, JoinHandle<()>)> {
    let (tx, mut rx) = mpsc::channel::<InferenceRequest>(32);

    let handle = thread::spawn(move || {
        let ctx_size = engine.config.ctx_size;
        let mut session = match Session::new(engine.clone(), ctx_size) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("failed to create initial session: {e}");
                return;
            }
        };

        while let Some(request) = rx.blocking_recv() {
            let result = process_request(
                &engine,
                &mut session,
                &request,
                &kv_cache_dir,
                spec_config.as_ref(),
            );

            if let Err(e) = result {
                let _ = request
                    .response_tx
                    .blocking_send(GenerationEvent::Error(e.to_string()));
            }
        }
    });

    Ok((InferenceHandle { tx }, handle))
}

fn kvc_save_path(cache_dir: &Path, tokens: &[u32]) -> PathBuf {
    // FNV-1a — stable across compiler versions, unlike DefaultHasher.
    let mut hash: u64 = 0xcbf29ce484222325;
    for tok in tokens {
        let bytes = tok.to_le_bytes();
        for b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    cache_dir.join(format!("{hash:016x}.kvc"))
}

fn process_request(
    engine: &Arc<Engine>,
    session: &mut Session,
    request: &InferenceRequest,
    kv_cache_dir: &Option<PathBuf>,
    spec_config: Option<&SpecConfig>,
) -> Result<()> {
    let eos_token = engine.tokenizer.eos_token();

    // Try prefix match if cache dir is set
    let suffix = if let Some(dir) = kv_cache_dir {
        match kv_disk::load_prefix_match(dir, &request.prompt_tokens, session, engine)? {
            Some(suffix) => suffix,
            None => {
                // No match — reuse session memory, just clear state
                session.invalidate();
                request.prompt_tokens.clone()
            }
        }
    } else {
        session.invalidate();
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
        let path = kvc_save_path(dir, &request.prompt_tokens);
        if let Err(e) = kv_disk::save_session(&path, session, kv_disk::SaveReason::Cold) {
            tracing::warn!("failed to save KVC after prefill: {e}");
        }
    }

    // Generate tokens
    let mut completion_tokens = 0u32;
    let logits = session.logits();
    let mut token =
        Session::argmax(logits).ok_or_else(|| anyhow::anyhow!("no logits available"))?;

    let mut finish_reason = "stop";
    loop {
        if token == eos_token {
            break;
        }
        if completion_tokens >= request.max_tokens {
            finish_reason = "length";
            break;
        }

        if let Some(spec) = spec_config {
            // Speculative decoding: may accept multiple tokens per step.
            let old_len = session.tokens().len();
            let next_token = {
                let logits = session.eval_token_speculative(engine, spec)?;
                Session::argmax(logits)
                    .ok_or_else(|| anyhow::anyhow!("speculative returned empty logits"))?
            };
            let new_tokens = &session.tokens()[old_len..];
            for &t in new_tokens {
                if t == eos_token {
                    finish_reason = "stop";
                    let _ = request.response_tx.blocking_send(GenerationEvent::Done {
                        prompt_tokens: request.prompt_tokens.len() as u32,
                        completion_tokens,
                        finish_reason,
                    });
                    return Ok(());
                }
                if completion_tokens >= request.max_tokens {
                    finish_reason = "length";
                    break;
                }
                let text = engine.tokenizer.decode(t).to_string();
                if request
                    .response_tx
                    .blocking_send(GenerationEvent::Token(text))
                    .is_err()
                {
                    return Ok(()); // client disconnected
                }
                completion_tokens += 1;
            }
            token = next_token;
            if next_token == eos_token {
                finish_reason = "stop";
                break;
            }
            if completion_tokens >= request.max_tokens {
                finish_reason = "length";
                break;
            }
        } else {
            // Standard single-token generation.
            let text = engine.tokenizer.decode(token).to_string();
            if request
                .response_tx
                .blocking_send(GenerationEvent::Token(text))
                .is_err()
            {
                return Ok(()); // client disconnected
            }
            completion_tokens += 1;
            session.eval_token(token)?;
            token = Session::argmax(session.logits())
                .ok_or_else(|| anyhow::anyhow!("no logits after eval"))?;
        }
    }

    let _ = request.response_tx.blocking_send(GenerationEvent::Done {
        prompt_tokens: request.prompt_tokens.len() as u32,
        completion_tokens,
        finish_reason,
    });

    Ok(())
}
