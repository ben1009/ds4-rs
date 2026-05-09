use anyhow::Result;
use std::sync::Arc;

use crate::engine::Engine;

/// An inference session holding mutable state.
pub struct Session {
    engine: Arc<Engine>,
    tokens: Vec<u32>,
    pos: u32,
    #[allow(dead_code)]
    ctx_size: u32,
}

impl Session {
    pub fn new(engine: Arc<Engine>, ctx_size: u32) -> Result<Self> {
        tracing::info!("Creating session with ctx_size={ctx_size}");
        Ok(Self {
            engine,
            tokens: Vec::new(),
            pos: 0,
            ctx_size,
        })
    }

    /// Run prefill for the entire prompt. Returns logits for the last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        tracing::info!("Prefill: {} tokens", tokens.len());
        // TODO: implement forward pass via Metal graph
        self.tokens.extend_from_slice(tokens);
        self.pos = tokens.len() as u32;
        Ok(vec![0.0; self.engine.config.n_vocab as usize])
    }

    /// Evaluate one decode token. Returns logits for the next token.
    pub fn eval_token(&mut self, token: u32) -> Result<Vec<f32>> {
        // TODO: implement single-token forward pass
        self.tokens.push(token);
        self.pos += 1;
        Ok(vec![0.0; self.engine.config.n_vocab as usize])
    }

    /// Greedy argmax: select the token with highest logit.
    pub fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    pub fn pos(&self) -> u32 {
        self.pos
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}
