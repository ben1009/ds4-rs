use std::sync::Arc;

use anyhow::Result;

use crate::engine::Engine;

/// An inference session holding mutable state.
pub struct Session {
    engine: Arc<Engine>,
    tokens: Vec<u32>,
    pos: u32,
    #[allow(dead_code)]
    ctx_size: u32,
    logits: Vec<f32>,
}

impl Session {
    pub fn new(engine: Arc<Engine>, ctx_size: u32) -> Result<Self> {
        tracing::info!("Creating session with ctx_size={ctx_size}");
        let n_vocab = engine.config.n_vocab as usize;
        Ok(Self {
            engine,
            tokens: Vec::new(),
            pos: 0,
            ctx_size,
            logits: vec![0.0; n_vocab],
        })
    }

    /// Run prefill for the entire prompt. Returns logits for the last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        tracing::info!("Prefill: {} tokens", tokens.len());
        // TODO: implement forward pass
        self.tokens.extend_from_slice(tokens);
        self.pos = self.tokens.len() as u32;
        self.logits.fill(0.0);
        Ok(&self.logits)
    }

    /// Evaluate one decode token. Returns logits for the next token.
    pub fn eval_token(&mut self, token: u32) -> Result<&[f32]> {
        // TODO: implement single-token forward pass
        self.tokens.push(token);
        self.pos += 1;
        self.logits.fill(0.0);
        Ok(&self.logits)
    }

    /// Greedy argmax: select the token with highest logit.
    /// NaN values are ignored. Returns `None` if `logits` is empty.
    pub fn argmax(logits: &[f32]) -> Option<u32> {
        let (idx, _) = logits.iter().enumerate().fold(
            (None, f32::NEG_INFINITY),
            |(idx_max, val_max), (idx, &val)| {
                if val > val_max {
                    (Some(idx as u32), val)
                } else {
                    (idx_max, val_max)
                }
            },
        );
        idx
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
