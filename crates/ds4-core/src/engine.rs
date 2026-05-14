use std::{path::Path, sync::Arc};

use anyhow::Result;

use crate::{config::ModelConfig, model::WeightMap, ops::rope::RopeFreqs, tokenizer::Tokenizer};

/// The inference engine. Holds loaded model weights and tokenizer.
/// Immutable after creation — safe to share across threads via Arc.
pub struct Engine {
    pub weights: WeightMap,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
    /// Precomputed RoPE frequency cache (avoids re-computing sin/cos per layer).
    pub rope_freqs: RopeFreqs,
}

impl Engine {
    /// Load a model from a GGUF file.
    pub fn open(model_path: &Path) -> Result<Arc<Self>> {
        tracing::info!("Loading model from: {}", model_path.display());

        let weights = WeightMap::open(model_path)?;
        let config = weights.config.clone();
        let tokenizer = Tokenizer::from_metadata(weights.metadata())?;

        let rope_freqs = RopeFreqs::new(&crate::ops::rope::RopeParams {
            n_rot: 64,
            base: 10000.0,
            yarn: Some(crate::ops::rope::YarnParams {
                scale_factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                orig_ctx: 65536.0,
                attn_factor: None,
            }),
        });

        tracing::info!(
            "Engine ready: {} layers, {} vocab, ctx {}",
            config.n_layer,
            config.n_vocab,
            config.ctx_size,
        );

        Ok(Arc::new(Self {
            weights,
            tokenizer,
            config,
            rope_freqs,
        }))
    }
}
