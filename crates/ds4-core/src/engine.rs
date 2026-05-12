use std::{path::Path, sync::Arc};

use anyhow::Result;

use crate::{config::ModelConfig, model::WeightMap, tokenizer::Tokenizer};

/// The inference engine. Holds loaded model weights and tokenizer.
/// Immutable after creation — safe to share across threads via Arc.
pub struct Engine {
    pub weights: WeightMap,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
}

impl Engine {
    /// Load a model from a GGUF file.
    pub fn open(model_path: &Path) -> Result<Arc<Self>> {
        tracing::info!("Loading model from: {}", model_path.display());

        let weights = WeightMap::open(model_path)?;
        let config = weights.config.clone();
        let tokenizer = Tokenizer::from_metadata(weights.metadata())?;

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
        }))
    }
}
