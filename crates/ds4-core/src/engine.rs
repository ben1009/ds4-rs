use std::{path::Path, sync::Arc};

use anyhow::{Result, bail};

use crate::{config::ModelConfig, model::WeightMap, ops::rope::RopeFreqs, tokenizer::Tokenizer};

/// Hardcoded RoPE frequency base, matches `DS4_ROPE_FREQ_BASE` in antirez/ds4.
/// Cross-checked against the GGUF `*.rope.freq_base` metadata at load time.
const ROPE_FREQ_BASE: f32 = 10000.0;

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

        // Match antirez/ds4: `config_expect_f32("rope.freq_base", …, DS4_ROPE_FREQ_BASE)`.
        // The metadata value isn't an override; it's a sanity check on the GGUF.
        // Bitwise compare — the upstream check is exact equality on the same
        // 10000.0 constant, so any mismatch deserves a hard failure rather
        // than a tolerance bucket.
        if config.rope_theta != ROPE_FREQ_BASE {
            bail!(
                "rope.freq_base mismatch: GGUF metadata says {}, expected {ROPE_FREQ_BASE}",
                config.rope_theta
            );
        }

        let rope_freqs = RopeFreqs::new(&crate::ops::rope::RopeParams {
            n_rot: 64,
            base: ROPE_FREQ_BASE,
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
