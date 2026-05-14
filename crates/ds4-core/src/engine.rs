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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses real filesystem, unsupported under miri isolation"
    )]
    fn open_missing_file_errors() {
        let result = Engine::open(Path::new("/nonexistent/ds4-engine-test/file.gguf"));
        assert!(result.is_err());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn open_empty_file_errors() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "ds4-engine-empty-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&[])
            .unwrap();
        let result = Engine::open(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn open_wrong_magic_errors() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "ds4-engine-badmagic-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        // Wrong magic bytes followed by zeros.
        let mut buf = vec![0u8; 32];
        buf[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
        let result = Engine::open(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses real filesystem, unsupported under miri isolation"
    )]
    fn open_directory_errors() {
        let result = Engine::open(&std::env::temp_dir());
        assert!(result.is_err());
    }
}
