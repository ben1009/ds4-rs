use std::{path::Path, sync::Arc};

use anyhow::{Result, bail};

use crate::{config::ModelConfig, model::WeightMap, ops::rope::RopeFreqs, tokenizer::Tokenizer};

/// Hardcoded RoPE frequency base, matches `DS4_ROPE_FREQ_BASE` in antirez/ds4.
/// Cross-checked against the GGUF `*.rope.freq_base` metadata at load time.
const ROPE_FREQ_BASE: f32 = 10000.0;
/// Long-context RoPE frequency base used by the streaming compressor's
/// emitted-row rotation. Mirrors `DS4_ROPE_LONG_FREQ_BASE` in ds4.c — the
/// compressor sees position indices on the order of `pos / ratio`, so the
/// rotation runs at a different base from the per-token attention RoPE.
const ROPE_LONG_FREQ_BASE: f32 = 160_000.0;
/// Long-context YaRN scale factor (compressor RoPE only). Matches the
/// `freq_scale = 1/16` in the C reference.
const ROPE_LONG_SCALE_FACTOR: f32 = 16.0;

/// The inference engine. Holds loaded model weights and tokenizer.
/// Immutable after creation — safe to share across threads via Arc.
pub struct Engine {
    pub weights: WeightMap,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
    /// Precomputed RoPE frequency cache (avoids re-computing sin/cos per layer).
    pub rope_freqs: RopeFreqs,
    /// Long-context RoPE frequency cache used by the streaming compressor.
    /// `freq_base = 160_000`, `scale_factor = 16`. Built once at engine open
    /// alongside [`Engine::rope_freqs`].
    pub rope_freqs_long: RopeFreqs,
    /// Optional MTP draft model weights loaded from a separate GGUF file.
    /// `None` when speculative decoding is not enabled.
    pub mtp_weights: Option<WeightMap>,
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

        // Long-context RoPE for the streaming compressor (attn_compressor_*).
        // The base is `160_000` and the scale matches the dense path's
        // `1/16` so that, beyond the YaRN ramp, dimensions interpolate the
        // same way. The C reference passes the same YaRN params for both
        // freq caches; only the base differs.
        let rope_freqs_long = RopeFreqs::new(&crate::ops::rope::RopeParams {
            n_rot: 64,
            base: ROPE_LONG_FREQ_BASE,
            yarn: Some(crate::ops::rope::YarnParams {
                scale_factor: ROPE_LONG_SCALE_FACTOR,
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
            rope_freqs_long,
            mtp_weights: None,
        }))
    }

    /// Load a model with an optional MTP draft model for speculative decoding.
    ///
    /// The MTP GGUF must have compatible `n_embd` and `n_vocab` values.
    pub fn open_with_mtp(model_path: &Path, mtp_path: &Path) -> Result<Arc<Self>> {
        let mut engine = Self::open(model_path)?;
        let mtp_map = WeightMap::open(mtp_path)?;

        // Validate compatibility.
        let main = &engine.config;
        let mtp_cfg = &mtp_map.config;
        if mtp_cfg.n_embd != main.n_embd {
            bail!(
                "MTP n_embd mismatch: main={}, mtp={}",
                main.n_embd,
                mtp_cfg.n_embd
            );
        }
        if mtp_cfg.n_vocab != main.n_vocab {
            bail!(
                "MTP n_vocab mismatch: main={}, mtp={}",
                main.n_vocab,
                mtp_cfg.n_vocab
            );
        }

        tracing::info!("MTP draft model loaded from: {}", mtp_path.display());
        Arc::get_mut(&mut engine)
            .expect("open_with_mtp: sole owner")
            .mtp_weights = Some(mtp_map);
        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ds4-engine-{tag}-{}-{seq}.gguf",
            std::process::id()
        ))
    }

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
        let path = unique_path("empty");
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
        let path = unique_path("badmagic");
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
