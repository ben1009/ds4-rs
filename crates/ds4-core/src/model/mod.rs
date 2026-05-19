//! Model loading, weight accessors, layer views, and forward pass.
//!
//! See rfcs/0002-forward-pass.md §2. The module hierarchy:
//!
//! * `weights` — typed [`WeightMap`] accessors (`q8_0`, `f16`, `f32`, ...).
//! * `layer`   — per-layer borrowed weight views.
//! * `kv_cache` — MLA latent KV cache.
//! * `forward` — end-to-end forward pass orchestration.

use std::path::Path;

use anyhow::Result;

use crate::{
    config::ModelConfig,
    gguf::{GgufMmap, TensorInfo},
};

pub mod compressor;
pub mod forward;
pub mod kv_cache;
pub mod layer;
pub mod weights;

/// Memory-mapped model weights with named tensor lookup.
/// The mmap stays alive as long as WeightMap exists.
pub struct WeightMap {
    pub config: ModelConfig,
    gguf: GgufMmap,
}

impl WeightMap {
    /// Load a GGUF model file and extract config + weight map.
    pub fn open(path: &Path) -> Result<Self> {
        let gguf = GgufMmap::open(path)?;
        let config = ModelConfig::from_metadata(&gguf.content.metadata)?;

        tracing::info!(
            "Loaded model: {} layers, {} vocab, {} embd, {} experts",
            config.n_layer,
            config.n_vocab,
            config.n_embd,
            config.n_expert,
        );

        Ok(Self { config, gguf })
    }

    /// Get raw bytes for a tensor by name.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        self.gguf.tensor_data(name)
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.gguf.content.tensors.get(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.gguf
            .content
            .tensors
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    /// Access the underlying mmap.
    pub fn mmap(&self) -> &memmap2::Mmap {
        &self.gguf.mmap
    }

    /// Access GGUF metadata.
    pub fn metadata(&self) -> &std::collections::HashMap<String, crate::gguf::Value> {
        &self.gguf.content.metadata
    }
}
