use anyhow::Result;
use std::path::Path;

use crate::config::ModelConfig;
use crate::gguf::{GgufMmap, TensorInfo};

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
        self.gguf.content.tensors.keys().map(|s| s.as_str()).collect()
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
