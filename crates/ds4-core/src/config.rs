use anyhow::Result;

use crate::gguf::Value;

/// DeepSeek V4 Flash model configuration extracted from GGUF metadata.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub n_layer: u32,
    pub n_ff: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_hc: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    pub ctx_size: u32,
}

impl ModelConfig {
    pub fn from_metadata(metadata: &std::collections::HashMap<String, Value>) -> Result<Self> {
        let get_u32 = |key: &str| -> Result<u32> {
            metadata
                .get(key)
                .and_then(|v| v.to_u32())
                .ok_or_else(|| anyhow::anyhow!("Missing metadata key: {key}"))
        };

        let get_f32 = |key: &str| -> Result<f32> {
            metadata
                .get(key)
                .and_then(|v| v.to_f32())
                .ok_or_else(|| anyhow::anyhow!("Missing metadata key: {key}"))
        };

        let n_embd = get_u32("llama.embedding_length")?;

        Ok(Self {
            n_vocab: get_u32("llama.vocab_size")?,
            n_embd,
            n_head: get_u32("llama.attention.head_count")?,
            n_kv_head: get_u32("llama.attention.head_count_kv")?,
            n_layer: get_u32("llama.block_count")?,
            n_ff: get_u32("llama.feed_forward_length")?,
            n_expert: get_u32("llama.expert_count").ok().unwrap_or(0),
            n_expert_used: get_u32("llama.expert_used_count").ok().unwrap_or(0),
            n_hc: get_u32("ds4.hc_count").ok().unwrap_or(4),
            head_dim: get_u32("llama.attention.key_length")
                .or_else(|_| get_u32("llama.attention.head_dim"))
                .ok()
                .unwrap_or_else(|| {
                    n_embd / get_u32("llama.attention.head_count")
                        .ok()
                        .filter(|&v| v != 0)
                        .unwrap_or(1)
                }),
            rope_theta: get_f32("llama.rope.freq_base").ok().unwrap_or(10000.0),
            ctx_size: get_u32("llama.context_length").ok().unwrap_or(32768),
        })
    }
}
