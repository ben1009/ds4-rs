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

        let n_vocab = get_u32("llama.vocab_size")?;
        let n_embd = get_u32("llama.embedding_length")?;
        let n_head = get_u32("llama.attention.head_count")?;
        let n_kv_head = get_u32("llama.attention.head_count_kv")?;
        let n_layer = get_u32("llama.block_count")?;
        let n_ff = get_u32("llama.feed_forward_length")?;

        let head_dim = get_u32("llama.attention.key_length")
            .or_else(|_| get_u32("llama.attention.head_dim"))
            .unwrap_or_else(|_| n_embd.checked_div(n_head).unwrap_or(n_embd));

        Ok(Self {
            n_vocab,
            n_embd,
            n_head,
            n_kv_head,
            n_layer,
            n_ff,
            n_expert: get_u32("llama.expert_count").unwrap_or(0),
            n_expert_used: get_u32("llama.expert_used_count").unwrap_or(0),
            n_hc: get_u32("ds4.hc_count").unwrap_or(4),
            head_dim,
            rope_theta: get_f32("llama.rope.freq_base").unwrap_or(10000.0),
            ctx_size: get_u32("llama.context_length").unwrap_or(32768),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn base_metadata() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("llama.vocab_size".to_string(), Value::U32(32000));
        m.insert("llama.embedding_length".to_string(), Value::U32(4096));
        m.insert("llama.attention.head_count".to_string(), Value::U32(32));
        m.insert("llama.attention.head_count_kv".to_string(), Value::U32(8));
        m.insert("llama.block_count".to_string(), Value::U32(32));
        m.insert("llama.feed_forward_length".to_string(), Value::U32(14336));
        m
    }

    #[test]
    fn mandatory_fields_parsed() {
        let cfg = ModelConfig::from_metadata(&base_metadata()).unwrap();
        assert_eq!(cfg.n_vocab, 32000);
        assert_eq!(cfg.n_embd, 4096);
        assert_eq!(cfg.n_head, 32);
        assert_eq!(cfg.n_kv_head, 8);
        assert_eq!(cfg.n_layer, 32);
        assert_eq!(cfg.n_ff, 14336);
    }

    #[test]
    fn optional_fields_have_defaults() {
        let cfg = ModelConfig::from_metadata(&base_metadata()).unwrap();
        assert_eq!(cfg.n_expert, 0);
        assert_eq!(cfg.n_expert_used, 0);
        assert_eq!(cfg.n_hc, 4);
        assert_eq!(cfg.rope_theta, 10000.0);
        assert_eq!(cfg.ctx_size, 32768);
    }

    #[test]
    fn head_dim_derived_from_n_embd_over_n_head() {
        // 4096 / 32 = 128
        let cfg = ModelConfig::from_metadata(&base_metadata()).unwrap();
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn head_dim_uses_key_length_when_present() {
        let mut m = base_metadata();
        m.insert("llama.attention.key_length".to_string(), Value::U32(256));
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert_eq!(cfg.head_dim, 256);
    }

    #[test]
    fn head_dim_div_by_zero_falls_back_to_n_embd() {
        let mut m = base_metadata();
        m.insert("llama.attention.head_count".to_string(), Value::U32(0));
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        // checked_div on 0 returns None, so head_dim falls back to n_embd.
        assert_eq!(cfg.head_dim, 4096);
    }

    #[test]
    fn optional_overrides_take_effect() {
        let mut m = base_metadata();
        m.insert("llama.expert_count".to_string(), Value::U32(64));
        m.insert("llama.expert_used_count".to_string(), Value::U32(2));
        m.insert("ds4.hc_count".to_string(), Value::U32(8));
        m.insert("llama.rope.freq_base".to_string(), Value::F32(500000.0));
        m.insert("llama.context_length".to_string(), Value::U32(16384));
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert_eq!(cfg.n_expert, 64);
        assert_eq!(cfg.n_expert_used, 2);
        assert_eq!(cfg.n_hc, 8);
        assert_eq!(cfg.rope_theta, 500000.0);
        assert_eq!(cfg.ctx_size, 16384);
    }

    #[test]
    fn missing_mandatory_field_errors() {
        let mut m = base_metadata();
        m.remove("llama.vocab_size");
        assert!(ModelConfig::from_metadata(&m).is_err());
    }
}
