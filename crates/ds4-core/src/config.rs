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
            let v = metadata
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("Missing metadata key: {key}"))?;
            v.to_u32()
                .ok_or_else(|| anyhow::anyhow!("Type mismatch for {key}: expected u32"))
        };

        let get_f32 = |key: &str| -> Result<f32> {
            let v = metadata
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("Missing metadata key: {key}"))?;
            v.to_f32()
                .ok_or_else(|| anyhow::anyhow!("Type mismatch for {key}: expected f32"))
        };

        let n_vocab = get_u32("deepseek4.vocab_size")?;
        let n_embd = get_u32("deepseek4.embedding_length")?;
        let n_head = get_u32("deepseek4.attention.head_count")?;
        if n_head == 0 {
            anyhow::bail!("deepseek4.attention.head_count must be > 0");
        }
        let n_kv_head = get_u32("deepseek4.attention.head_count_kv")?;
        let n_layer = get_u32("deepseek4.block_count")?;
        let n_ff = get_u32("deepseek4.expert_feed_forward_length")?;

        let head_dim = get_u32("deepseek4.attention.key_length")
            .or_else(|_| get_u32("deepseek4.attention.head_dim"))
            .unwrap_or(n_embd / n_head);

        Ok(Self {
            n_vocab,
            n_embd,
            n_head,
            n_kv_head,
            n_layer,
            n_ff,
            n_expert: get_u32("deepseek4.expert_count").unwrap_or(0),
            n_expert_used: get_u32("deepseek4.expert_used_count").unwrap_or(0),
            n_hc: get_u32("deepseek4.hyper_connection.count").unwrap_or(4),
            head_dim,
            rope_theta: get_f32("deepseek4.rope.freq_base").unwrap_or(10000.0),
            ctx_size: get_u32("deepseek4.context_length").unwrap_or(32768),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn base_metadata() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("deepseek4.vocab_size".to_string(), Value::U32(32000));
        m.insert("deepseek4.embedding_length".to_string(), Value::U32(4096));
        m.insert("deepseek4.attention.head_count".to_string(), Value::U32(32));
        m.insert(
            "deepseek4.attention.head_count_kv".to_string(),
            Value::U32(8),
        );
        m.insert("deepseek4.block_count".to_string(), Value::U32(32));
        m.insert(
            "deepseek4.expert_feed_forward_length".to_string(),
            Value::U32(14336),
        );
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
        m.insert(
            "deepseek4.attention.key_length".to_string(),
            Value::U32(256),
        );
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert_eq!(cfg.head_dim, 256);
    }

    #[test]
    fn head_count_zero_errors() {
        let mut m = base_metadata();
        m.insert("deepseek4.attention.head_count".to_string(), Value::U32(0));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(
            err.to_string().contains("head_count"),
            "expected head_count error, got: {err}"
        );
    }

    #[test]
    fn optional_overrides_take_effect() {
        let mut m = base_metadata();
        m.insert("deepseek4.expert_count".to_string(), Value::U32(64));
        m.insert("deepseek4.expert_used_count".to_string(), Value::U32(2));
        m.insert(
            "deepseek4.hyper_connection.count".to_string(),
            Value::U32(8),
        );
        m.insert("deepseek4.rope.freq_base".to_string(), Value::F32(500000.0));
        m.insert("deepseek4.context_length".to_string(), Value::U32(16384));
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
        m.remove("deepseek4.vocab_size");
        assert!(ModelConfig::from_metadata(&m).is_err());
    }

    #[test]
    fn type_mismatch_for_u32_errors() {
        let mut m = base_metadata();
        m.insert(
            "deepseek4.vocab_size".to_string(),
            Value::String("not a number".to_string()),
        );
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Type mismatch") && msg.contains("deepseek4.vocab_size"),
            "expected type-mismatch error mentioning the key, got: {msg}"
        );
    }

    #[test]
    fn u32_field_accepts_u64_in_range() {
        let mut m = base_metadata();
        m.insert("deepseek4.vocab_size".to_string(), Value::U64(50000));
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert_eq!(cfg.n_vocab, 50000);
    }

    #[test]
    fn u32_field_rejects_u64_overflow() {
        let mut m = base_metadata();
        m.insert("deepseek4.vocab_size".to_string(), Value::U64(u64::MAX));
        assert!(ModelConfig::from_metadata(&m).is_err());
    }

    #[test]
    fn head_dim_falls_back_to_head_dim_key() {
        let mut m = base_metadata();
        m.insert("deepseek4.attention.head_dim".to_string(), Value::U32(192));
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert_eq!(cfg.head_dim, 192);
    }

    #[test]
    fn key_length_takes_precedence_over_head_dim() {
        let mut m = base_metadata();
        m.insert(
            "deepseek4.attention.key_length".to_string(),
            Value::U32(160),
        );
        m.insert("deepseek4.attention.head_dim".to_string(), Value::U32(192));
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert_eq!(cfg.head_dim, 160);
    }

    #[test]
    fn rope_theta_widens_from_f64() {
        let mut m = base_metadata();
        m.insert(
            "deepseek4.rope.freq_base".to_string(),
            Value::F64(123_456.789),
        );
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert!((cfg.rope_theta - 123_456.79_f32).abs() < 1.0);
    }

    #[test]
    fn rope_theta_type_mismatch_falls_back_to_default() {
        let mut m = base_metadata();
        m.insert(
            "deepseek4.rope.freq_base".to_string(),
            Value::String("abc".to_string()),
        );
        let cfg = ModelConfig::from_metadata(&m).unwrap();
        assert_eq!(cfg.rope_theta, 10000.0);
    }

    #[test]
    fn each_mandatory_field_individually_required() {
        let keys = [
            "deepseek4.vocab_size",
            "deepseek4.embedding_length",
            "deepseek4.attention.head_count",
            "deepseek4.attention.head_count_kv",
            "deepseek4.block_count",
            "deepseek4.expert_feed_forward_length",
        ];
        for k in keys {
            let mut m = base_metadata();
            m.remove(k);
            let err = ModelConfig::from_metadata(&m).unwrap_err();
            assert!(
                err.to_string().contains(k),
                "expected error to mention {k}, got: {err}"
            );
        }
    }
}
