use anyhow::Result;

use crate::gguf::Value;

/// Mirrors `ds4_layer_compress_ratio` in antirez/ds4 ds4.c (lines 410-415):
///   * layers `0..2` are dense (ratio = 0)
///   * even layers `>= 2` use ratio 4 (also carry an indexer; PR 3)
///   * odd layers `>= 2` use ratio 128
pub fn layer_compress_ratio(il: u32) -> u32 {
    if il < 2 {
        0
    } else if il & 1 == 0 {
        4
    } else {
        128
    }
}

/// GGUF metadata key holding the per-layer compress ratio array. Mirrors
/// `validate_compress_ratio_metadata` in antirez/ds4 ds4.c (lines 2474-2505).
pub const COMPRESS_RATIOS_KEY: &str = "deepseek4.attention.compress_ratios";

/// Indexer head count. Hard-coded to 64 in the C reference
/// (`DS4_N_INDEXER_HEAD`); GGUF carries it under
/// `deepseek4.attention.indexer.head_count` and `validate_*` checks it
/// against this value.
pub const INDEXER_HEAD: u32 = 64;
/// Indexer head dim. Hard-coded to 128 in the C reference
/// (`DS4_N_INDEXER_HEAD_DIM`).
pub const INDEXER_HEAD_DIM: u32 = 128;
/// Indexer top-k cap. Hard-coded to 512 in the C reference
/// (`DS4_N_INDEXER_TOP_K`). When `n_comp > top_k`, the indexer picks the
/// `top_k` highest-scoring compressed rows and masks the rest.
pub const INDEXER_TOP_K: u32 = 512;

/// GGUF metadata keys for the indexer constants. The C reference reads
/// these and `config_expect_u32`s each one against the matching `INDEXER_*`
/// constant. We mirror that as optional validation: keys may be absent on
/// the synthetic minimal-GGUF tests, but if present they must match.
pub const INDEXER_HEAD_KEY: &str = "deepseek4.attention.indexer.head_count";
pub const INDEXER_HEAD_DIM_KEY: &str = "deepseek4.attention.indexer.key_length";
pub const INDEXER_TOP_K_KEY: &str = "deepseek4.attention.indexer.top_k";

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
    /// Q LoRA rank — sizes the `attn_q_a` down-projection output and
    /// `attn_q_a_norm` (`deepseek4.attention.q_lora_rank`).
    pub q_lora_rank: u32,
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
        let q_lora_rank = get_u32("deepseek4.attention.q_lora_rank")?;
        if q_lora_rank == 0 {
            anyhow::bail!("deepseek4.attention.q_lora_rank must be > 0");
        }

        let head_dim = get_u32("deepseek4.attention.key_length")
            .or_else(|_| get_u32("deepseek4.attention.head_dim"))
            .unwrap_or(n_embd / n_head);

        // Indexer constants are hard-coded in the C reference but the
        // GGUF carries them; if present, validate them against our
        // constants so a model with an unexpected indexer shape fails
        // fast at load time. Mirrors `config_expect_u32` calls in ds4.c
        // (lines 2608-2610).
        for (key, expected) in [
            (INDEXER_HEAD_KEY, INDEXER_HEAD),
            (INDEXER_HEAD_DIM_KEY, INDEXER_HEAD_DIM),
            (INDEXER_TOP_K_KEY, INDEXER_TOP_K),
        ] {
            if let Some(v) = metadata.get(key) {
                let got = v
                    .to_u32()
                    .ok_or_else(|| anyhow::anyhow!("Type mismatch for {key}: expected u32"))?;
                if got != expected {
                    anyhow::bail!("{key}: GGUF says {got}, expected {expected}");
                }
            }
        }

        // Validate `deepseek4.attention.compress_ratios` against the
        // hard-coded `ds4_layer_compress_ratio` schedule when present. The
        // key is optional for back-compat with the synthetic minimal-GGUF
        // tests, matching `validate_compress_ratio_metadata` in ds4.c.
        if let Some(value) = metadata.get(COMPRESS_RATIOS_KEY) {
            let arr = value.to_array().ok_or_else(|| {
                anyhow::anyhow!("{COMPRESS_RATIOS_KEY}: expected array, got {value:?}")
            })?;
            if arr.len() < n_layer as usize {
                anyhow::bail!(
                    "{COMPRESS_RATIOS_KEY}: array length {} < n_layer {n_layer}",
                    arr.len(),
                );
            }
            for (il, entry) in arr.iter().take(n_layer as usize).enumerate() {
                let got = entry.to_u32().ok_or_else(|| {
                    anyhow::anyhow!("{COMPRESS_RATIOS_KEY}[{il}]: expected u32, got {entry:?}")
                })?;
                let want = layer_compress_ratio(il as u32);
                if got != want {
                    anyhow::bail!("{COMPRESS_RATIOS_KEY}[{il}]: GGUF says {got}, expected {want}",);
                }
            }
        }

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
            q_lora_rank,
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
        m.insert(
            "deepseek4.attention.q_lora_rank".to_string(),
            Value::U32(1024),
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
        assert_eq!(cfg.q_lora_rank, 1024);
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
    fn q_lora_rank_zero_errors() {
        let mut m = base_metadata();
        m.insert("deepseek4.attention.q_lora_rank".to_string(), Value::U32(0));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(
            err.to_string().contains("q_lora_rank"),
            "expected q_lora_rank error, got: {err}"
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
            "deepseek4.attention.q_lora_rank",
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

    // ---------------------------------------------------------------------
    // layer_compress_ratio + compress_ratios metadata validation
    // ---------------------------------------------------------------------

    #[test]
    fn layer_compress_ratio_truth_table() {
        // n_layer for DS4 Flash = 43, so layer indices run 0..=42.
        assert_eq!(layer_compress_ratio(0), 0);
        assert_eq!(layer_compress_ratio(1), 0);
        assert_eq!(layer_compress_ratio(2), 4);
        assert_eq!(layer_compress_ratio(3), 128);
        assert_eq!(layer_compress_ratio(4), 4);
        assert_eq!(layer_compress_ratio(5), 128);
        assert_eq!(layer_compress_ratio(41), 128);
        assert_eq!(layer_compress_ratio(42), 4);
    }

    fn ratios_for(n_layer: u32) -> Vec<Value> {
        (0..n_layer)
            .map(|il| Value::U32(layer_compress_ratio(il)))
            .collect()
    }

    #[test]
    fn compress_ratios_matching_array_passes() {
        let mut m = base_metadata();
        // base_metadata uses n_layer = 32, but the schedule is fully
        // determined by the layer index, so any prefix length >= n_layer
        // works.
        m.insert(
            COMPRESS_RATIOS_KEY.to_string(),
            Value::Array(ratios_for(32)),
        );
        ModelConfig::from_metadata(&m).expect("matching array should pass");
    }

    #[test]
    fn compress_ratios_missing_is_allowed() {
        // No COMPRESS_RATIOS_KEY in base_metadata — should still parse.
        ModelConfig::from_metadata(&base_metadata()).expect("missing array OK for back-compat");
    }

    #[test]
    fn compress_ratios_mismatch_errors() {
        let mut m = base_metadata();
        let mut arr = ratios_for(32);
        // Flip layer 4 from 4 -> 128 to force a mismatch against the schedule.
        arr[4] = Value::U32(128);
        m.insert(COMPRESS_RATIOS_KEY.to_string(), Value::Array(arr));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("compress_ratios") && msg.contains("[4]"),
            "expected compress_ratios[4] error, got: {msg}",
        );
    }

    #[test]
    fn compress_ratios_too_short_errors() {
        let mut m = base_metadata();
        m.insert(COMPRESS_RATIOS_KEY.to_string(), Value::Array(ratios_for(8)));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(
            err.to_string().contains("array length"),
            "expected array length error, got: {err}",
        );
    }

    #[test]
    fn compress_ratios_wrong_value_type_errors() {
        let mut m = base_metadata();
        let mut arr = ratios_for(32);
        arr[3] = Value::String("nope".to_string());
        m.insert(COMPRESS_RATIOS_KEY.to_string(), Value::Array(arr));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(err.to_string().contains("[3]"), "got: {err}");
    }

    #[test]
    fn compress_ratios_non_array_errors() {
        let mut m = base_metadata();
        m.insert(COMPRESS_RATIOS_KEY.to_string(), Value::U32(4));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(err.to_string().contains("expected array"), "got: {err}");
    }

    // ---------------------------------------------------------------------
    // Indexer constant validation
    // ---------------------------------------------------------------------

    #[test]
    fn indexer_keys_optional_for_back_compat() {
        // base_metadata has none of the indexer keys — must still parse.
        ModelConfig::from_metadata(&base_metadata()).expect("missing indexer keys OK");
    }

    #[test]
    fn indexer_keys_matching_pass() {
        let mut m = base_metadata();
        m.insert(INDEXER_HEAD_KEY.to_string(), Value::U32(INDEXER_HEAD));
        m.insert(
            INDEXER_HEAD_DIM_KEY.to_string(),
            Value::U32(INDEXER_HEAD_DIM),
        );
        m.insert(INDEXER_TOP_K_KEY.to_string(), Value::U32(INDEXER_TOP_K));
        ModelConfig::from_metadata(&m).expect("matching indexer constants pass");
    }

    #[test]
    fn indexer_head_mismatch_errors() {
        let mut m = base_metadata();
        m.insert(INDEXER_HEAD_KEY.to_string(), Value::U32(32));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("indexer.head_count") && msg.contains("32"),
            "got: {msg}",
        );
    }

    #[test]
    fn indexer_top_k_mismatch_errors() {
        let mut m = base_metadata();
        m.insert(INDEXER_TOP_K_KEY.to_string(), Value::U32(256));
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(err.to_string().contains("indexer.top_k"), "got: {err}");
    }

    #[test]
    fn indexer_key_wrong_type_errors() {
        let mut m = base_metadata();
        m.insert(
            INDEXER_HEAD_KEY.to_string(),
            Value::String("nope".to_string()),
        );
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(err.to_string().contains("Type mismatch"), "got: {err}");
    }
}
