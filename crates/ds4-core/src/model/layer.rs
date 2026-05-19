//! Per-layer borrowed weight views.
//!
//! See rfcs/0002-forward-pass.md §2. A `LayerWeights` is built once at engine
//! init and holds nothing but borrowed slices/views into the mmap'd GGUF.
//! It is cheap to copy (all refs / small structs).

use anyhow::Result;

use crate::{
    config::layer_compress_ratio,
    model::{WeightMap, kv_cache::HEAD_DIM},
    ops::matmul::WeightView,
};

/// Streaming-compressor weights for one layer, only present when
/// `layer_compress_ratio(il) != 0`. Mirrors the four `attn_compressor_*`
/// tensors in antirez/ds4 ds4.c (lines 2370-2405, 2675-2680).
#[derive(Clone, Copy, Debug)]
pub struct CompressorLayerWeights<'a> {
    /// `[N_EMBD, width]` F16. Multiplied against the post-RMSNorm activation
    /// to produce the per-token KV state row.
    pub kv: WeightView<'a>,
    /// `[N_EMBD, width]` F16. Same shape as `kv`; gives the score row.
    pub gate: WeightView<'a>,
    /// Positional bias added to the gate output, shape `[width, ratio]`.
    /// Loaded as an F16 view; the kernel dequants on access.
    pub ape: WeightView<'a>,
    /// `[HEAD_DIM]` F32. RMSNorm weight applied after pooling and before
    /// the long-context RoPE rotation.
    pub norm: &'a [f32],
}

/// Borrowed views for one transformer layer.
///
/// Naming sticks to the antirez/ds4 tensor conventions so grepping cross-repo
/// works.
#[derive(Clone, Copy, Debug)]
pub struct LayerWeights<'a> {
    // --- Attention norms & projections ------------------------------------
    pub attn_norm: &'a [f32],
    pub attn_q_a: WeightView<'a>,
    pub attn_q_a_norm: &'a [f32],
    pub attn_q_b: WeightView<'a>,
    pub attn_kv: WeightView<'a>,
    pub attn_kv_a_norm: &'a [f32],
    pub attn_sinks: &'a [f32],
    pub attn_output_a: WeightView<'a>,
    pub attn_output_b: WeightView<'a>,

    // --- Hyper-connection control -----------------------------------------
    pub hc_attn_fn: WeightView<'a>,
    pub hc_attn_scale: &'a [f32],
    pub hc_attn_base: &'a [f32],
    pub hc_ffn_fn: WeightView<'a>,
    pub hc_ffn_scale: &'a [f32],
    pub hc_ffn_base: &'a [f32],

    // --- FFN norms & shared expert ----------------------------------------
    pub ffn_norm: &'a [f32],
    pub ffn_gate_inp: WeightView<'a>,
    pub ffn_gate_shexp: WeightView<'a>,
    pub ffn_up_shexp: WeightView<'a>,
    pub ffn_down_shexp: WeightView<'a>,

    // --- Routed experts ---------------------------------------------------
    /// IQ2_XXS, IQ4_XS, or similar — dtype varies by model variant.
    pub ffn_gate_exps: WeightView<'a>,
    /// Same dtype as `ffn_gate_exps`.
    pub ffn_up_exps: WeightView<'a>,
    /// Q2_K, Q4_K, IQ4_NL, or similar.
    pub ffn_down_exps: WeightView<'a>,

    // --- Hash routing (layers 0–2 only) -----------------------------------
    pub ffn_gate_tid2eid: Option<&'a [i32]>,

    // --- Top-k routing (layers 3+ only) -----------------------------------
    /// Per-expert bias added to router probs only for top-k *selection*.
    /// The unbiased probs are still used for the per-expert weight.
    pub ffn_exp_probs_b: Option<&'a [f32]>,

    // --- Streaming compressor (layers with ratio != 0) --------------------
    /// `Some` for layers `2..n_layer` with `layer_compress_ratio(il) != 0`.
    /// Dense layers (`il < 2`) hold `None`. PR 4 will plumb these into the
    /// mixed attention path; for this PR they're only consumed by the
    /// streaming compressor's emit step.
    pub compressor: Option<CompressorLayerWeights<'a>>,
}

impl<'a> LayerWeights<'a> {
    /// Load all tensors for layer `il` from the weight map.
    pub fn from_map(map: &'a WeightMap, il: u32) -> Result<Self> {
        use crate::model::kv_cache::{K_PE_DIM, KV_LATENT_DIM};

        let prefix = format!("blk.{il}.");
        let n_embd = map.config.n_embd as usize;
        let n_head = map.config.n_head as usize;
        let n_hc = map.config.n_hc as usize;
        let q_lora_rank = map.config.q_lora_rank as usize;
        let _n_ff = map.config.n_ff as usize;
        let _n_expert = map.config.n_expert as usize;

        // mix tail layout: [pre (n_hc) | post (n_hc) | comb (n_hc * n_hc)]
        let hc_base_dim = n_hc
            .checked_mul(
                n_hc.checked_add(2).ok_or_else(|| {
                    anyhow::anyhow!("LayerWeights: hc base dim (n_hc + 2) overflow")
                })?,
            )
            .ok_or_else(|| anyhow::anyhow!("LayerWeights: hc base dim overflow"))?;

        let f32_1d = |name: &str, n: usize| map.f32_1d(&format!("{prefix}{name}"), n);
        let q8_0 = |name: &str| map.q8_0(&format!("{prefix}{name}"));
        let f16 = |name: &str| map.f16(&format!("{prefix}{name}"));

        // Hash routing table is optional (only present for layers 0–2).
        // Layout is `[n_vocab, n_expert_used]` (per-token row of expert
        // ids, with `n_expert_used` as the inner stride), so the row
        // stride here must match the forward-pass indexer
        // (`tid2eid[token * n_expert_used + slot]`).
        let tid2eid_name = format!("{prefix}ffn_gate_tid2eid.weight");
        let ffn_gate_tid2eid = if map.tensor_info(&tid2eid_name).is_some() {
            let n_expert_used = map.config.n_expert_used as usize;
            let expect_elems = n_expert_used
                .checked_mul(map.config.n_vocab as usize)
                .ok_or_else(|| anyhow::anyhow!("{tid2eid_name}: tid2eid size overflow"))?;
            Some(map.i32_1d(&tid2eid_name, expect_elems)?)
        } else {
            None
        };

        // Top-k bias is optional (only present for layers 3+, where the GGUF
        // exporter writes it as `blk.{il}.exp_probs_b.bias`).
        let probs_b_name = format!("{prefix}exp_probs_b.bias");
        let ffn_exp_probs_b = if map.tensor_info(&probs_b_name).is_some() {
            Some(map.f32_1d(&probs_b_name, map.config.n_expert as usize)?)
        } else {
            None
        };

        // attn_q_a is a Q8_0 down-projection from n_embd → q_lora_rank.
        // The norm sits on the LoRA-rank side, not n_embd.
        let attn_q_a = q8_0("attn_q_a.weight")?;

        // Streaming compressor (only for layers with ratio != 0). Shapes
        // match the C reference (ds4.c:2370-2405):
        //   * kv   F16 [N_EMBD, width]
        //   * gate F16 [N_EMBD, width]
        //   * ape  F16 [width, ratio]
        //   * norm F32 [HEAD_DIM]
        // where width = coff * HEAD_DIM, coff = (ratio == 4) ? 2 : 1.
        let ratio = layer_compress_ratio(il);
        let compressor = if ratio == 0 {
            None
        } else {
            let coff = if ratio == 4 { 2 } else { 1 };
            let width = coff * HEAD_DIM;
            let kv = f16("attn_compressor_kv.weight")?;
            if kv.in_features() != n_embd || kv.out_features() != width {
                anyhow::bail!(
                    "{prefix}attn_compressor_kv.weight: expected [{n_embd}, {width}], got [{}, {}]",
                    kv.in_features(),
                    kv.out_features(),
                );
            }
            let gate = f16("attn_compressor_gate.weight")?;
            if gate.in_features() != n_embd || gate.out_features() != width {
                anyhow::bail!(
                    "{prefix}attn_compressor_gate.weight: expected [{n_embd}, {width}], got [{}, {}]",
                    gate.in_features(),
                    gate.out_features(),
                );
            }
            let ape = f16("attn_compressor_ape.weight")?;
            if ape.in_features() != width || ape.out_features() != ratio as usize {
                anyhow::bail!(
                    "{prefix}attn_compressor_ape.weight: expected [{width}, {ratio}], got [{}, {}]",
                    ape.in_features(),
                    ape.out_features(),
                );
            }
            let norm = f32_1d("attn_compressor_norm.weight", HEAD_DIM)?;
            Some(CompressorLayerWeights {
                kv,
                gate,
                ape,
                norm,
            })
        };

        Ok(Self {
            attn_norm: f32_1d("attn_norm.weight", n_embd)?,
            attn_q_a,
            attn_q_a_norm: f32_1d("attn_q_a_norm.weight", q_lora_rank)?,
            attn_q_b: q8_0("attn_q_b.weight")?,
            attn_kv: q8_0("attn_kv.weight")?,
            // attn_kv_a_norm spans the full N_HEAD_DIM (= KV_LATENT_DIM + K_PE_DIM = 512)
            // row produced by attn_kv. The C reference (`rms_norm_weight(kv, raw,
            // attn_kv_a_norm, DS4_N_HEAD_DIM, DS4_RMS_EPS)`) normalises the entire
            // 512-dim row with a 512-wide weight; the split into latent / k_pe
            // happens *after* the norm, when RoPE rotates the last 64 dims.
            attn_kv_a_norm: f32_1d("attn_kv_a_norm.weight", KV_LATENT_DIM + K_PE_DIM)?,
            attn_sinks: f32_1d("attn_sinks.weight", n_head)?,
            attn_output_a: q8_0("attn_output_a.weight")?,
            attn_output_b: q8_0("attn_output_b.weight")?,

            hc_attn_fn: f16("hc_attn_fn.weight")?,
            hc_attn_scale: f32_1d("hc_attn_scale.weight", 3)?,
            hc_attn_base: f32_1d("hc_attn_base.weight", hc_base_dim)?,
            hc_ffn_fn: f16("hc_ffn_fn.weight")?,
            hc_ffn_scale: f32_1d("hc_ffn_scale.weight", 3)?,
            hc_ffn_base: f32_1d("hc_ffn_base.weight", hc_base_dim)?,

            ffn_norm: f32_1d("ffn_norm.weight", n_embd)?,
            ffn_gate_inp: f16("ffn_gate_inp.weight")?,
            ffn_gate_shexp: q8_0("ffn_gate_shexp.weight")?,
            ffn_up_shexp: q8_0("ffn_up_shexp.weight")?,
            ffn_down_shexp: q8_0("ffn_down_shexp.weight")?,

            // Routed experts: dtype varies by model variant (e.g. IQ2_XXS for gate/up
            // and Q2_K for down in the 16B model). `quant_weight` auto-dispatches by
            // the actual GGML type recorded in the GGUF tensor metadata.
            ffn_gate_exps: map.quant_weight(&format!("{prefix}ffn_gate_exps.weight"))?,
            ffn_up_exps: map.quant_weight(&format!("{prefix}ffn_up_exps.weight"))?,
            ffn_down_exps: map.quant_weight(&format!("{prefix}ffn_down_exps.weight"))?,

            ffn_gate_tid2eid,
            ffn_exp_probs_b,
            compressor,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn write_minimal_gguf(path: &std::path::Path) {
        let mut buf: Vec<u8> = Vec::new();
        let u32le = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        let u64le = |buf: &mut Vec<u8>, v: u64| buf.extend_from_slice(&v.to_le_bytes());
        let strle = |buf: &mut Vec<u8>, s: &str| {
            u64le(buf, s.len() as u64);
            buf.extend_from_slice(s.as_bytes());
        };
        let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            strle(buf, k);
            u32le(buf, 4);
            u32le(buf, v);
        };
        let kv_arr_string = |buf: &mut Vec<u8>, k: &str, values: &[String]| {
            strle(buf, k);
            u32le(buf, 9);
            u32le(buf, 8);
            u64le(buf, values.len() as u64);
            for v in values {
                strle(buf, v);
            }
        };
        u32le(&mut buf, crate::gguf::GGUF_MAGIC);
        u32le(&mut buf, 3);
        u64le(&mut buf, 0);
        let tokens: Vec<String> = crate::tokenizer::synthetic_byte_tokens();
        u64le(&mut buf, 8);
        kv_u32(&mut buf, "deepseek4.vocab_size", 256);
        kv_u32(&mut buf, "deepseek4.embedding_length", 16);
        kv_u32(&mut buf, "deepseek4.attention.head_count", 4);
        kv_u32(&mut buf, "deepseek4.attention.head_count_kv", 4);
        kv_u32(&mut buf, "deepseek4.block_count", 2);
        kv_u32(&mut buf, "deepseek4.expert_feed_forward_length", 32);
        kv_u32(&mut buf, "deepseek4.attention.q_lora_rank", 8);
        kv_arr_string(&mut buf, "tokenizer.ggml.tokens", &tokens);
        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    fn open_map() -> WeightMap {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ds4-layer-test-{}-{}.gguf",
            std::process::id(),
            seq,
        ));
        write_minimal_gguf(&path);
        let map = WeightMap::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        map
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn from_map_missing_tensors_errors() {
        let m = open_map();
        let err = LayerWeights::from_map(&m, 0).unwrap_err();
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn from_map_higher_layer_also_errors() {
        let m = open_map();
        assert!(LayerWeights::from_map(&m, 7).is_err());
    }
}
