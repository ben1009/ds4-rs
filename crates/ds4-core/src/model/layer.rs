//! Per-layer borrowed weight views.
//!
//! See rfcs/0002-forward-pass.md §2. A `LayerWeights` is built once at engine
//! init and holds nothing but borrowed slices/views into the mmap'd GGUF.
//! It is cheap to copy (all refs / small structs).

use anyhow::Result;

use crate::{model::WeightMap, ops::matmul::WeightView};

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
    /// IQ2_XXS, IQ4_K, or similar — dtype varies by model variant.
    /// Stored as raw bytes until the quant kernels land (PRs 5–6).
    pub ffn_gate_exps: &'a [u8],
    /// Same dtype as `ffn_gate_exps`.
    pub ffn_up_exps: &'a [u8],
    /// Q2_K, Q4_K, or similar.
    pub ffn_down_exps: &'a [u8],

    // --- Hash routing (layers 0–2 only) -----------------------------------
    pub ffn_gate_tid2eid: Option<&'a [i32]>,
}

impl<'a> LayerWeights<'a> {
    /// Load all tensors for layer `il` from the weight map.
    pub fn from_map(map: &'a WeightMap, il: u32) -> Result<Self> {
        use crate::model::kv_cache::KV_LATENT_DIM;

        let prefix = format!("blk.{il}.");
        let n_embd = map.config.n_embd as usize;
        let n_head = map.config.n_head as usize;
        let n_hc = map.config.n_hc as usize;
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
        let tid2eid_name = format!("{prefix}ffn_gate_tid2eid.weight");
        let ffn_gate_tid2eid = if map.tensor_info(&tid2eid_name).is_some() {
            let expect_elems = 6usize
                .checked_mul(map.config.n_vocab as usize)
                .ok_or_else(|| anyhow::anyhow!("{tid2eid_name}: tid2eid size overflow"))?;
            Some(map.i32_1d(&tid2eid_name, expect_elems)?)
        } else {
            None
        };

        // Bind attn_q_a first so we can size attn_q_a_norm from its out_features.
        let attn_q_a = q8_0("attn_q_a.weight")?;
        let q_a_rank = attn_q_a.out_features();

        Ok(Self {
            attn_norm: f32_1d("attn_norm.weight", n_embd)?,
            attn_q_a,
            attn_q_a_norm: f32_1d("attn_q_a_norm.weight", q_a_rank)?,
            attn_q_b: q8_0("attn_q_b.weight")?,
            attn_kv: q8_0("attn_kv.weight")?,
            attn_kv_a_norm: f32_1d("attn_kv_a_norm.weight", KV_LATENT_DIM)?,
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

            // For the routed experts, the GGUF dtype can be IQ2_XXS, IQ4_K, Q2_K, Q4_K,
            // etc. The WeightMap accessor will validate the actual dtype when these
            // variants are added to WeightView. For now we store raw bytes and a
            // typed view will be built once the quant kernels land (PRs 5–6).
            ffn_gate_exps: map.tensor_bytes(&format!("{prefix}ffn_gate_exps.weight"))?,
            ffn_up_exps: map.tensor_bytes(&format!("{prefix}ffn_up_exps.weight"))?,
            ffn_down_exps: map.tensor_bytes(&format!("{prefix}ffn_down_exps.weight"))?,

            ffn_gate_tid2eid,
        })
    }
}
