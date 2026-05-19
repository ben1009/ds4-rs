//! Forward pass orchestration.
//!
//! See rfcs/0002-forward-pass.md §2 / §3. Phase 1 implements a CPU reference
//! forward pass: single-threaded, f32 activations, sliding-window attention
//! (no compressor/indexer), no FP8 KV round-trip.

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::{
    config::ModelConfig,
    engine::Engine,
    gguf::GgmlType,
    model::{WeightMap, kv_cache::KvCache, layer::LayerWeights},
    ops::{
        hc::{hc_control_split, hc_from_plain_embedding, hc_post, hc_weighted_sum},
        matmul::{expert_subview, matmul_row},
        norm::{rms_norm, rms_norm_no_weight},
        rope::{apply_rope, apply_rope_inverse},
        softmax::sqrt_softplus,
        swiglu::{sigmoid_stable, silu, swiglu},
    },
    quant::q8_0,
    session::Session,
};

/// Run a single decode step: compute logits for the token at `session.pos()`.
///
/// **Contract:** `session.pos()` must be the index of the token to evaluate
/// (i.e. `< session.tokens().len()`), and the prior `pos` tokens must already
/// have been written into the KV cache. The wiring in [`Session::eval_token`]
/// is responsible for upholding this — `forward_decode` only validates.
pub fn forward_decode(session: &mut Session, engine: &Arc<Engine>) -> Result<Vec<f32>> {
    let model = &engine.weights;
    let config = &engine.config;
    let pos = session.pos() as usize;
    if pos >= session.tokens().len() {
        bail!(
            "forward_decode: pos {pos} out of range — session has {} tokens",
            session.tokens().len()
        );
    }

    // --- Token embedding ---------------------------------------------------
    let token = session.tokens()[pos];
    let mut plain = vec![0.0f32; config.n_embd as usize];
    embed_token(model, token, &mut plain)?;

    // --- Copy embedding into HC streams ------------------------------------
    let n_hc = config.n_hc as usize;
    let n_embd = config.n_embd as usize;
    let hc_dim = n_hc
        .checked_mul(n_embd)
        .ok_or_else(|| anyhow::anyhow!("HC dimension overflow"))?;
    let mut residual_hc = vec![0.0f32; hc_dim];
    hc_from_plain_embedding(&mut residual_hc, &plain, n_embd, n_hc);

    // --- Per-layer loop ----------------------------------------------------
    // Layers are built once at engine open. For Phase 1 we rebuild them
    // on every call; this is fine for a reference implementation.
    for il in 0..config.n_layer {
        let layer = LayerWeights::from_map(model, il)?;
        let mut attn_out = vec![0.0f32; n_embd];
        let (attn_post, attn_comb) = layer_attention_decode(
            &mut attn_out,
            engine,
            &layer,
            &residual_hc,
            session.kv_cache_mut(),
            il as usize,
            pos,
        )?;

        // HC post for attention.
        let mut after_attn_hc = vec![0.0f32; hc_dim];
        hc_post(
            &mut after_attn_hc,
            &attn_out,
            &residual_hc,
            &attn_post,
            &attn_comb,
            n_embd,
            n_hc,
        );

        // FFN sublayer.
        let mut ffn_out = vec![0.0f32; n_embd];
        let (ffn_post, ffn_comb) = layer_ffn_decode(
            &mut ffn_out,
            config,
            &layer,
            &after_attn_hc,
            il as usize,
            token,
        )?;

        // HC post for FFN.
        hc_post(
            &mut residual_hc,
            &ffn_out,
            &after_attn_hc,
            &ffn_post,
            &ffn_comb,
            n_embd,
            n_hc,
        );
    }

    // --- Output head -------------------------------------------------------
    let mut logits = vec![0.0f32; config.n_vocab as usize];
    output_head(model, config, &residual_hc, &mut logits)?;

    Ok(logits)
}

// =========================================================================
// Embedding
// =========================================================================

fn embed_token(model: &WeightMap, token: u32, out: &mut [f32]) -> Result<()> {
    let info = model
        .tensor_info("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;
    if info.dtype != GgmlType::F16 {
        bail!("token_embd.weight: expected F16, got {:?}", info.dtype);
    }
    let n_embd = info.dims[0] as usize;
    let n_vocab = info.dims[1] as usize;
    assert_eq!(out.len(), n_embd, "embed_token: out len mismatch");
    if token as usize >= n_vocab {
        bail!("token id {token} >= vocab size {n_vocab}");
    }

    let bytes = model.tensor_bytes("token_embd.weight")?;
    let row_size = n_embd
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("embed_token: row size overflow"))?;
    let row_off = (token as usize)
        .checked_mul(row_size)
        .ok_or_else(|| anyhow::anyhow!("embed_token: row offset overflow"))?;
    let row_end = row_off
        .checked_add(row_size)
        .ok_or_else(|| anyhow::anyhow!("embed_token: row end overflow"))?;
    let row = &bytes
        .get(row_off..row_end)
        .ok_or_else(|| anyhow::anyhow!("embed_token: row out of bounds"))?;
    for (o, chunk) in out.iter_mut().zip(row.chunks_exact(2)) {
        *o = q8_0::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(())
}

// =========================================================================
// Attention (decode)
// =========================================================================

fn layer_attention_decode(
    out: &mut [f32],
    engine: &Engine,
    layer: &LayerWeights<'_>,
    residual_hc: &[f32],
    kv_cache: &mut KvCache,
    il: usize,
    pos: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let config = &engine.config;
    let n_embd = config.n_embd as usize;
    let n_hc = config.n_hc as usize;
    let n_head = config.n_head as usize;
    let n_head_dim = config.head_dim as usize;
    let _n_rot = 64usize;
    let q_dim = n_head * n_head_dim;

    // HC pre: project control, Sinkhorn split, weighted sum.
    let mut attn_cur = vec![0.0f32; n_embd];
    let mut post = vec![0.0f32; n_hc];
    let mut comb = vec![0.0f32; n_hc * n_hc];
    hc_pre(
        &engine.weights,
        layer,
        residual_hc,
        &mut attn_cur,
        &mut post,
        &mut comb,
        n_embd,
        n_hc,
    )?;

    // Attention RMSNorm.
    let mut attn_norm = vec![0.0f32; n_embd];
    rms_norm(&attn_cur, layer.attn_norm, 1e-6, &mut attn_norm);

    // Q projection (low-rank).
    let mut q = vec![0.0f32; q_dim];
    q_projection_decode(
        &engine.weights,
        layer,
        &attn_norm,
        &mut q,
        n_head,
        n_head_dim,
    )?;

    // KV projection (matches antirez/ds4 ds4.c `layer_kv_projection_normed_one`).
    // The attn_kv matmul produces a single DS4_N_HEAD_DIM = 512 wide row, which
    // is RMSNorm'd as one vector with the 512-wide attn_kv_a_norm weight, then
    // RoPE'd in place over the trailing K_PE_DIM tail. The merged row is pushed
    // into the per-layer raw ring buffer; the ring enforces the SWA window
    // itself (oldest-evicting on overflow).
    use crate::model::kv_cache::{HEAD_DIM, KV_LATENT_DIM};
    let mut kv_raw = vec![0.0f32; HEAD_DIM];
    matmul_row(layer.attn_kv, &attn_norm, &mut kv_raw);

    let mut kv_normed = vec![0.0f32; HEAD_DIM];
    rms_norm(&kv_raw, layer.attn_kv_a_norm, 1e-6, &mut kv_normed);

    apply_rope(&mut kv_normed[KV_LATENT_DIM..], pos, &engine.rope_freqs);

    // RoPE per head on Q (uses precomputed frequency cache from Engine).
    for h in 0..n_head {
        let qh = &mut q[h * n_head_dim..(h + 1) * n_head_dim];
        apply_rope(qh, pos, &engine.rope_freqs);
    }

    kv_cache.layer_mut(il).push(&kv_normed);

    // Attention over cached KV rows.
    let mut heads = vec![0.0f32; q_dim];
    attention_rows(&mut heads, layer, &q, kv_cache, il, n_head, n_head_dim)?;

    // Inverse RoPE per head on attention output before grouped projection.
    for h in 0..n_head {
        let oh = &mut heads[h * n_head_dim..(h + 1) * n_head_dim];
        apply_rope_inverse(oh, pos, &engine.rope_freqs);
    }

    // Grouped output projection.
    grouped_out_decode(&engine.weights, layer, &heads, out, n_head, n_head_dim)?;

    Ok((post, comb))
}

#[allow(clippy::too_many_arguments)]
fn hc_pre(
    _model: &WeightMap,
    layer: &LayerWeights<'_>,
    residual_hc: &[f32],
    out: &mut [f32],
    post: &mut [f32],
    comb: &mut [f32],
    n_embd: usize,
    n_hc: usize,
) -> Result<()> {
    // 1. RMSNorm no weight on residual_hc.
    let mut flat = vec![0.0f32; n_hc * n_embd];
    rms_norm_no_weight(residual_hc, 1e-6, &mut flat);

    // 2. Control projection (F16 matmul).
    // mix shape: [2*n_hc + n_hc*n_hc]
    let mut mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
    matmul_row(layer.hc_attn_fn, &flat, &mut mix);

    // 3. Sinkhorn split: produce pre-weights (and post/comb for caller).
    let mut pre = vec![0.0f32; n_hc];
    hc_control_split(
        &mix,
        layer.hc_attn_scale,
        layer.hc_attn_base,
        &mut pre,
        post,
        comb,
        n_hc,
        20,
        1e-6,
    );

    // 4. Weighted sum of streams into the per-sublayer input.
    hc_weighted_sum(residual_hc, &pre, out, n_embd, n_hc);

    Ok(())
}

fn hc_pre_ffn(
    layer: &LayerWeights<'_>,
    residual_hc: &[f32],
    out: &mut [f32],
    post: &mut [f32],
    comb: &mut [f32],
    n_embd: usize,
    n_hc: usize,
) -> Result<()> {
    let mut flat = vec![0.0f32; n_hc * n_embd];
    rms_norm_no_weight(residual_hc, 1e-6, &mut flat);
    let mut mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
    matmul_row(layer.hc_ffn_fn, &flat, &mut mix);
    let mut pre = vec![0.0f32; n_hc];
    hc_control_split(
        &mix,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        &mut pre,
        post,
        comb,
        n_hc,
        20,
        1e-6,
    );
    hc_weighted_sum(residual_hc, &pre, out, n_embd, n_hc);
    Ok(())
}

fn q_projection_decode(
    _model: &WeightMap,
    layer: &LayerWeights<'_>,
    norm: &[f32],
    q: &mut [f32],
    n_head: usize,
    head_dim: usize,
) -> Result<()> {
    // Q = attn_q_b(RMSNorm(attn_q_a(norm)))
    let q_a_rank = layer.attn_q_a.out_features();
    let mut qr = vec![0.0f32; q_a_rank];
    matmul_row(layer.attn_q_a, norm, &mut qr);

    let mut qr_norm = vec![0.0f32; q_a_rank];
    rms_norm(&qr, layer.attn_q_a_norm, 1e-6, &mut qr_norm);

    matmul_row(layer.attn_q_b, &qr_norm, q);

    // Per-head RMSNorm (no weight).
    for h in 0..n_head {
        let head = &mut q[h * head_dim..(h + 1) * head_dim];
        let scale = rms_scale(head, 1e-6);
        for v in head.iter_mut() {
            *v *= scale;
        }
    }

    Ok(())
}

fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let n = x.len();
    if n == 0 {
        return 1.0 / eps.sqrt();
    }
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let mean = sum_sq / n as f32;
    1.0 / (mean + eps).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn attention_rows(
    out_heads: &mut [f32],
    layer: &LayerWeights<'_>,
    q: &[f32],
    kv_cache: &KvCache,
    il: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<()> {
    use crate::model::kv_cache::HEAD_DIM;

    // DS4 MLA: the cached row is a single HEAD_DIM = 512 vector, used as
    // *both* K (for scoring) and V (for the weighted sum). `attn_kv_a_norm`
    // already normalised the full row before RoPE rotated only the last 64
    // dims. Q · K and Σ wᵢ · V both run over the same 512-dim row, matching
    // `layer_attention_rows_one` in antirez/ds4 ds4.c.
    assert_eq!(head_dim, HEAD_DIM);

    let kv_window = kv_cache.layer(il).rows();
    attention_rows_inner(out_heads, layer.attn_sinks, q, kv_window, n_head);
    Ok(())
}

/// Sink-aware attention over the cached KV window. Shared math between the
/// per-layer decode call and unit tests — no `LayerWeights` dependency so the
/// math can be exercised against hand-crafted inputs.
///
/// Buffer contract:
/// * `kv_window` is `[window_len, HEAD_DIM]` row-major.
/// * `q`, `out_heads` are `[n_head, HEAD_DIM]`.
/// * `sinks` has one logit per head.
fn attention_rows_inner(
    out_heads: &mut [f32],
    sinks: &[f32],
    q: &[f32],
    kv_window: &[f32],
    n_head: usize,
) {
    use crate::model::kv_cache::HEAD_DIM;

    let head_dim = HEAD_DIM;
    let kq_scale = 1.0 / (head_dim as f32).sqrt();
    assert_eq!(out_heads.len(), n_head * head_dim);
    assert_eq!(q.len(), n_head * head_dim);
    assert_eq!(sinks.len(), n_head);
    assert!(kv_window.len().is_multiple_of(head_dim));
    let window_len = kv_window.len() / head_dim;

    let mut scores = vec![0.0f32; window_len];

    for h in 0..n_head {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let oh = &mut out_heads[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);

        let mut max_score = sinks[h];

        for (i, kv) in kv_window.chunks_exact(head_dim).enumerate() {
            let dot: f32 = qh.iter().zip(kv.iter()).map(|(&q, &k)| q * k).sum();
            let score = dot * kq_scale;
            scores[i] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut denom = (sinks[h] - max_score).exp();
        for (i, kv) in kv_window.chunks_exact(head_dim).enumerate() {
            let weight = (scores[i] - max_score).exp();
            denom += weight;
            for (o, &v) in oh.iter_mut().zip(kv.iter()) {
                *o += v * weight;
            }
        }

        let inv = 1.0 / (denom + 1e-9);
        for v in oh.iter_mut() {
            *v *= inv;
        }
    }
}

fn grouped_out_decode(
    _model: &WeightMap,
    layer: &LayerWeights<'_>,
    heads: &[f32],
    out: &mut [f32],
    n_head: usize,
    head_dim: usize,
) -> Result<()> {
    let n_groups = 8usize;
    let group_heads = n_head / n_groups;
    let group_dim = head_dim
        .checked_mul(group_heads)
        .ok_or_else(|| anyhow::anyhow!("grouped_out_decode: group_dim overflow"))?;
    let rank = 1024usize;

    let mut low = vec![
        0.0f32;
        n_groups
            .checked_mul(rank)
            .ok_or_else(|| anyhow::anyhow!("grouped_out_decode: low len overflow"))?
    ];

    // `attn_output_a` is logically n_groups stacked `(group_dim, rank)`
    // matrices along the out axis (matches `matvec_q8_0_grouped_rows` in
    // antirez/ds4 ds4.c — the loader correctly reads in=group_dim,
    // out=n_groups*rank). Slice the weight bytes per group so each
    // matmul_row sees its own (group_dim → rank) submatrix.
    let (bytes, total_in, total_out) = match layer.attn_output_a {
        crate::ops::matmul::WeightView::Q8_0 {
            bytes,
            in_features,
            out_features,
        } => (bytes, in_features, out_features),
        _ => bail!("grouped_out_decode: attn_output_a must be Q8_0"),
    };
    if total_in != group_dim {
        bail!("grouped_out_decode: attn_output_a in_features {total_in} != group_dim {group_dim}",);
    }
    let expected_out = n_groups
        .checked_mul(rank)
        .ok_or_else(|| anyhow::anyhow!("grouped_out_decode: expected_out overflow"))?;
    if total_out != expected_out {
        bail!(
            "grouped_out_decode: attn_output_a out_features {total_out} != n_groups*rank {expected_out}",
        );
    }
    let blocks_per_row = group_dim / q8_0::BLOCK_SIZE;
    let bytes_per_row = blocks_per_row
        .checked_mul(q8_0::BYTES_PER_BLOCK)
        .ok_or_else(|| anyhow::anyhow!("grouped_out_decode: bytes_per_row overflow"))?;
    let bytes_per_group = rank
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("grouped_out_decode: bytes_per_group overflow"))?;

    for (g, (group_input, group_output)) in heads
        .chunks_exact(group_dim)
        .zip(low.chunks_exact_mut(rank))
        .enumerate()
    {
        let group_view = crate::ops::matmul::WeightView::Q8_0 {
            bytes: &bytes[g * bytes_per_group..(g + 1) * bytes_per_group],
            in_features: group_dim,
            out_features: rank,
        };
        matmul_row(group_view, group_input, group_output);
    }

    matmul_row(layer.attn_output_b, &low, out);
    Ok(())
}

// =========================================================================
// FFN (decode)
// =========================================================================

fn layer_ffn_decode(
    out: &mut [f32],
    config: &ModelConfig,
    layer: &LayerWeights<'_>,
    residual_hc: &[f32],
    il: usize,
    token: u32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n_embd = config.n_embd as usize;
    let n_hc = config.n_hc as usize;
    let n_expert = config.n_expert as usize;
    let n_expert_used = config.n_expert_used as usize;

    // HC pre for FFN.
    let mut ffn_cur = vec![0.0f32; n_embd];
    let mut post = vec![0.0f32; n_hc];
    let mut comb = vec![0.0f32; n_hc * n_hc];
    hc_pre_ffn(
        layer,
        residual_hc,
        &mut ffn_cur,
        &mut post,
        &mut comb,
        n_embd,
        n_hc,
    )?;

    // FFN RMSNorm.
    let mut ffn_norm = vec![0.0f32; n_embd];
    rms_norm(&ffn_cur, layer.ffn_norm, 1e-6, &mut ffn_norm);

    // Shared expert.
    let mut shared_out = vec![0.0f32; n_embd];
    shared_expert_decode(layer, &ffn_norm, &mut shared_out)?;

    // Routed experts.
    let mut moe_out = vec![0.0f32; n_embd];
    if n_expert > 0 && n_expert_used > 0 {
        routed_moe_decode(
            layer,
            &ffn_norm,
            &mut moe_out,
            il,
            token,
            n_expert,
            n_expert_used,
        )?;
    }

    // Sum shared + routed.
    for i in 0..n_embd {
        out[i] = shared_out[i] + moe_out[i];
    }

    Ok((post, comb))
}

fn shared_expert_decode(layer: &LayerWeights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
    let hidden = layer.ffn_gate_shexp.out_features();

    let mut gate = vec![0.0f32; hidden];
    let mut up = vec![0.0f32; hidden];
    matmul_row(layer.ffn_gate_shexp, x, &mut gate);
    matmul_row(layer.ffn_up_shexp, x, &mut up);

    let mut mid = vec![0.0f32; hidden];
    swiglu(&gate, &up, &mut mid);

    matmul_row(layer.ffn_down_shexp, &mid, out);
    Ok(())
}

// =========================================================================
// Routed MoE (decode)
// =========================================================================
//
// Mirrors `layer_routed_moe_one` in antirez/ds4 ds4.c:
//
// 1. probs = sqrt(softplus(ffn_gate_inp @ x))    — F16 router matvec, per-elem
// 2. selected[6]:
//    * hash layers (`ffn_gate_tid2eid` present): selected[i] = tid2eid[token*6 + i]
//    * top-k layers: selected = top-k indices of (probs + ffn_exp_probs_b) (bias only steers
//      selection, not the per-expert weight)
// 3. weight[i] = probs[selected[i]]              — unbiased sum = max(sum(weight), 2^-14) weight[i]
//    = (weight[i] / sum) * 1.5         — DS4_EXPERT_WEIGHT_SCALE
// 4. for each selected expert eid with weight w: gate = ffn_gate_exps[eid] @ x up   =
//    ffn_up_exps[eid]   @ x clamp gate (positive side) and up (both sides) to ±10 mid  = silu(gate)
//    * up * w out += ffn_down_exps[eid] @ mid

const EXPERT_WEIGHT_SCALE: f32 = 1.5;
/// Floor for `sum(unbiased_weights)` before division. Matches `2^-14` (the
/// smallest normal f16) in the C reference; used to avoid divide-by-zero
/// when the router output collapses.
const EXPERT_WEIGHT_SUM_EPS: f32 = 1.0 / 16384.0;
/// SwiGLU clamp magnitude for routed experts. Matches `DS4_SWIGLU_CLAMP_EXP`.
const SWIGLU_CLAMP_EXP: f32 = 10.0;

fn routed_moe_decode(
    layer: &LayerWeights<'_>,
    x: &[f32],
    out: &mut [f32],
    _il: usize,
    token: u32,
    n_expert: usize,
    n_expert_used: usize,
) -> Result<()> {
    // Router logits → sqrt(softplus) per-element gating.
    let mut probs = vec![0.0f32; n_expert];
    matmul_row(layer.ffn_gate_inp, x, &mut probs);
    for p in probs.iter_mut() {
        *p = sqrt_softplus(*p);
    }

    let mut selected = vec![0usize; n_expert_used];
    if let Some(table) = layer.ffn_gate_tid2eid {
        // Hash routing — table is laid out [n_vocab, n_expert_used].
        let row_off = (token as usize)
            .checked_mul(n_expert_used)
            .ok_or_else(|| anyhow::anyhow!("routed_moe: hash router token offset overflow"))?;
        let row_end = row_off
            .checked_add(n_expert_used)
            .ok_or_else(|| anyhow::anyhow!("routed_moe: hash router row end overflow"))?;
        let row = table.get(row_off..row_end).ok_or_else(|| {
            anyhow::anyhow!(
                "routed_moe: hash table out of range for token {token} (need {row_end}, have {})",
                table.len(),
            )
        })?;
        for (slot, &eid) in row.iter().enumerate() {
            if eid < 0 || (eid as usize) >= n_expert {
                bail!("routed_moe: tid2eid[{token},{slot}] = {eid} not in 0..{n_expert}");
            }
            selected[slot] = eid as usize;
        }
    } else if let Some(bias) = layer.ffn_exp_probs_b {
        // Biased top-k. Bias only shifts the *selection*; the per-expert
        // weight still uses the unbiased prob, so we score against a
        // bias-shifted copy of `probs`.
        let mut selection = probs.clone();
        for (s, &b) in selection.iter_mut().zip(bias.iter()) {
            *s += b;
        }
        topk_indices_desc(&selection, n_expert_used, &mut selected);
    } else {
        // Unbiased top-k (no `exp_probs_b.bias` tensor): score directly
        // against `probs` without an extra clone.
        topk_indices_desc(&probs, n_expert_used, &mut selected);
    }

    // Per-expert weights from the unbiased probs.
    let mut weights = vec![0.0f32; n_expert_used];
    let mut sum = 0.0f32;
    for (&eid, w) in selected.iter().zip(weights.iter_mut()) {
        *w = probs[eid];
        sum += *w;
    }
    if sum < EXPERT_WEIGHT_SUM_EPS {
        sum = EXPERT_WEIGHT_SUM_EPS;
    }
    let scale = EXPERT_WEIGHT_SCALE / sum;
    for w in weights.iter_mut() {
        *w *= scale;
    }

    // Accumulate each selected expert.
    out.fill(0.0);
    let mut down = vec![0.0f32; out.len()];
    for (slot, &eid) in selected.iter().enumerate() {
        run_routed_expert(layer, eid, n_expert, x, weights[slot], &mut down)?;
        for (o, &d) in out.iter_mut().zip(down.iter()) {
            *o += d;
        }
    }

    Ok(())
}

fn run_routed_expert(
    layer: &LayerWeights<'_>,
    eid: usize,
    n_expert: usize,
    x: &[f32],
    weight: f32,
    out: &mut [f32],
) -> Result<()> {
    let gate_w = expert_subview(layer.ffn_gate_exps, eid, n_expert);
    let up_w = expert_subview(layer.ffn_up_exps, eid, n_expert);
    let down_w = expert_subview(layer.ffn_down_exps, eid, n_expert);

    let hidden = gate_w.out_features();
    let mut gate = vec![0.0f32; hidden];
    let mut up = vec![0.0f32; hidden];
    matmul_row(gate_w, x, &mut gate);
    matmul_row(up_w, x, &mut up);

    let mut mid = vec![0.0f32; hidden];
    for (m, (&g_val, &u_val)) in mid.iter_mut().zip(gate.iter().zip(up.iter())) {
        let g = g_val.min(SWIGLU_CLAMP_EXP);
        let u = u_val.clamp(-SWIGLU_CLAMP_EXP, SWIGLU_CLAMP_EXP);
        *m = silu(g) * u * weight;
    }

    matmul_row(down_w, &mid, out);
    Ok(())
}

/// Pick the indices of the top-`k` largest entries in `values`, in descending
/// order of value, into `out`. Ties resolve toward the lower index. `k` is
/// expected to be small (6 for DS4) so a linear scan per slot is cheaper than
/// a heap, and we track "already-taken" by scanning `out[..slot]` rather than
/// allocating a side `taken: Vec<bool>` buffer.
///
/// NaN safety: the comparison is `values[i] > best_v` with `best_v` seeded at
/// `f32::NEG_INFINITY`. `NaN > x` is always false in IEEE-754, so any NaN
/// inputs are skipped on every slot rather than poisoning the result. When
/// every remaining value is NaN or -inf — which happens running this engine
/// against a model file whose F16 weights contain NaN bit-patterns (e.g. the
/// shipped DS4-Flash q2-imatrix GGUF) — we fall back to the lowest still-
/// unselected index. That mirrors the C reference's behaviour: there
/// `best_i` is left at its sentinel when no value compares greater, and
/// downstream Q8_K activation pre-quant casts NaN-per-row to integer 0
/// regardless, so picking sequential indices in this degenerate case keeps
/// the forward path running with no further damage.
fn topk_indices_desc(values: &[f32], k: usize, out: &mut [usize]) {
    debug_assert!(k <= values.len());
    debug_assert_eq!(out.len(), k);
    for slot in 0..k {
        let mut best_i = usize::MAX;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in values.iter().enumerate() {
            if out[..slot].contains(&i) {
                continue;
            }
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        if best_i == usize::MAX {
            // All remaining values are NaN / -inf. Fall back to the lowest
            // unselected index so we keep emitting distinct slots and the
            // forward pass can continue.
            best_i = (0..values.len())
                .find(|i| !out[..slot].contains(i))
                .expect("topk_indices_desc: k > values.len() should be impossible");
        }
        out[slot] = best_i;
    }
}

// =========================================================================
// Output head
// =========================================================================

fn output_head(
    model: &WeightMap,
    config: &ModelConfig,
    residual_hc: &[f32],
    logits: &mut [f32],
) -> Result<()> {
    let n_embd = config.n_embd as usize;
    let n_hc = config.n_hc as usize;
    let hc_dim = n_hc
        .checked_mul(n_embd)
        .ok_or_else(|| anyhow::anyhow!("output_head: HC dim overflow"))?;

    // Learned HC reduction (matches `output_hc_head_one` in antirez/ds4 ds4.c):
    //   1. flat = rms_norm_no_weight(residual_hc, eps=1e-6)        // [hc_dim]
    //   2. pre  = output_hc_fn @ flat                              // [n_hc]
    //   3. w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + eps_hc
    //   4. out  = sum_h residual_hc[h] * w[h]                      // [n_embd]
    //
    // The bias/scale tensors are tiny F32 vectors:
    //   * output_hc_fn.weight    F16 shape [n_hc, hc_dim]
    //   * output_hc_scale.weight F32 scalar
    //   * output_hc_base.weight  F32 shape [n_hc]
    let mut flat = vec![0.0f32; hc_dim];
    rms_norm_no_weight(residual_hc, 1e-6, &mut flat);

    let hc_fn = model.f16("output_hc_fn.weight")?;
    let mut pre = vec![0.0f32; n_hc];
    matmul_row(hc_fn, &flat, &mut pre);

    let scale = model.f32_1d("output_hc_scale.weight", 1)?;
    let base = model.f32_1d("output_hc_base.weight", n_hc)?;
    let mut weights = vec![0.0f32; n_hc];
    output_hc_weights(&pre, scale[0], base, &mut weights);

    let mut plain = vec![0.0f32; n_embd];
    hc_weighted_sum(residual_hc, &weights, &mut plain, n_embd, n_hc);

    // Final RMSNorm.
    let mut norm = vec![0.0f32; n_embd];
    let output_norm = model.f32_1d("output_norm.weight", n_embd)?;
    rms_norm(&plain, output_norm, 1e-6, &mut norm);

    // Output projection.
    let output_weight = model.q8_0("output.weight")?;
    matmul_row(output_weight, &norm, logits);

    Ok(())
}

/// Per-stream weight epsilon for the output HC reduction.
/// Matches `DS4_HC_EPS` in antirez/ds4 ds4.c (= 1e-6f).
const HC_EPS: f32 = 1.0e-6;

/// Compute per-stream output HC weights:
/// `weights[i] = sigmoid_stable(pre[i] * scale + base[i]) + HC_EPS`.
///
/// Pure math, factored out for testing — matches `output_hc_head_one` in
/// antirez/ds4 ds4.c.
fn output_hc_weights(pre: &[f32], scale: f32, base: &[f32], weights: &mut [f32]) {
    assert_eq!(pre.len(), weights.len());
    assert_eq!(base.len(), weights.len());
    for (w, (&p, &b)) in weights.iter_mut().zip(pre.iter().zip(base.iter())) {
        *w = sigmoid_stable(p * scale + b) + HC_EPS;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_scale_unit_vector() {
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let s = rms_scale(&x, 1e-12);
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rms_scale_zero_vector_is_eps_bound() {
        let x = [0.0f32; 8];
        let s = rms_scale(&x, 1e-6);
        let expected = 1.0 / 1e-6f32.sqrt();
        assert!((s - expected).abs() / expected < 1e-3);
    }

    #[test]
    fn rms_scale_empty_returns_eps_inverse() {
        let s = rms_scale(&[], 1e-6);
        let expected = 1.0 / 1e-6f32.sqrt();
        assert!((s - expected).abs() / expected < 1e-3);
    }

    #[test]
    fn rms_scale_matches_inverse_rms() {
        let x = [3.0f32, 4.0, 0.0, 0.0];
        // mean square = 25/4 = 6.25, sqrt = 2.5, scale = 1/sqrt(6.25 + eps) ~= 0.4
        let s = rms_scale(&x, 1e-12);
        assert!((s - 0.4).abs() < 1e-4);
    }

    #[test]
    fn rms_scale_scale_invariance() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let xs: Vec<f32> = x.iter().map(|v| v * 10.0).collect();
        let a = rms_scale(&x, 1e-12);
        let b = rms_scale(&xs, 1e-12);
        // Scaling input by k should scale rms_scale by 1/k.
        assert!((a - b * 10.0).abs() / a < 1e-3);
    }

    #[test]
    fn rms_scale_negative_inputs_match_positive() {
        let p = rms_scale(&[1.0, 2.0, 3.0], 1e-12);
        let n = rms_scale(&[-1.0, -2.0, -3.0], 1e-12);
        assert!((p - n).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Routed MoE helpers
    // -----------------------------------------------------------------------

    #[test]
    fn topk_returns_descending_indices() {
        let v = [0.1f32, 0.9, 0.4, 0.7, 0.2, 0.8];
        let mut out = [0usize; 3];
        topk_indices_desc(&v, 3, &mut out);
        // Descending order: 0.9 (idx 1), 0.8 (idx 5), 0.7 (idx 3).
        assert_eq!(out, [1, 5, 3]);
    }

    #[test]
    fn topk_breaks_ties_toward_lower_index() {
        let v = [0.5f32, 0.5, 0.5, 0.5];
        let mut out = [0usize; 2];
        topk_indices_desc(&v, 2, &mut out);
        // First scan picks index 0 (strict `>`), second picks index 1.
        assert_eq!(out, [0, 1]);
    }

    #[test]
    fn topk_full_length_is_full_sort() {
        let v = [3.0f32, 1.0, 4.0, 1.5, 9.0, 2.0];
        let mut out = [0usize; 6];
        topk_indices_desc(&v, 6, &mut out);
        // Sorted descending: 9 (4), 4 (2), 3 (0), 2 (5), 1.5 (3), 1 (1).
        assert_eq!(out, [4, 2, 0, 5, 3, 1]);
    }

    #[test]
    fn topk_handles_negative_values() {
        let v = [-1.0f32, -3.0, -2.0, -0.5];
        let mut out = [0usize; 2];
        topk_indices_desc(&v, 2, &mut out);
        // Largest is -0.5 (idx 3), then -1.0 (idx 0).
        assert_eq!(out, [3, 0]);
    }

    #[test]
    fn topk_k_one_returns_argmax() {
        let v = [-5.0f32, 12.5, 3.3, 0.0];
        let mut out = [0usize; 1];
        topk_indices_desc(&v, 1, &mut out);
        assert_eq!(out, [1]);
    }

    #[test]
    fn topk_skips_nan_values() {
        // NaN > x is always false, so the linear scan should pick the
        // largest *non-NaN* entry for every slot. With three NaNs and one
        // real value, the first slot picks the real value and subsequent
        // slots fall back to NaN entries (any of them is fine — only the
        // non-NaN winner is asserted here).
        let v = [f32::NAN, 0.5, f32::NAN, f32::NAN];
        let mut out = [0usize; 1];
        topk_indices_desc(&v, 1, &mut out);
        assert_eq!(out, [1], "topk should pick the non-NaN value");
    }

    #[test]
    fn topk_all_nan_falls_back_to_lowest_indices() {
        // Models whose F16 weights contain NaN bit-patterns (e.g. the
        // shipped DS4-Flash q2-imatrix GGUF) can produce all-NaN router
        // probability vectors. In that degenerate case the C reference
        // leaves `best_i` at its sentinel and downstream Q8_K activation
        // pre-quant kills the NaN regardless. Mirror that behaviour by
        // falling back to sequential lowest-index selection.
        let v = [f32::NAN; 8];
        let mut out = [0usize; 4];
        topk_indices_desc(&v, 4, &mut out);
        assert_eq!(out, [0, 1, 2, 3], "all-NaN should pick sequential indices");
    }

    #[test]
    fn topk_all_neg_inf_falls_back_to_lowest_indices() {
        let v = [f32::NEG_INFINITY; 5];
        let mut out = [0usize; 3];
        topk_indices_desc(&v, 3, &mut out);
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn expert_weight_scale_floor_protects_against_collapse() {
        // Reproduces the C reference's `sum < 6.103515625e-5 -> sum = 6.103515625e-5`
        // floor: when probs collapse to ~0 the floor caps the rescale factor.
        let probs = [1e-8f32, 1e-8, 1e-8];
        let mut sum: f32 = probs.iter().sum();
        if sum < EXPERT_WEIGHT_SUM_EPS {
            sum = EXPERT_WEIGHT_SUM_EPS;
        }
        let scale = EXPERT_WEIGHT_SCALE / sum;
        // Without the floor, scale would be ~5e7 — way past the C reference's
        // bounded ~24576. The floor caps it.
        assert!(scale <= EXPERT_WEIGHT_SCALE / EXPERT_WEIGHT_SUM_EPS + 1e-3);
    }

    #[test]
    fn expert_weights_sum_to_scale_when_probs_nondegenerate() {
        // Pre-floor: sum(weights/sum) * scale == scale exactly.
        let probs = [0.2f32, 0.5, 0.3];
        let sum: f32 = probs.iter().sum();
        let scaled: Vec<f32> = probs
            .iter()
            .map(|&p| p / sum * EXPERT_WEIGHT_SCALE)
            .collect();
        let total: f32 = scaled.iter().sum();
        assert!((total - EXPERT_WEIGHT_SCALE).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Attention (MLA) — covers the merged 512-dim KV row.
    //
    // These tests exercise `attention_rows_inner`, the pure helper that takes
    // already-cached rows and produces per-head attention output. Each cached
    // row is HEAD_DIM = 512 wide (no latent/k_pe split at this layer — the
    // ring stores the merged post-RoPE row).
    // -----------------------------------------------------------------------

    use crate::model::kv_cache::HEAD_DIM;

    #[test]
    fn attention_single_kv_row_is_that_row() {
        // With one cached token, softmax weight on that single row is 1.0
        // when the sink is far below the dot. Pick q = the cached row so the
        // dot is large positive; the output should equal that row.
        let n_head = 1usize;

        let mut q = vec![0.0f32; n_head * HEAD_DIM];
        for (i, v) in q.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 0.01;
        }

        let kv: Vec<f32> = q.clone();
        let sinks = vec![-1e30f32; n_head];

        let mut out = vec![0.0f32; n_head * HEAD_DIM];
        attention_rows_inner(&mut out, &sinks, &q, &kv, n_head);

        for i in 0..HEAD_DIM {
            assert!(
                (out[i] - q[i]).abs() < 1e-5,
                "dim {i}: out={} q={}",
                out[i],
                q[i],
            );
        }
    }

    #[test]
    fn attention_dominated_sink_zeros_output() {
        // Sink logit huge relative to any score → row weight collapses to ~0
        // and the output goes to ~0 across the full HEAD_DIM.
        let n_head = 1usize;
        let q = vec![0.1f32; n_head * HEAD_DIM];
        let kv = vec![0.5f32; HEAD_DIM];
        let sinks = vec![1e6f32; n_head];

        let mut out = vec![0.0f32; n_head * HEAD_DIM];
        attention_rows_inner(&mut out, &sinks, &q, &kv, n_head);

        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-3, "dim {i} not collapsed: {v}");
        }
    }

    #[test]
    fn attention_uniform_rows_average_into_full_head() {
        // Three identical rows, sink set far below scores. The weighted
        // average over identical rows is just that row.
        let n_head = 2usize;
        let window = 3usize;

        let row: Vec<f32> = (0..HEAD_DIM).map(|i| (i as f32) * 0.001).collect();
        let mut kv = vec![0.0f32; window * HEAD_DIM];
        for r in 0..window {
            kv[r * HEAD_DIM..(r + 1) * HEAD_DIM].copy_from_slice(&row);
        }

        let mut q = vec![0.0f32; n_head * HEAD_DIM];
        for h in 0..n_head {
            q[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&row);
        }
        let sinks = vec![-1e30f32; n_head];

        let mut out = vec![0.0f32; n_head * HEAD_DIM];
        attention_rows_inner(&mut out, &sinks, &q, &kv, n_head);

        for h in 0..n_head {
            let oh = &out[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            for d in 0..HEAD_DIM {
                assert!(
                    (oh[d] - row[d]).abs() < 1e-4,
                    "head {h} dim {d}: oh={} expected={}",
                    oh[d],
                    row[d],
                );
            }
        }
    }

    #[test]
    fn attention_per_head_independence() {
        // Two heads with different sink logits should produce different
        // outputs even with the same q/k/v window. Catches accidental
        // sharing across heads.
        let n_head = 2usize;
        let q = vec![0.1f32; n_head * HEAD_DIM];
        let kv = vec![0.4f32; HEAD_DIM];
        let sinks = vec![-100.0f32, 100.0f32];

        let mut out = vec![0.0f32; n_head * HEAD_DIM];
        attention_rows_inner(&mut out, &sinks, &q, &kv, n_head);

        let head0 = &out[..HEAD_DIM];
        let head1 = &out[HEAD_DIM..];
        let mag0: f32 = head0.iter().map(|v| v.abs()).sum();
        let mag1: f32 = head1.iter().map(|v| v.abs()).sum();
        assert!(
            mag0 > mag1 * 10.0,
            "head0 mag {mag0} should dominate {mag1}"
        );
    }

    // -----------------------------------------------------------------------
    // Output HC reduction
    // -----------------------------------------------------------------------

    #[test]
    fn output_hc_weights_match_reference_formula() {
        // Spot-check against the C reference:
        //   w[i] = sigmoid_stable(pre[i] * scale + base[i]) + HC_EPS
        let pre = [0.0f32, 1.0, -1.0, 2.0];
        let scale = 0.5f32;
        let base = [0.0f32, -0.25, 0.5, 1.0];
        let mut got = [0.0f32; 4];
        output_hc_weights(&pre, scale, &base, &mut got);

        for (i, &w) in got.iter().enumerate() {
            let z = pre[i] * scale + base[i];
            let expected = 1.0 / (1.0 + (-z).exp()) + HC_EPS;
            assert!(
                (w - expected).abs() < 1e-6,
                "i={i}: got {w} want {expected}",
            );
        }
    }

    #[test]
    fn output_hc_weights_floor_above_eps() {
        // sigmoid_stable saturates at 0 / 1 for large magnitude inputs; the
        // +HC_EPS floor keeps weights strictly positive.
        let pre = [-1e6f32, 1e6];
        let base = [0.0f32, 0.0];
        let mut w = [0.0f32; 2];
        output_hc_weights(&pre, 1.0, &base, &mut w);
        assert!(
            w[0] >= HC_EPS,
            "negative-saturated weight {} below eps",
            w[0]
        );
        assert!(
            w[1] > 1.0 - 1e-3,
            "positive-saturated weight {} too low",
            w[1]
        );
        assert!(w[0].is_finite() && w[1].is_finite());
    }

    #[test]
    fn output_hc_weights_zero_input_is_half() {
        // sigmoid(0) = 0.5, so weights collapse to 0.5 + eps when pre/base/scale all zero.
        let pre = [0.0f32; 4];
        let base = [0.0f32; 4];
        let mut w = [0.0f32; 4];
        output_hc_weights(&pre, 0.0, &base, &mut w);
        for v in w {
            assert!((v - (0.5 + HC_EPS)).abs() < 1e-6);
        }
    }
}
