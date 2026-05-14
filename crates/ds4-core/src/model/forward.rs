//! Forward pass orchestration.
//!
//! See rfcs/0002-forward-pass.md §2 / §3. Phase 1 implements a CPU reference
//! forward pass: single-threaded, f32 activations, sliding-window attention
//! (no compressor/indexer), no FP8 KV round-trip.
//!
//! Missing pieces (PRs 5–6):
//! * IQ2_XXS / Q2_K / Q4_K / IQ4_K quant types and matmul dispatch.
//! * Routed expert MoE is stubbed out; only the shared expert runs.

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::{
    config::ModelConfig,
    engine::Engine,
    model::{WeightMap, kv_cache::KvCache, layer::LayerWeights},
    ops::{
        hc::{hc_control_split, hc_from_plain_embedding, hc_post, hc_weighted_sum},
        matmul::matmul_row,
        norm::{rms_norm, rms_norm_no_weight},
        rope::{apply_rope, apply_rope_inverse},
        swiglu::swiglu,
    },
    quant::q8_0,
    session::Session,
};

/// Run a single decode step: compute logits for the token at `session.pos()`.
///
/// This is the entry point called by `Session::eval_token`.
pub fn forward_decode(session: &mut Session, engine: &Arc<Engine>) -> Result<Vec<f32>> {
    let model = &engine.weights;
    let config = &engine.config;
    let pos = session.pos() as usize;

    // --- Token embedding ---------------------------------------------------
    let mut plain = vec![0.0f32; config.n_embd as usize];
    embed_token(model, session.tokens()[pos], &mut plain)?;

    // --- Copy embedding into HC streams ------------------------------------
    let n_hc = config.n_hc as usize;
    let n_embd = config.n_embd as usize;
    let hc_dim = n_hc * n_embd;
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
        let (ffn_post, ffn_comb) =
            layer_ffn_decode(&mut ffn_out, config, &layer, &after_attn_hc, il as usize)?;

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
    for (i, chunk) in row.chunks_exact(2).enumerate() {
        out[i] = q8_0::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
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

    // KV projection (matches antirez/ds4 ds4.c). The attn_kv matmul produces
    // a single DS4_N_HEAD_DIM = 512 wide row, split into:
    //   * KV_LATENT_DIM = 448 — non-positional ("nope") slice, RMSNorm'd and cached
    //   * K_PE_DIM      =  64 — decoupled RoPE key, RoPE'd and cached separately
    // The MLA up-projection of the latent into per-head K/V lands in PR #5;
    // Phase 1 caches both pieces with the canonical dims so the cache asserts
    // don't fire and downstream attention can read the correct shape.
    use crate::model::kv_cache::{K_PE_DIM, KV_LATENT_DIM};
    let kv_full_dim = KV_LATENT_DIM + K_PE_DIM;
    let mut kv_raw = vec![0.0f32; kv_full_dim];
    matmul_row(layer.attn_kv, &attn_norm, &mut kv_raw);

    let mut kv_latent = vec![0.0f32; KV_LATENT_DIM];
    rms_norm(
        &kv_raw[..KV_LATENT_DIM],
        layer.attn_kv_a_norm,
        1e-6,
        &mut kv_latent,
    );

    let mut k_pe = kv_raw[KV_LATENT_DIM..].to_vec();
    apply_rope(&mut k_pe, pos, &engine.rope_freqs);

    // RoPE per head on Q (uses precomputed frequency cache from Engine).
    for h in 0..n_head {
        let qh = &mut q[h * n_head_dim..(h + 1) * n_head_dim];
        apply_rope(qh, pos, &engine.rope_freqs);
    }

    // Store latent + k_pe in the cache and advance the watermark so the
    // just-written token participates in this step's softmax.
    kv_cache.write_latent(il, pos, &kv_latent)?;
    kv_cache.write_k_pe(il, pos, &k_pe)?;
    if pos + 1 > kv_cache.len() {
        kv_cache.set_pos(pos + 1);
    }

    // Attention over cached KV rows.
    let mut heads = vec![0.0f32; q_dim];
    let kv_len = (pos + 1).min(kv_cache.ctx_size());
    let sliding_window = 128usize;
    let start_pos = kv_len.saturating_sub(sliding_window);

    attention_rows(
        &mut heads, layer, &q, kv_cache, il, start_pos, kv_len, n_head, n_head_dim,
    )?;

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
    let mut qr = vec![0.0f32; 1024];
    matmul_row(layer.attn_q_a, norm, &mut qr);

    let mut qr_norm = vec![0.0f32; 1024];
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
    start_pos: usize,
    end_pos: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<()> {
    use crate::model::kv_cache::KV_LATENT_DIM;

    let sinks = layer.attn_sinks;
    let kq_scale = 1.0 / (head_dim as f32).sqrt();
    let window_len = end_pos - start_pos;

    // Borrow the contiguous layer prefix once and slice the active window —
    // chunks_exact then yields one per-token row without intermediate allocs.
    let layer_prefix = kv_cache.latent_layer_prefix(il, end_pos);
    let window = &layer_prefix[start_pos * KV_LATENT_DIM..end_pos * KV_LATENT_DIM];

    let mut scores = vec![0.0f32; window_len];

    for h in 0..n_head {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let oh = &mut out_heads[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);

        let mut max_score = sinks[h];

        for (i, kv) in window.chunks_exact(KV_LATENT_DIM).enumerate() {
            // qh is `head_dim` wide; kv is the full latent. zip caps at head_dim.
            let score = qh.iter().zip(kv.iter()).map(|(&q, &k)| q * k).sum::<f32>() * kq_scale;
            scores[i] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut denom = (sinks[h] - max_score).exp();
        for (i, kv) in window.chunks_exact(KV_LATENT_DIM).enumerate() {
            let weight = (scores[i] - max_score).exp();
            denom += weight;
            for (o, &k) in oh.iter_mut().zip(kv.iter()) {
                *o += k * weight;
            }
        }

        let inv = 1.0 / denom;
        for v in oh.iter_mut() {
            *v *= inv;
        }
    }

    Ok(())
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

    for (group_input, group_output) in heads
        .chunks_exact(group_dim)
        .zip(low.chunks_exact_mut(rank))
    {
        matmul_row(layer.attn_output_a, group_input, group_output);
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
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n_embd = config.n_embd as usize;
    let n_hc = config.n_hc as usize;

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

    // Router + routed experts (stubbed — needs IQ2_XXS / Q2_K).
    let mut moe_out = vec![0.0f32; n_embd];
    if layer.ffn_gate_tid2eid.is_some() || il >= 3 {
        // TODO: implement routed MoE once IQ2_XXS / Q2_K / Q4_K / IQ4_K
        // quant types and matmul dispatch land (PRs 5–6).
        // For now, routed expert contribution is zero.
        moe_out.fill(0.0);
    }

    // Sum shared + routed.
    for i in 0..n_embd {
        out[i] = shared_out[i] + moe_out[i];
    }

    Ok((post, comb))
}

fn shared_expert_decode(layer: &LayerWeights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
    let hidden = 2048usize; // DS4_N_FF_EXP

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

    // Reduce HC streams to a single vector.
    let mut plain = vec![0.0f32; n_embd];
    // The output uses a learned combine matrix, but for Phase 1 we can sum
    // the streams. TODO: load output_weights from GGUF and do proper HC reduce.
    for h in 0..n_hc {
        for d in 0..n_embd {
            plain[d] += residual_hc[h * n_embd + d];
        }
    }

    // Final RMSNorm.
    let mut norm = vec![0.0f32; n_embd];
    let output_norm = model.f32_1d("output_norm.weight", n_embd)?;
    rms_norm(&plain, output_norm, 1e-6, &mut norm);

    // Output projection.
    let output_weight = model.q8_0("output.weight")?;
    matmul_row(output_weight, &norm, logits);

    Ok(())
}
