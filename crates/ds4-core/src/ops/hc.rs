//! Hyper-Connection (HC) ops: Sinkhorn split, weighted sum, and post mix.
//!
//! See rfcs/0002-forward-pass.md §2 / §3.7. DS4 Flash uses n_hc = 4 parallel
//! residual streams. Each sub-layer consumes a weighted sum of the streams
//! (the "pre" step) and the sub-layer output is mixed back into the streams
//! via a learned post-gate and doubly-stochastic combine matrix (the "post"
//! step).
//!
//! The control vector is projected from the RMSNorm'd HC state by a small
//! F16 matmul (`hc_attn_fn` / `hc_ffn_fn`). The projected `mix` vector is
//! then split into pre-weights (sigmoid + eps), post-gates (2*sigmoid), and
//! a combine matrix (softmax-rows + Sinkhorn iterations).
//!
//! Reference: `hc_split_sinkhorn_one`, `hc_weighted_sum_one`, `hc_post_one`,
//! `hc_from_plain_embedding` in antirez/ds4 ds4.c.

use crate::ops::swiglu::sigmoid_stable;

/// Compute HC control split from the projected mix vector.
///
/// Inputs:
/// * `mix`   — projected control vector, shape `[2*n_hc + n_hc*n_hc]`.
/// * `scale` — three scalars: `[pre_scale, post_scale, comb_scale]`.
/// * `base`  — biases matching `mix` shape.
///
/// Outputs:
/// * `pre`  — pre-weights for the weighted sum, shape `[n_hc]`.
/// * `post` — post-gates for the HC post step, shape `[n_hc]`.
/// * `comb` — combine matrix for the HC post step, shape `[n_hc*n_hc]`. Flat layout: `comb[src +
///   dst * n_hc]` (row=dst, col=src in row-major).
///
/// The algorithm matches `hc_split_sinkhorn_one` in ds4.c exactly:
/// 1. pre[i]  = sigmoid(mix[i] * pre_scale + base[i]) + eps
/// 2. post[i] = 2 * sigmoid(mix[n_hc+i] * post_scale + base[n_hc+i])
/// 3. comb is initialised from the tail of mix/base, row-softmax'd, then `iters` rounds of Sinkhorn
///    column-row normalisation.
#[allow(clippy::too_many_arguments)]
pub fn hc_control_split(
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    pre: &mut [f32],
    post: &mut [f32],
    comb: &mut [f32],
    n_hc: usize,
    iters: usize,
    eps: f32,
) {
    assert_eq!(pre.len(), n_hc, "hc_control_split: pre len mismatch");
    assert_eq!(post.len(), n_hc, "hc_control_split: post len mismatch");
    assert_eq!(
        comb.len(),
        n_hc * n_hc,
        "hc_control_split: comb len mismatch"
    );
    assert!(
        scale.len() >= 3,
        "hc_control_split: scale must have at least 3 elements"
    );
    assert!(
        mix.len() >= 2 * n_hc + n_hc * n_hc,
        "hc_control_split: mix too short"
    );
    assert!(
        base.len() >= 2 * n_hc + n_hc * n_hc,
        "hc_control_split: base too short"
    );

    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    // Pre weights: sigmoid + eps.
    for (i, pre_i) in pre.iter_mut().enumerate().take(n_hc) {
        let z = mix[i] * pre_scale + base[i];
        *pre_i = sigmoid_stable(z) + eps;
    }

    // Post gates: 2 * sigmoid.
    for (i, post_i) in post.iter_mut().enumerate().take(n_hc) {
        let off = n_hc + i;
        let z = mix[off] * post_scale + base[off];
        *post_i = 2.0 * sigmoid_stable(z);
    }

    // Combine matrix: write directly into the caller-provided `comb` slot.
    // c[idx] where idx = src + dst * n_hc  (dst = row, src = col)
    let c = &mut comb[..n_hc * n_hc];
    for dst in 0..n_hc {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let off = 2 * n_hc + idx;
            let v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if v > row_max {
                row_max = v;
            }
        }

        // Row softmax.
        let mut row_sum = 0.0f32;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let e = (c[idx] - row_max).exp();
            c[idx] = e;
            row_sum += e;
        }

        let inv = 1.0 / (row_sum + eps);
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    // Initial column normalisation (per src).
    for src in 0..n_hc {
        let mut sum = 0.0f32;
        for dst in 0..n_hc {
            sum += c[src + dst * n_hc];
        }
        let inv = 1.0 / (sum + eps);
        for dst in 0..n_hc {
            c[src + dst * n_hc] *= inv;
        }
    }

    // Sinkhorn iterations: row norm then column norm.
    for _ in 1..iters {
        // Row normalisation (per dst).
        for dst in 0..n_hc {
            let mut sum = 0.0f32;
            for src in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for src in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }

        // Column normalisation (per src).
        for src in 0..n_hc {
            let mut sum = 0.0f32;
            for dst in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for dst in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
    }
}

/// Weighted sum of HC streams into a single vector.
///
/// `streams` has shape `[n_hc, n_embd]` in row-major order
/// (stream `h` starts at `h * n_embd`).
/// `weights` has shape `[n_hc]`.
/// `out` has shape `[n_embd]`.
///
/// `out[d] = sum_h streams[h * n_embd + d] * weights[h]`
pub fn hc_weighted_sum(
    streams: &[f32],
    weights: &[f32],
    out: &mut [f32],
    n_embd: usize,
    n_hc: usize,
) {
    assert_eq!(
        streams.len(),
        n_hc * n_embd,
        "hc_weighted_sum: streams len mismatch"
    );
    assert_eq!(weights.len(), n_hc, "hc_weighted_sum: weights len mismatch");
    assert_eq!(out.len(), n_embd, "hc_weighted_sum: out len mismatch");

    out.fill(0.0);
    for (h, &w) in weights.iter().enumerate().take(n_hc) {
        let base = h * n_embd;
        for d in 0..n_embd {
            out[d] += streams[base + d] * w;
        }
    }
}

/// HC post step: inject sub-layer output and mix residual streams.
///
/// `out_hc`      — output HC state, shape `[n_hc, n_embd]`.
/// `block_out`   — sub-layer output, shape `[n_embd]`.
/// `residual_hc` — previous HC state, shape `[n_hc, n_embd]`.
/// `post`        — post-gates, shape `[n_hc]`.
/// `comb`        — combine matrix, shape `[n_hc*n_hc]`.
///   Storage layout matches `hc_control_split`: `comb[src + dst*n_hc]`
///   stores the entry that maps source stream `src` into destination
///   stream `dst`.
///
/// For each destination stream `dst` and embedding dim `d`:
/// ```text
/// out_hc[dst, d] = block_out[d] * post[dst]
///                + sum_src comb[src + dst * n_hc] * residual_hc[src, d]
/// ```
///
/// This indexing matches `hc_post_one` in ds4.c exactly.
pub fn hc_post(
    out_hc: &mut [f32],
    block_out: &[f32],
    residual_hc: &[f32],
    post: &[f32],
    comb: &[f32],
    n_embd: usize,
    n_hc: usize,
) {
    assert_eq!(out_hc.len(), n_hc * n_embd, "hc_post: out_hc len mismatch");
    assert_eq!(block_out.len(), n_embd, "hc_post: block_out len mismatch");
    assert_eq!(
        residual_hc.len(),
        n_hc * n_embd,
        "hc_post: residual_hc len mismatch"
    );
    assert_eq!(post.len(), n_hc, "hc_post: post len mismatch");
    assert_eq!(comb.len(), n_hc * n_hc, "hc_post: comb len mismatch");

    // Loop order src → dst → d so the inner loop sweeps contiguous slices of
    // residual_hc[src, ..] and out_hc[dst, ..] (both n_embd-wide), letting the
    // compiler vectorise the FMA over the embedding dimension.
    for (dst, &post_gate) in post.iter().enumerate().take(n_hc) {
        let out_row = &mut out_hc[dst * n_embd..(dst + 1) * n_embd];
        for (o, &b) in out_row.iter_mut().zip(block_out.iter()) {
            *o = b * post_gate;
        }
        for src in 0..n_hc {
            let c = comb[src + dst * n_hc];
            let res_row = &residual_hc[src * n_embd..(src + 1) * n_embd];
            for (o, &r) in out_row.iter_mut().zip(res_row.iter()) {
                *o += c * r;
            }
        }
    }
}

/// Copy a plain embedding vector into all HC streams.
///
/// `out_hc` has shape `[n_hc, n_embd]`.
/// `x` has shape `[n_embd]`.
pub fn hc_from_plain_embedding(out_hc: &mut [f32], x: &[f32], n_embd: usize, n_hc: usize) {
    assert_eq!(
        out_hc.len(),
        n_hc * n_embd,
        "hc_from_plain_embedding: out_hc len mismatch"
    );
    assert_eq!(x.len(), n_embd, "hc_from_plain_embedding: x len mismatch");

    for h in 0..n_hc {
        out_hc[h * n_embd..(h + 1) * n_embd].copy_from_slice(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0) + tol
    }

    #[test]
    fn hc_from_plain_embedding_copies_correctly() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 12];
        hc_from_plain_embedding(&mut out, &x, 3, 4);
        for h in 0..4 {
            assert_eq!(&out[h * 3..(h + 1) * 3], &[1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn hc_weighted_sum_basic() {
        // 2 streams, 3 dims.
        // stream 0 = [1, 2, 3], stream 1 = [4, 5, 6]
        let streams = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![0.5f32, 2.0];
        let mut out = vec![0.0f32; 3];
        hc_weighted_sum(&streams, &weights, &mut out, 3, 2);
        // out[d] = 0.5*s0[d] + 2.0*s1[d]
        assert_eq!(out[0], 0.5 * 1.0 + 2.0 * 4.0);
        assert_eq!(out[1], 0.5 * 2.0 + 2.0 * 5.0);
        assert_eq!(out[2], 0.5 * 3.0 + 2.0 * 6.0);
    }

    #[test]
    fn hc_control_split_shape_and_ranges() {
        let n_hc = 4;
        let mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        let scale = vec![1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        let mut pre = vec![0.0f32; n_hc];
        let mut post = vec![0.0f32; n_hc];
        let mut comb = vec![0.0f32; n_hc * n_hc];

        hc_control_split(
            &mix, &scale, &base, &mut pre, &mut post, &mut comb, n_hc, 20, 1e-6,
        );

        // Pre: sigmoid(0) + eps = 0.5 + eps
        for &p in &pre {
            assert!(p > 0.5 && p < 0.50001, "pre = {p}");
        }

        // Post: 2 * sigmoid(0) = 1.0
        for &p in &post {
            assert!((p - 1.0).abs() < 1e-6, "post = {p}");
        }

        // Comb: row sums and column sums should be ~1 (doubly stochastic).
        for dst in 0..n_hc {
            let row_sum: f32 = (0..n_hc).map(|src| comb[src + dst * n_hc]).sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "row {dst} sum = {row_sum}");
        }
        for src in 0..n_hc {
            let col_sum: f32 = (0..n_hc).map(|dst| comb[src + dst * n_hc]).sum();
            assert!((col_sum - 1.0).abs() < 1e-4, "col {src} sum = {col_sum}");
        }
    }

    #[test]
    fn hc_post_matches_manual() {
        let n_hc = 2;
        let n_embd = 3;
        let block_out = vec![1.0f32, 2.0, 3.0];
        let residual_hc = vec![
            10.0f32, 20.0, 30.0, // src 0
            40.0f32, 50.0, 60.0, // src 1
        ];
        let post = vec![1.0f32, 1.0];
        // comb stored as comb[src + dst*n_hc]
        // For dst=0: use comb[0 + 0*2]=0.5, comb[1 + 0*2]=0.5
        // For dst=1: use comb[0 + 1*2]=0.5, comb[1 + 1*2]=0.5
        let comb = vec![0.5f32, 0.5, 0.5, 0.5];
        let mut out_hc = vec![0.0f32; n_hc * n_embd];

        hc_post(
            &mut out_hc,
            &block_out,
            &residual_hc,
            &post,
            &comb,
            n_embd,
            n_hc,
        );

        // dst=0, d=0: block[0]*post[0] + comb[0+0*2]*res[0] + comb[1+0*2]*res[3]
        //            = 1*1 + 0.5*10 + 0.5*40 = 1 + 5 + 20 = 26
        assert!((out_hc[0] - 26.0).abs() < 1e-6, "out[0] = {}", out_hc[0]);
        // dst=1, d=0: 1*1 + 0.5*10 + 0.5*40 = 26 (same because comb is uniform)
        assert!((out_hc[3] - 26.0).abs() < 1e-6, "out[3] = {}", out_hc[3]);
    }

    #[test]
    fn hc_post_uses_split_layout_for_asymmetric_comb() {
        // Non-symmetric comb to verify the post-step honours the
        // `comb[src + dst*n_hc]` layout produced by `hc_control_split`.
        let n_hc = 2;
        let n_embd = 1;
        let block_out = vec![0.0f32];
        let residual_hc = vec![1.0f32, 10.0]; // src 0 = 1, src 1 = 10
        let post = vec![0.0f32, 0.0];
        // Row-major by dst:
        //   dst=0: [comb(src=0,dst=0)=0.7, comb(src=1,dst=0)=0.3]
        //   dst=1: [comb(src=0,dst=1)=0.2, comb(src=1,dst=1)=0.8]
        let comb = vec![0.7f32, 0.3, 0.2, 0.8];
        let mut out_hc = vec![0.0f32; n_hc * n_embd];

        hc_post(
            &mut out_hc,
            &block_out,
            &residual_hc,
            &post,
            &comb,
            n_embd,
            n_hc,
        );

        // dst=0: 0.7*1 + 0.3*10 = 3.7
        assert!((out_hc[0] - 3.7).abs() < 1e-6, "out[0] = {}", out_hc[0]);
        // dst=1: 0.2*1 + 0.8*10 = 8.2
        assert!((out_hc[1] - 8.2).abs() < 1e-6, "out[1] = {}", out_hc[1]);
    }

    #[test]
    fn sinkhorn_iterations_produce_doubly_stochastic() {
        let n_hc = 4;
        let mut mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        // Give the combine tail some structure.
        for i in 0..(n_hc * n_hc) {
            mix[2 * n_hc + i] = (i as f32) * 0.1 - 0.8;
        }
        let scale = vec![1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        let mut pre = vec![0.0f32; n_hc];
        let mut post = vec![0.0f32; n_hc];
        let mut comb = vec![0.0f32; n_hc * n_hc];

        hc_control_split(
            &mix, &scale, &base, &mut pre, &mut post, &mut comb, n_hc, 20, 1e-6,
        );

        // Check doubly-stochastic property.
        for dst in 0..n_hc {
            let s: f32 = (0..n_hc).map(|src| comb[src + dst * n_hc]).sum();
            assert!((s - 1.0).abs() < 1e-4, "row {dst} sum = {s}");
        }
        for src in 0..n_hc {
            let s: f32 = (0..n_hc).map(|dst| comb[src + dst * n_hc]).sum();
            assert!((s - 1.0).abs() < 1e-4, "col {src} sum = {s}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri's f32 emulation produces sub-ULP non-determinism")]
    fn is_deterministic_across_runs() {
        let n_hc = 4;
        let mix: Vec<f32> = (0..(2 * n_hc + n_hc * n_hc))
            .map(|i| ((i as f32) * 0.03 - 1.0).sin())
            .collect();
        let scale = vec![0.5f32, 0.3, 0.2];
        let base: Vec<f32> = (0..mix.len()).map(|i| (i as f32) * 0.01).collect();
        let mut pre1 = vec![0.0f32; n_hc];
        let mut post1 = vec![0.0f32; n_hc];
        let mut comb1 = vec![0.0f32; n_hc * n_hc];
        let mut pre2 = vec![0.0f32; n_hc];
        let mut post2 = vec![0.0f32; n_hc];
        let mut comb2 = vec![0.0f32; n_hc * n_hc];

        hc_control_split(
            &mix, &scale, &base, &mut pre1, &mut post1, &mut comb1, n_hc, 20, 1e-6,
        );
        hc_control_split(
            &mix, &scale, &base, &mut pre2, &mut post2, &mut comb2, n_hc, 20, 1e-6,
        );

        assert_eq!(pre1, pre2);
        assert_eq!(post1, post2);
        assert_eq!(comb1, comb2);
    }

    #[test]
    fn hc_weighted_sum_zero_weights_yields_zero() {
        let streams = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![0.0f32, 0.0];
        let mut out = vec![9.0f32; 3];
        hc_weighted_sum(&streams, &weights, &mut out, 3, 2);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn hc_weighted_sum_single_stream_acts_as_scale() {
        let streams = vec![1.0f32, -2.0, 3.5];
        let weights = vec![2.0f32];
        let mut out = vec![0.0f32; 3];
        hc_weighted_sum(&streams, &weights, &mut out, 3, 1);
        assert_eq!(out, vec![2.0, -4.0, 7.0]);
    }

    #[test]
    fn hc_weighted_sum_clears_existing_output() {
        // out is accumulator-style; pre-existing values must be wiped.
        let streams = vec![1.0f32, 2.0];
        let weights = vec![1.0f32];
        let mut out = vec![100.0f32, 200.0];
        hc_weighted_sum(&streams, &weights, &mut out, 2, 1);
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn hc_post_zero_post_gate_drops_block_out() {
        let n_hc = 2;
        let n_embd = 2;
        let block_out = vec![100.0f32, 200.0];
        let residual_hc = vec![1.0f32, 2.0, 3.0, 4.0];
        let post = vec![0.0f32, 0.0];
        // Identity comb (each dst takes only its same-index src).
        let comb = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut out_hc = vec![0.0f32; n_hc * n_embd];
        hc_post(
            &mut out_hc,
            &block_out,
            &residual_hc,
            &post,
            &comb,
            n_embd,
            n_hc,
        );
        assert_eq!(out_hc, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn hc_post_zero_comb_isolates_block_out_times_post() {
        let n_hc = 2;
        let n_embd = 2;
        let block_out = vec![5.0f32, 7.0];
        let residual_hc = vec![1.0f32, 2.0, 3.0, 4.0];
        let post = vec![0.5f32, 2.0];
        let comb = vec![0.0f32; 4];
        let mut out_hc = vec![0.0f32; n_hc * n_embd];
        hc_post(
            &mut out_hc,
            &block_out,
            &residual_hc,
            &post,
            &comb,
            n_embd,
            n_hc,
        );
        assert_eq!(out_hc, vec![2.5, 3.5, 10.0, 14.0]);
    }

    #[test]
    fn hc_from_plain_embedding_n_hc_one() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        hc_from_plain_embedding(&mut out, &x, 3, 1);
        assert_eq!(out, x);
    }

    #[test]
    fn hc_control_split_with_zero_iters_skips_sinkhorn_loop() {
        // iters=0 means the loop `for _ in 1..0` runs zero times, but the
        // initial row-softmax + column-norm still execute. Columns should
        // sum to ~1; rows generally won't.
        let n_hc = 4;
        let mut mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        for i in 0..(n_hc * n_hc) {
            mix[2 * n_hc + i] = (i as f32) * 0.2 - 1.5;
        }
        let scale = vec![1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        let mut pre = vec![0.0f32; n_hc];
        let mut post = vec![0.0f32; n_hc];
        let mut comb = vec![0.0f32; n_hc * n_hc];
        hc_control_split(
            &mix, &scale, &base, &mut pre, &mut post, &mut comb, n_hc, 0, 1e-6,
        );
        for src in 0..n_hc {
            let col_sum: f32 = (0..n_hc).map(|dst| comb[src + dst * n_hc]).sum();
            assert!((col_sum - 1.0).abs() < 1e-4, "col {src} sum = {col_sum}");
        }
    }

    #[test]
    fn hc_control_split_pre_eps_floor() {
        // Strongly negative mix*pre_scale drives sigmoid -> 0, so pre -> eps.
        let n_hc = 2;
        let mut mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        mix[0] = -1000.0;
        mix[1] = -1000.0;
        let scale = vec![1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        let mut pre = vec![0.0f32; n_hc];
        let mut post = vec![0.0f32; n_hc];
        let mut comb = vec![0.0f32; n_hc * n_hc];
        let eps = 1e-3;
        hc_control_split(
            &mix, &scale, &base, &mut pre, &mut post, &mut comb, n_hc, 5, eps,
        );
        for &p in &pre {
            assert!((p - eps).abs() < 1e-6, "pre = {p}, expected ~{eps}");
        }
    }

    #[test]
    fn hc_control_split_post_range_zero_to_two() {
        let n_hc = 2;
        let mut mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        // post indexes are mix[n_hc..2*n_hc].
        mix[n_hc] = -1000.0;
        mix[n_hc + 1] = 1000.0;
        let scale = vec![1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        let mut pre = vec![0.0f32; n_hc];
        let mut post = vec![0.0f32; n_hc];
        let mut comb = vec![0.0f32; n_hc * n_hc];
        hc_control_split(
            &mix, &scale, &base, &mut pre, &mut post, &mut comb, n_hc, 5, 1e-6,
        );
        assert!(post[0].abs() < 1e-6, "post[0] = {}", post[0]);
        assert!((post[1] - 2.0).abs() < 1e-6, "post[1] = {}", post[1]);
    }

    #[test]
    fn hc_control_split_base_offsets_pre_post() {
        // With mix=0 and scale=1, pre_z = base[i], post_z = base[n_hc+i].
        let n_hc = 2;
        let mix = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        let scale = vec![1.0f32, 1.0, 1.0];
        let mut base = vec![0.0f32; 2 * n_hc + n_hc * n_hc];
        base[0] = 1.0;
        base[1] = -1.0;
        base[n_hc] = 2.0;
        base[n_hc + 1] = -2.0;
        let mut pre = vec![0.0f32; n_hc];
        let mut post = vec![0.0f32; n_hc];
        let mut comb = vec![0.0f32; n_hc * n_hc];
        hc_control_split(
            &mix, &scale, &base, &mut pre, &mut post, &mut comb, n_hc, 5, 0.0,
        );
        let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
        assert!((pre[0] - sigmoid(1.0)).abs() < 1e-6, "pre[0] = {}", pre[0]);
        assert!((pre[1] - sigmoid(-1.0)).abs() < 1e-6, "pre[1] = {}", pre[1]);
        assert!(
            (post[0] - 2.0 * sigmoid(2.0)).abs() < 1e-6,
            "post[0] = {}",
            post[0]
        );
        assert!(
            (post[1] - 2.0 * sigmoid(-2.0)).abs() < 1e-6,
            "post[1] = {}",
            post[1]
        );
    }

    #[test]
    fn hc_post_linear_in_block_out() {
        // out_hc[dst, d] depends linearly on block_out[d] via post[dst].
        // Doubling block_out should add (post[dst] * orig_block_out[d]) to each row.
        let n_hc = 2;
        let n_embd = 3;
        let block_out = vec![1.0f32, 2.0, 3.0];
        let residual_hc = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let post = vec![0.5f32, 1.5];
        let comb = vec![0.7f32, 0.3, 0.2, 0.8];

        let mut out1 = vec![0.0f32; n_hc * n_embd];
        let mut out2 = vec![0.0f32; n_hc * n_embd];
        hc_post(
            &mut out1,
            &block_out,
            &residual_hc,
            &post,
            &comb,
            n_embd,
            n_hc,
        );
        let block_out_2: Vec<f32> = block_out.iter().map(|v| v * 2.0).collect();
        hc_post(
            &mut out2,
            &block_out_2,
            &residual_hc,
            &post,
            &comb,
            n_embd,
            n_hc,
        );
        for (dst, &p) in post.iter().enumerate().take(n_hc) {
            for (d, &b) in block_out.iter().enumerate().take(n_embd) {
                let idx = dst * n_embd + d;
                let diff = out2[idx] - out1[idx];
                let expect = p * b;
                assert!(
                    (diff - expect).abs() < 1e-5,
                    "dst={dst} d={d}: diff {diff} != {expect}",
                );
            }
        }
    }
}
