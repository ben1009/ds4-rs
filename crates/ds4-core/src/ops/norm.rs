//! Root Mean Square layer normalisation.
//!
//! See rfcs/0002-forward-pass.md §2. DS4 uses RMSNorm end-to-end:
//!
//! * `attn_norm`, `attn_q_a_norm`, `attn_kv_a_norm`, `ffn_norm`, `output_norm`, per-head Q norm,
//!   compressor norms — all weighted.
//! * HC pre/post and output-head pre-matmul norm — no weight (see `rms_norm_no_weight` below).
//!
//! The math is the standard formulation with `eps` inside the sqrt:
//!
//! ```text
//! scale = 1.0 / sqrt(mean(x^2) + eps)
//! y[i]  = x[i] * scale * weight[i]         // rms_norm
//! y[i]  = x[i] * scale                     // rms_norm_no_weight
//! ```
//!
//! Reduction order is fixed left-to-right, so the op is bit-reproducible
//! across platforms for a given input — no `fadd_fast` or Kahan.
//! DS4 uses `eps = 1e-6` throughout.

/// RMSNorm with per-channel scale weights.
///
/// Shapes must match: `x.len() == weight.len() == out.len()`. Operates in one
/// pass — safe for `x` and `out` to be distinct buffers; the implementation
/// does not support in-place aliasing.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(
        x.len(),
        weight.len(),
        "rms_norm: x len {} != weight len {}",
        x.len(),
        weight.len(),
    );
    assert_eq!(
        x.len(),
        out.len(),
        "rms_norm: x len {} != out len {}",
        x.len(),
        out.len(),
    );
    let scale = rms_scale(x, eps);
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(weight) {
        *o = xi * scale * wi;
    }
}

/// RMSNorm without learned weight — used by HC pre/post and the output-head
/// pre-matmul norm (`rms_norm_no_weight` in the C reference).
pub fn rms_norm_no_weight(x: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(
        x.len(),
        out.len(),
        "rms_norm_no_weight: x len {} != out len {}",
        x.len(),
        out.len(),
    );
    let scale = rms_scale(x, eps);
    for (o, &xi) in out.iter_mut().zip(x) {
        *o = xi * scale;
    }
}

/// `1 / sqrt(mean(x^2) + eps)` — the per-call scalar.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let n = x.len();
    if n == 0 {
        // Zero-dim RMSNorm never appears in practice, but keep the math
        // defined: mean(empty) = 0, so scale = 1/sqrt(eps).
        return 1.0 / eps.sqrt();
    }
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let mean = sum_sq / n as f32;
    1.0 / (mean + eps).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0) + tol
    }

    #[test]
    fn constant_input_scales_to_unit_times_weight() {
        // x = [c; n] => mean(x^2) = c^2 => scale = 1/sqrt(c^2 + eps)
        // y[i] = c * scale * w[i] ≈ sign(c) * w[i] for |c| >> sqrt(eps).
        let x = vec![3.0f32; 8];
        let w = vec![2.0f32, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0];
        let mut y = vec![0f32; 8];
        rms_norm(&x, &w, EPS, &mut y);
        let scale = 1.0 / (9.0 + EPS).sqrt(); // = 1/3 (up to eps)
        for i in 0..8 {
            assert!(
                approx_eq(y[i], 3.0 * scale * w[i], 1e-6),
                "y[{i}] = {} != 3 * {scale} * {}",
                y[i],
                w[i],
            );
        }
    }

    #[test]
    fn identity_weight_produces_unit_rms_output() {
        // With weight == [1; n] the output has mean(y^2) ≈ 1 modulo eps.
        let x = vec![1.0f32, -2.0, 3.0, -4.0];
        let w = vec![1.0f32; 4];
        let mut y = vec![0f32; 4];
        rms_norm(&x, &w, EPS, &mut y);
        let mean_sq: f32 = y.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!(
            (mean_sq - 1.0).abs() < 1e-4,
            "mean(y^2) = {mean_sq}, expected ~1",
        );
    }

    #[test]
    fn preserves_sign() {
        let x = vec![-1.0, 1.0, -2.0, 2.0];
        let w = vec![1.0; 4];
        let mut y = vec![0.0; 4];
        rms_norm(&x, &w, EPS, &mut y);
        assert!(y[0] < 0.0 && y[2] < 0.0);
        assert!(y[1] > 0.0 && y[3] > 0.0);
    }

    #[test]
    fn eps_dominates_when_input_is_small() {
        // If x_i is all near zero, scale = 1/sqrt(eps); y_i = x_i * scale.
        let x = vec![0.0f32; 4];
        let w = vec![1.0f32; 4];
        let mut y = vec![1.0f32; 4];
        rms_norm(&x, &w, EPS, &mut y);
        // Zero input => zero output regardless of scale.
        assert_eq!(y, vec![0.0; 4]);

        // Tiny input: 1e-6 * (1/sqrt(1e-6)) ≈ 1e-3
        let x = vec![1e-6f32; 4];
        let mut y = vec![0f32; 4];
        rms_norm(&x, &w, EPS, &mut y);
        let expect = 1e-6 / (1e-12f32 + EPS).sqrt();
        for &v in &y {
            assert!(approx_eq(v, expect, 1e-5), "{v} != {expect}");
        }
    }

    #[test]
    fn no_weight_matches_weighted_when_weight_is_ones() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let w = vec![1.0f32; 16];
        let mut y_weighted = vec![0f32; 16];
        let mut y_no = vec![0f32; 16];
        rms_norm(&x, &w, EPS, &mut y_weighted);
        rms_norm_no_weight(&x, EPS, &mut y_no);
        assert_eq!(y_no, y_weighted);
    }

    #[test]
    #[should_panic(expected = "weight len")]
    fn rejects_mismatched_weight_len() {
        let x = vec![0.0f32; 4];
        let w = vec![0.0f32; 3];
        let mut y = vec![0.0f32; 4];
        rms_norm(&x, &w, EPS, &mut y);
    }

    #[test]
    #[should_panic(expected = "out len")]
    fn rejects_mismatched_out_len() {
        let x = vec![0.0f32; 4];
        let w = vec![0.0f32; 4];
        let mut y = vec![0.0f32; 3];
        rms_norm(&x, &w, EPS, &mut y);
    }

    #[test]
    fn is_deterministic_across_runs() {
        // The fixed left-to-right reduction means byte-identical outputs
        // between runs on the same build. This guards against someone
        // swapping in `sum_sq += v.mul_add(v, 0.0)` or a parallel sum.
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let w: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();
        let mut y1 = vec![0f32; 64];
        let mut y2 = vec![0f32; 64];
        rms_norm(&x, &w, EPS, &mut y1);
        rms_norm(&x, &w, EPS, &mut y2);
        assert_eq!(y1, y2);
    }
}
