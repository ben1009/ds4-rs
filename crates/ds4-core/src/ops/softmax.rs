//! Softmax and router activation.
//!
//! See rfcs/0002-forward-pass.md §2 / §3.4. Two functions live here:
//!
//! * [`softmax`] — standard stable softmax used in attention and MoE routing.
//! * [`sqrt_softplus`] — the `sqrt(softplus(x))` gating function used by the DS4 router (see §3.4).

/// Stable softmax in-place.
///
/// `out[i] = exp(x[i] - max) / sum_j(exp(x[j] - max))`
///
/// Panics if `x.len() != out.len()`.
pub fn softmax(x: &[f32], out: &mut [f32]) {
    assert_eq!(
        x.len(),
        out.len(),
        "softmax: x len {} != out len {}",
        x.len(),
        out.len(),
    );
    if x.is_empty() {
        return;
    }

    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for i in 0..x.len() {
        out[i] = (x[i] - max).exp();
        sum += out[i];
    }

    let inv_sum = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv_sum;
    }
}

/// `sqrt(softplus(x))` — DS4 router activation.
///
/// Matches `softplus_stable` in antirez/ds4 exactly:
/// * `x > 20.0`  → `sqrt(x)`
/// * `x < -20.0` → `sqrt(exp(x))`
/// * otherwise   → `sqrt(ln(1 + exp(x)))`
pub fn sqrt_softplus(x: f32) -> f32 {
    softplus_stable(x).sqrt()
}

fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0) + tol
    }

    #[test]
    fn softmax_of_zeros_is_uniform() {
        let x = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        softmax(&x, &mut out);
        for &v in &out {
            assert!((v - 0.25).abs() < 1e-6, "expected 0.25, got {v}");
        }
    }

    #[test]
    fn softmax_preserves_ranking() {
        let x = vec![1.0f32, 3.0, 2.0, 0.0];
        let mut out = vec![0.0f32; 4];
        softmax(&x, &mut out);
        assert!(out[1] > out[2] && out[2] > out[0] && out[0] > out[3]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let mut out = vec![0.0f32; 64];
        softmax(&x, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn softmax_is_stable_for_large_values() {
        // Without the max-subtraction trick this would overflow to inf.
        let x = vec![100.0f32, 101.0, 102.0];
        let mut out = vec![0.0f32; 3];
        softmax(&x, &mut out);
        assert!(
            out.iter().all(|&v| v.is_finite()),
            "softmax overflow: got {:?}",
            out
        );
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_empty_is_noop() {
        let x: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        softmax(&x, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn softplus_stable_matches_piecewise_formula() {
        // Middle branch.
        let x = 0.5f32;
        let direct = (1.0 + x.exp()).ln();
        assert!((softplus_stable(x) - direct).abs() < 1e-6);

        // Large positive branch.
        let x = 25.0f32;
        assert_eq!(softplus_stable(x), x);

        // Large negative branch.
        let x = -25.0f32;
        assert!((softplus_stable(x) - x.exp()).abs() < 1e-6);
    }

    #[test]
    fn sqrt_softplus_is_finite_for_extremes() {
        assert!(sqrt_softplus(100.0).is_finite());
        assert!(sqrt_softplus(-100.0).is_finite());
        assert!(sqrt_softplus(-100.0) < 1e-20);
    }

    #[test]
    fn sqrt_softplus_monotonic() {
        let xs: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        for w in xs.windows(2) {
            assert!(
                sqrt_softplus(w[1]) > sqrt_softplus(w[0]),
                "monotonicity failed at {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn is_deterministic_across_runs() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.17 - 2.7).sin()).collect();
        let mut a = vec![0.0f32; 32];
        let mut b = vec![0.0f32; 32];
        softmax(&x, &mut a);
        softmax(&x, &mut b);
        assert_eq!(a, b, "softmax must be deterministic");
    }
}
