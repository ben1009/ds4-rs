//! SwiGLU activation.
//!
//! See rfcs/0002-forward-pass.md §2. DS4 uses SwiGLU for both the shared
//! expert and the routed experts:
//!
//! ```text
//! out[i] = silu(gate[i]) * up[i]
//! silu(x) = x * sigmoid(x)
//! ```
//!
//! Routed experts additionally clamp gate (positive side only) and up
//! (both sides) to ±`DS4_SWIGLU_CLAMP_EXP` (10.0) before SwiGLU. That
//! clamping is handled by the caller / expert matmul path, not by the
//! base `swiglu` kernel here.
//!
//! Reference: `swiglu()` in antirez/ds4 ds4.c.

/// `silu(x) = x * sigmoid(x)` — the SiLU / Swish activation.
pub fn silu(x: f32) -> f32 {
    x * sigmoid_stable(x)
}

/// Element-wise SwiGLU: `out[i] = silu(gate[i]) * up[i]`.
///
/// Panics if the three slices do not have the same length.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(
        gate.len(),
        up.len(),
        "swiglu: gate len {} != up len {}",
        gate.len(),
        up.len(),
    );
    assert_eq!(
        gate.len(),
        out.len(),
        "swiglu: gate len {} != out len {}",
        gate.len(),
        out.len(),
    );
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// Numerically-stable sigmoid.
///
/// * `x >= 0` → `1 / (1 + exp(-x))`
/// * `x < 0`  → `exp(x) / (1 + exp(x))`
///
/// Reference: `sigmoid_stable()` in antirez/ds4 ds4.c.
pub fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0) + tol
    }

    #[test]
    fn sigmoid_stable_matches_direct() {
        let cases = [-5.0f32, -1.0, 0.0, 1.0, 5.0];
        for &x in &cases {
            let direct = 1.0 / (1.0 + (-x).exp());
            assert!(
                approx_eq(sigmoid_stable(x), direct, 1e-6),
                "sigmoid({x}): got {}, expected {}",
                sigmoid_stable(x),
                direct
            );
        }
    }

    #[test]
    fn sigmoid_stable_avoids_overflow() {
        // Direct 1/(1+exp(100)) would overflow; stable path does not.
        assert!(sigmoid_stable(100.0).is_finite());
        assert!(sigmoid_stable(-100.0).is_finite());
        assert!(sigmoid_stable(100.0) > 0.9999);
        assert!(sigmoid_stable(-100.0) < 1e-20);
    }

    #[test]
    fn silu_is_zero_at_zero() {
        assert_eq!(silu(0.0), 0.0);
    }

    #[test]
    fn silu_sign_follows_x() {
        assert!(silu(2.0) > 0.0);
        assert!(silu(-2.0) < 0.0);
    }

    #[test]
    fn swiglu_basic() {
        // gate = [0, 1, 2], up = [1, 1, 1]
        // silu(0) = 0, silu(1) ≈ 0.731, silu(2) ≈ 1.762
        let gate = vec![0.0f32, 1.0, 2.0];
        let up = vec![1.0f32, 1.0, 1.0];
        let mut out = vec![0.0f32; 3];
        swiglu(&gate, &up, &mut out);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 0.7310586).abs() < 1e-5);
        assert!((out[2] - 1.7615942).abs() < 1e-5);
    }

    #[test]
    fn swiglu_up_scales() {
        // gate = [1, 1], up = [2, 3]
        let gate = vec![1.0f32, 1.0];
        let up = vec![2.0f32, 3.0];
        let mut out = vec![0.0f32; 2];
        swiglu(&gate, &up, &mut out);
        let s = silu(1.0);
        assert!((out[0] - s * 2.0).abs() < 1e-6);
        assert!((out[1] - s * 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "gate len")]
    fn rejects_mismatched_gate_up() {
        swiglu(&[1.0; 4], &[1.0; 3], &mut [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "gate len")]
    fn rejects_mismatched_out() {
        swiglu(&[1.0; 4], &[1.0; 4], &mut [0.0; 3]);
    }

    #[test]
    fn is_deterministic_across_runs() {
        let gate: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1 - 3.2).sin()).collect();
        let up: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.13 + 1.1).cos()).collect();
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        swiglu(&gate, &up, &mut a);
        swiglu(&gate, &up, &mut b);
        assert_eq!(a, b, "swiglu must be deterministic");
    }
}
