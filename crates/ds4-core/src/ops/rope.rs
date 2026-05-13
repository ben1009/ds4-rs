//! Partial rotary position embedding with YaRN long-context scaling.
//!
//! See rfcs/0002-forward-pass.md §2 / §3.4. DS4 Flash applies RoPE only to
//! the *last* `n_rot = 64` dims of each 512-dim head (the first `n_nope =
//! 448` dims are identity). The forward pass rotates query/key tails before
//! attention; an inverse rotation is applied to the attention output before
//! the grouped output projection.
//!
//! YaRN parameters (from DS4 Flash):
//!
//! * `scale_factor = 16`
//! * `beta_fast = 32`, `beta_slow = 1`
//! * `orig_ctx = 65536`
//!
//! YaRN interpolates per-dim frequencies between "full extrapolation" (short
//! wavelength; exactly standard RoPE) and "full interpolation" (long
//! wavelength; scaled down by `scale_factor`) with a smooth ramp. Inside
//! the ramp, freq = interpolated * (1 - r) + extrapolated * r.
//!
//! References:
//! * `rope_tail_ext_inplace` in antirez/ds4 ds4.c
//! * `kernel_dsv4_rope_tail_f32` in ds4/metal/dsv4_rope.metal

/// Per-call RoPE parameters. Computing the per-dim frequency cache once per
/// sequence and reusing it across tokens keeps the cos/sin cost out of the
/// attention hot loop.
#[derive(Clone, Debug)]
pub struct RopeParams {
    /// RoPE rotation dim — must be even. DS4 uses 64.
    pub n_rot: usize,
    /// Base for the standard frequency schedule (`base^(-2i/n_rot)`). DS4
    /// attention uses 10_000; the KV compressor uses 160_000.
    pub base: f32,
    /// YaRN long-context settings. `None` disables YaRN and falls back to
    /// plain RoPE.
    pub yarn: Option<YarnParams>,
}

/// YaRN ramp parameters. `scale_factor > 1` enables scaling.
#[derive(Clone, Debug)]
pub struct YarnParams {
    pub scale_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// Model's original (pre-scaled) training context length.
    pub orig_ctx: f32,
    /// Optional attention-factor multiplier applied after rotation
    /// (YaRN recommends a small magnitude boost to compensate for lost
    /// entropy). `None` = 1.0.
    pub attn_factor: Option<f32>,
}

/// Per-dim frequency cache. `freq[i]` is the angular frequency for dim pair
/// `(2i, 2i+1)` — i.e. `n_rot / 2` entries. `attn_factor` is the YaRN post-
/// rotation multiplier (1.0 when YaRN is disabled or unset).
#[derive(Clone, Debug)]
pub struct RopeFreqs {
    pub freqs: Vec<f32>,
    pub attn_factor: f32,
}

impl RopeFreqs {
    /// Build the per-dim frequency cache once per sequence.
    pub fn new(params: &RopeParams) -> Self {
        assert!(
            params.n_rot > 0 && params.n_rot.is_multiple_of(2),
            "n_rot must be positive and even"
        );
        let n_half = params.n_rot / 2;
        let mut freqs = vec![0f32; n_half];
        for (i, f) in freqs.iter_mut().enumerate() {
            let exp = (2 * i) as f32 / params.n_rot as f32;
            *f = 1.0 / params.base.powf(exp);
        }
        let attn_factor = if let Some(yarn) = &params.yarn {
            apply_yarn_scaling(&mut freqs, params.n_rot, params.base, yarn);
            yarn.attn_factor.unwrap_or(1.0)
        } else {
            1.0
        };
        Self { freqs, attn_factor }
    }
}

/// Apply partial RoPE to the last `n_rot` dims of `head` at absolute
/// position `pos`. The first `head.len() - n_rot` dims are untouched.
///
/// Rotation direction: standard RoPE (forward). For the inverse rotation
/// applied to the attention output, see [`apply_rope_inverse`].
///
/// `head.len()` must be at least `freqs.freqs.len() * 2`.
pub fn apply_rope(head: &mut [f32], pos: usize, freqs: &RopeFreqs) {
    rotate_tail(head, pos, freqs, /* inverse= */ false);
}

/// Inverse of [`apply_rope`] — applied to the attention *output* before the
/// grouped output projection in DS4 (see RFC 0002 §3.4).
///
/// When `attn_factor != 1.0` the inverse applies `1/attn_factor` so
/// `forward` followed by `inverse` is exactly the identity.
pub fn apply_rope_inverse(head: &mut [f32], pos: usize, freqs: &RopeFreqs) {
    rotate_tail(head, pos, freqs, /* inverse= */ true);
}

fn rotate_tail(head: &mut [f32], pos: usize, freqs: &RopeFreqs, inverse: bool) {
    let n_rot = freqs.freqs.len() * 2;
    assert!(
        head.len() >= n_rot,
        "rope: head len {} < n_rot {n_rot}",
        head.len(),
    );
    let tail_start = head.len() - n_rot;
    let tail = &mut head[tail_start..];
    let pos = pos as f32;
    let (sign, factor) = if inverse {
        (-1.0, 1.0 / freqs.attn_factor)
    } else {
        (1.0, freqs.attn_factor)
    };

    for (chunk, &freq) in tail.chunks_exact_mut(2).zip(freqs.freqs.iter()) {
        let theta = pos * freq;
        let (sin_t, cos_t) = theta.sin_cos();
        let sin = sign * sin_t;
        let x0 = chunk[0];
        let x1 = chunk[1];
        chunk[0] = (x0 * cos_t - x1 * sin) * factor;
        chunk[1] = (x0 * sin + x1 * cos_t) * factor;
    }
}

// ------ YaRN frequency scaling ------------------------------------------

fn apply_yarn_scaling(freqs: &mut [f32], n_rot: usize, base: f32, yarn: &YarnParams) {
    // Low/high correction dims: the pair of dim indices where the YaRN ramp
    // transitions between full extrapolation and full interpolation. These
    // are computed from the model's orig_ctx and beta_fast/slow wavelength
    // thresholds — see the YaRN paper §3.3 and DS4 ds4.c.
    let low = yarn_correction_dim(yarn.beta_fast, n_rot, base, yarn.orig_ctx);
    let high = yarn_correction_dim(yarn.beta_slow, n_rot, base, yarn.orig_ctx);
    let low = low.floor();
    let high = high.ceil();
    // Guard against a degenerate ramp (low >= high can happen with extreme
    // settings); clamp to a minimum width of 1 so the ramp stays
    // well-defined without collapsing to a step function.
    let ramp_width = (high - low).max(1.0);

    let inv_scale = 1.0 / yarn.scale_factor;
    for (i, freq) in freqs.iter_mut().enumerate() {
        // Ramp grows with dim index. Low dims are high-frequency / short-
        // wavelength — those should stay extrapolated (standard RoPE).
        // High dims are low-frequency / long-wavelength and get interpolated
        // (scaled by 1/scale_factor) to fit the extended context.
        let r = ((i as f32 - low) / ramp_width).clamp(0.0, 1.0);
        let interp = *freq * inv_scale;
        let extrap = *freq;
        *freq = extrap * (1.0 - r) + interp * r;
    }
}

/// Solve `base^(2 * d / n_rot) = orig_ctx / (2π * n_rotations)` for `d`.
///
/// `n_rotations` is the wavelength threshold (`beta_fast` / `beta_slow`);
/// the returned `d` is the dim index where that wavelength-to-ctx ratio
/// crosses. Guards against `base = 1.0` (degenerate) by clamping the
/// denominator away from zero.
fn yarn_correction_dim(n_rotations: f32, n_rot: usize, base: f32, orig_ctx: f32) -> f32 {
    let numerator = (orig_ctx / (n_rotations * 2.0 * std::f32::consts::PI)).ln();
    let denominator = base.ln().max(1e-9);
    n_rot as f32 * numerator / (2.0 * denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_params(n_rot: usize, base: f32) -> RopeParams {
        RopeParams {
            n_rot,
            base,
            yarn: None,
        }
    }

    fn ds4_params() -> RopeParams {
        RopeParams {
            n_rot: 64,
            base: 10_000.0,
            yarn: Some(YarnParams {
                scale_factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                orig_ctx: 65_536.0,
                attn_factor: None,
            }),
        }
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0) + tol
    }

    #[test]
    fn pos_zero_is_identity() {
        // At position 0, theta = 0 => cos=1, sin=0 => no change to tail.
        let freqs = RopeFreqs::new(&plain_params(4, 10_000.0));
        let mut head = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let before = head.clone();
        apply_rope(&mut head, 0, &freqs);
        assert_eq!(head, before);
    }

    #[test]
    fn nope_prefix_untouched() {
        // head = [nope | rot], nope dims must survive any rotation at any pos.
        let freqs = RopeFreqs::new(&plain_params(4, 10_000.0));
        let mut head = vec![100.0, 200.0, 300.0, 400.0, 1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut head, 7, &freqs);
        assert_eq!(&head[..4], &[100.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        let freqs = RopeFreqs::new(&plain_params(8, 10_000.0));
        let orig: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let mut head = orig.clone();
        apply_rope(&mut head, 42, &freqs);
        apply_rope_inverse(&mut head, 42, &freqs);
        for (i, (&got, &want)) in head.iter().zip(&orig).enumerate() {
            assert!(
                approx_eq(got, want, 1e-6),
                "dim {i}: got {got}, want {want}",
            );
        }
    }

    #[test]
    fn preserves_magnitude_per_pair() {
        // Rotation is a unitary transform on each (x0, x1) pair, so
        // x0^2 + x1^2 must be preserved (modulo attn_factor=1.0).
        let freqs = RopeFreqs::new(&plain_params(64, 10_000.0));
        let orig: Vec<f32> = (0..64).map(|i| ((i % 7) as f32) - 3.0).collect();
        let mut head = orig.clone();
        apply_rope(&mut head, 13, &freqs);
        for i in 0..32 {
            let before = orig[2 * i].powi(2) + orig[2 * i + 1].powi(2);
            let after = head[2 * i].powi(2) + head[2 * i + 1].powi(2);
            assert!(
                (before - after).abs() < 1e-4,
                "pair {i}: |before - after| = {}",
                (before - after).abs(),
            );
        }
    }

    #[test]
    fn yarn_ramp_extrapolates_low_dims_and_interpolates_high_dims() {
        // YaRN keeps low-index / short-wavelength dims at standard RoPE
        // (extrapolated, ratio = 1.0) and scales high-index / long-wavelength
        // dims down toward 1/scale_factor. For DS4 params the ramp spans
        // roughly [18, 33], so dim 0 is fully extrapolated and dim 31 is
        // mostly interpolated.
        let plain = RopeFreqs::new(&plain_params(64, 10_000.0));
        let scaled = RopeFreqs::new(&ds4_params());
        assert_eq!(plain.freqs.len(), scaled.freqs.len());

        let n = plain.freqs.len();
        let ratio_low = scaled.freqs[0] / plain.freqs[0];
        let ratio_high = scaled.freqs[n - 1] / plain.freqs[n - 1];

        // Dim 0 should be fully extrapolated (ratio = 1.0 exactly).
        assert!(
            approx_eq(ratio_low, 1.0, 1e-6),
            "dim 0 ratio = {ratio_low}, expected 1.0 for full extrapolation",
        );
        // Last dim should be dominated by interpolation — ratio well below
        // the mid-point between 1 and 1/16.
        assert!(
            ratio_high < 0.5,
            "dim {} ratio = {ratio_high}, expected << 0.5 for near-interpolation",
            n - 1,
        );
        // And strictly greater than 1/16 (DS4 ramp doesn't reach full
        // interpolation inside the cached range).
        assert!(
            ratio_high > 1.0 / 16.0,
            "dim {} ratio = {ratio_high}, expected > 1/16",
            n - 1,
        );
    }

    #[test]
    fn yarn_attn_factor_scales_output() {
        let params = RopeParams {
            n_rot: 4,
            base: 10_000.0,
            yarn: Some(YarnParams {
                scale_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                orig_ctx: 65_536.0,
                attn_factor: Some(2.5),
            }),
        };
        let freqs = RopeFreqs::new(&params);
        let mut head = vec![1.0, 0.0, 0.0, 1.0];
        apply_rope(&mut head, 0, &freqs);
        // At pos=0 rotation is identity, so the only effect is the scalar
        // multiply by attn_factor.
        assert_eq!(head, vec![2.5, 0.0, 0.0, 2.5]);
    }

    #[test]
    #[should_panic(expected = "n_rot")]
    fn rejects_short_head() {
        let freqs = RopeFreqs::new(&plain_params(8, 10_000.0));
        let mut head = vec![0.0; 4];
        apply_rope(&mut head, 0, &freqs);
    }

    #[test]
    fn ds4_head_shape_works_end_to_end() {
        // 512-dim head, 64 rotated — the concrete DS4 Flash shape.
        let freqs = RopeFreqs::new(&ds4_params());
        let mut head = vec![0.5f32; 512];
        // Fill the tail with something nontrivial so rotation does work.
        for (i, v) in head[448..].iter_mut().enumerate() {
            *v = (i as f32) * 0.01 - 0.32;
        }
        let tail_before: Vec<f32> = head[448..].to_vec();
        apply_rope(&mut head, 100, &freqs);
        // Nope prefix untouched.
        assert!(head[..448].iter().all(|&v| v == 0.5));
        // Tail changed.
        assert_ne!(&head[448..], tail_before.as_slice());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "miri's f32 cos/sin intrinsics are intentionally non-deterministic"
    )]
    fn is_deterministic_across_runs() {
        let freqs = RopeFreqs::new(&ds4_params());
        let input: Vec<f32> = (0..512)
            .map(|i| ((i as f32) * 0.01).rem_euclid(1.0))
            .collect();
        let mut a = input.clone();
        let mut b = input.clone();
        apply_rope(&mut a, 255, &freqs);
        apply_rope(&mut b, 255, &freqs);
        assert_eq!(a, b);
    }
}
