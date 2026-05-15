//! Forward-pass ops.
//!
//! See rfcs/0002-forward-pass.md §2 / §3.8. This module contains the current
//! Phase 1 CPU reference op helpers: matmul, RMSNorm, RoPE, softmax, SwiGLU,
//! and HC split/mix support.

pub mod hc;
pub mod matmul;
pub mod norm;
pub mod rope;
pub mod softmax;
pub mod swiglu;
