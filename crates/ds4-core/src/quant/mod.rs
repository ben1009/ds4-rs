//! Weight dequantisation + activation pre-quantisation kernels.
//!
//! See rfcs/0002-forward-pass.md §3.2 + §3.8. Each dtype lives in its own
//! submodule with a test vector. PR #1 shipped Q8_0 dequant; PR #4 adds
//! Q8_K (f32 → Q8_K pre-quant for the IQ-quant matmuls landing next).
//! IQ2_XXS / Q2_K / IQ4_K / Q4_K / F16 land in later PRs.

pub mod q8_0;
pub mod q8_k;
