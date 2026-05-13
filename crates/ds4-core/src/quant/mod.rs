//! Weight dequantisation kernels.
//!
//! See rfcs/0002-forward-pass.md §3.2 + §3.8. Each dtype lives in its own
//! submodule with a test vector. PR #1 ships only Q8_0; IQ2_XXS, Q2_K,
//! IQ4_K, Q4_K, and F16 land in later PRs.

pub mod q8_0;
