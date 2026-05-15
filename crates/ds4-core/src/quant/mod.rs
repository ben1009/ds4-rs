//! Weight dequantisation + activation pre-quantisation kernels.
//!
//! See rfcs/0002-forward-pass.md §3.2 + §3.8. Implemented dtypes currently
//! cover Q8_0 weights and Q8_K activation pre-quantisation. IQ2_XXS / Q2_K /
//! IQ4_K / Q4_K routed-expert kernels are the next numerical blockers. F16
//! weight matmul support lives in `ops::matmul`.

pub mod q8_0;
pub mod q8_k;
