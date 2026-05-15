//! Weight dequantisation + activation pre-quantisation kernels.
//!
//! See rfcs/0002-forward-pass.md §3.2 + §3.8. Implemented dtypes cover Q8_0
//! and F16 dense weights, the routed-expert quants IQ2_XXS / Q2_K / Q4_K /
//! IQ4_XS / IQ4_NL, and Q8_K activation pre-quantisation. F16 weight matmul
//! support lives in `ops::matmul`.

pub mod iq2_xxs;
mod iq4_codebook;
pub mod iq4_nl;
pub mod iq4_xs;
pub mod q2_k;
pub mod q4_k;
pub mod q8_0;
pub mod q8_k;
