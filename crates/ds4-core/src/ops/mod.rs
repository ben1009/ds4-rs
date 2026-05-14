//! Forward-pass ops.
//!
//! See rfcs/0002-forward-pass.md §2 / §3.8. PR #1 shipped matmul; PR #2
//! added RMSNorm; PR #3 adds RoPE. Remaining ops (softmax, SwiGLU, HC)
//! land in later PRs per the commit roadmap.

pub mod hc;
pub mod matmul;
pub mod norm;
pub mod rope;
pub mod softmax;
pub mod swiglu;
