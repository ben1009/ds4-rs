//! Forward-pass ops.
//!
//! See rfcs/0002-forward-pass.md §2 / §3.8. PR #1 shipped matmul; PR #2
//! adds RMSNorm. Remaining ops (RoPE, softmax, SwiGLU, HC) land in later
//! PRs per the commit roadmap.

pub mod matmul;
pub mod norm;
