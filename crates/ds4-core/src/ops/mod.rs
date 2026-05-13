//! Forward-pass ops.
//!
//! See rfcs/0002-forward-pass.md §2 / §3.8. PR #1 ships only matmul; RMSNorm,
//! RoPE, softmax, SwiGLU, and HC land in later PRs.

pub mod matmul;
