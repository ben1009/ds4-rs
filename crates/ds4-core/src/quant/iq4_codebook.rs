//! Shared codebook table for IQ4_NL and IQ4_XS.
//!
//! Both quantisation formats encode each weight as a 4-bit index into a fixed
//! 16-entry signed table. The table values are identical across IQ4_NL and
//! IQ4_XS — only the per-sub-block scaling and packing differ.
//!
//! Reference: `kvalues_iq4nl` in ggml-quants.c.

pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
