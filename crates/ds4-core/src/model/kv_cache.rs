//! MLA latent KV cache.
//!
//! See rfcs/0002-forward-pass.md §3.5 and antirez/ds4 ds4.c. The KV down-
//! projection produces a single 512-dim row per token (DS4_N_HEAD_DIM),
//! split into:
//!   * 448-dim kv_latent (the "nope" part — no positional encoding)
//!   * 64-dim  k_pe      (the decoupled RoPE key)
//!
//! Up-projection happens inside the attention kernel.
//!
//! Per-token footprint (f32): 448 + 64 = 512 floats = 2 KiB.
//! For ctx=4096 and 43 layers: ~352 MiB.
//!
//! The cache is pre-allocated to `[n_layer, ctx_size, dim]` at session creation
//! and written via O(1) slice mutations. No reallocation during generation.

use anyhow::{Result, bail};

/// Width of the non-positional ("nope") latent slice cached per token.
pub const KV_LATENT_DIM: usize = 448;
/// Width of the decoupled RoPE key cached per token.
pub const K_PE_DIM: usize = 64;

/// In-memory MLA KV cache.
pub struct KvCache {
    /// Flat buffer: `[n_layer, ctx_size, KV_LATENT_DIM]` in row-major order.
    latent: Vec<f32>,
    /// Flat buffer: `[n_layer, ctx_size, K_PE_DIM]` in row-major order.
    k_pe: Vec<f32>,
    /// Number of layers.
    n_layer: usize,
    /// Maximum context length (allocated capacity).
    ctx_size: usize,
    /// Watermark: number of tokens currently stored.
    pos: usize,
}

/// Compute `(layer * ctx_size + pos) * dim` with overflow checks.
fn checked_offset(layer: usize, ctx_size: usize, pos: usize, dim: usize) -> Result<usize> {
    layer
        .checked_mul(ctx_size)
        .and_then(|v| v.checked_add(pos))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("KvCache: offset overflow"))
}

impl KvCache {
    /// Pre-allocate the cache to full `[n_layer, ctx_size, dim]`.
    pub fn new(n_layer: usize, ctx_size: usize) -> Result<Self> {
        let latent_len = n_layer
            .checked_mul(ctx_size)
            .and_then(|v| v.checked_mul(KV_LATENT_DIM))
            .ok_or_else(|| anyhow::anyhow!("KvCache: latent buffer length overflow"))?;
        let k_pe_len = n_layer
            .checked_mul(ctx_size)
            .and_then(|v| v.checked_mul(K_PE_DIM))
            .ok_or_else(|| anyhow::anyhow!("KvCache: k_pe buffer length overflow"))?;
        tracing::info!(
            "KvCache: {n_layer} layers × {ctx_size} ctx = {} MiB",
            (latent_len + k_pe_len) * 4 / 1024 / 1024
        );
        Ok(Self {
            latent: vec![0.0f32; latent_len],
            k_pe: vec![0.0f32; k_pe_len],
            n_layer,
            ctx_size,
            pos: 0,
        })
    }

    /// Current number of cached tokens.
    pub fn len(&self) -> usize {
        self.pos
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    /// Maximum capacity in tokens.
    pub fn ctx_size(&self) -> usize {
        self.ctx_size
    }

    /// Number of layers.
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// Write the kv_latent (`KV_LATENT_DIM`) slice for `(layer, pos)`.
    pub fn write_latent(&mut self, layer: usize, pos: usize, data: &[f32]) -> Result<()> {
        if data.len() != KV_LATENT_DIM {
            bail!(
                "write_latent: expected {KV_LATENT_DIM} dims, got {}",
                data.len()
            );
        }
        if layer >= self.n_layer {
            bail!("write_latent: layer {layer} >= {}", self.n_layer);
        }
        if pos >= self.ctx_size {
            bail!(
                "write_latent: context overflow — pos {pos} >= ctx_size {}",
                self.ctx_size
            );
        }
        let off = checked_offset(layer, self.ctx_size, pos, KV_LATENT_DIM)?;
        self.latent[off..off + KV_LATENT_DIM].copy_from_slice(data);
        Ok(())
    }

    /// Write the decoupled RoPE key (`K_PE_DIM`) for `(layer, pos)`.
    pub fn write_k_pe(&mut self, layer: usize, pos: usize, data: &[f32]) -> Result<()> {
        if data.len() != K_PE_DIM {
            bail!("write_k_pe: expected {K_PE_DIM} dims, got {}", data.len());
        }
        if layer >= self.n_layer {
            bail!("write_k_pe: layer {layer} >= {}", self.n_layer);
        }
        if pos >= self.ctx_size {
            bail!(
                "write_k_pe: context overflow — pos {pos} >= ctx_size {}",
                self.ctx_size
            );
        }
        let off = checked_offset(layer, self.ctx_size, pos, K_PE_DIM)?;
        self.k_pe[off..off + K_PE_DIM].copy_from_slice(data);
        Ok(())
    }

    /// Read the kv_latent slice for `(layer, pos)`. Read paths trust the watermark.
    pub fn read_latent(&self, layer: usize, pos: usize) -> &[f32] {
        assert!(
            layer < self.n_layer,
            "read_latent: layer {layer} >= {}",
            self.n_layer
        );
        assert!(
            pos < self.ctx_size,
            "read_latent: pos {pos} >= {}",
            self.ctx_size
        );
        let off = checked_offset(layer, self.ctx_size, pos, KV_LATENT_DIM)
            .expect("KvCache: latent offset overflow (bounds already checked)");
        &self.latent[off..off + KV_LATENT_DIM]
    }

    /// Read the decoupled RoPE key for `(layer, pos)`.
    pub fn read_k_pe(&self, layer: usize, pos: usize) -> &[f32] {
        assert!(
            layer < self.n_layer,
            "read_k_pe: layer {layer} >= {}",
            self.n_layer
        );
        assert!(
            pos < self.ctx_size,
            "read_k_pe: pos {pos} >= {}",
            self.ctx_size
        );
        let off = checked_offset(layer, self.ctx_size, pos, K_PE_DIM)
            .expect("KvCache: k_pe offset overflow (bounds already checked)");
        &self.k_pe[off..off + K_PE_DIM]
    }

    /// Return a slice of all latent vectors for `layer` up to `len` tokens.
    pub fn latent_layer_prefix(&self, layer: usize, len: usize) -> &[f32] {
        assert!(layer < self.n_layer);
        assert!(len <= self.ctx_size);
        let off = layer
            .checked_mul(self.ctx_size)
            .and_then(|v| v.checked_mul(KV_LATENT_DIM))
            .expect("KvCache: latent prefix offset overflow");
        let span = len
            .checked_mul(KV_LATENT_DIM)
            .expect("KvCache: latent prefix length overflow");
        let end = off
            .checked_add(span)
            .expect("KvCache: latent prefix end overflow");
        &self.latent[off..end]
    }

    /// Return a slice of all k_pe vectors for `layer` up to `len` tokens.
    pub fn k_pe_layer_prefix(&self, layer: usize, len: usize) -> &[f32] {
        assert!(layer < self.n_layer);
        assert!(len <= self.ctx_size);
        let off = layer
            .checked_mul(self.ctx_size)
            .and_then(|v| v.checked_mul(K_PE_DIM))
            .expect("KvCache: k_pe prefix offset overflow");
        let span = len
            .checked_mul(K_PE_DIM)
            .expect("KvCache: k_pe prefix length overflow");
        let end = off
            .checked_add(span)
            .expect("KvCache: k_pe prefix end overflow");
        &self.k_pe[off..end]
    }

    /// Advance the position watermark.
    pub fn set_pos(&mut self, new_pos: usize) {
        assert!(
            new_pos <= self.ctx_size,
            "set_pos: {new_pos} > ctx_size {}",
            self.ctx_size
        );
        self.pos = new_pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zeros_and_shape() {
        let cache = KvCache::new(2, 8).unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.n_layer(), 2);
        assert_eq!(cache.ctx_size(), 8);
        assert!(cache.is_empty());
    }

    #[test]
    fn write_and_read_latent() {
        let mut cache = KvCache::new(2, 8).unwrap();
        let data: Vec<f32> = (0..KV_LATENT_DIM).map(|i| i as f32).collect();
        cache.write_latent(1, 3, &data).unwrap();
        let read = cache.read_latent(1, 3);
        assert_eq!(read, data.as_slice());
        // Other positions should still be zero.
        assert!(cache.read_latent(0, 3).iter().all(|&v| v == 0.0));
        assert!(cache.read_latent(1, 2).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn write_and_read_k_pe() {
        let mut cache = KvCache::new(2, 8).unwrap();
        let data: Vec<f32> = (0..K_PE_DIM).map(|i| i as f32 * 0.1).collect();
        cache.write_k_pe(0, 7, &data).unwrap();
        let read = cache.read_k_pe(0, 7);
        assert_eq!(read, data.as_slice());
    }

    #[test]
    fn layer_prefix_slices() {
        let mut cache = KvCache::new(1, 4).unwrap();
        for pos in 0..4 {
            let d: Vec<f32> = (0..KV_LATENT_DIM)
                .map(|i| (pos * 1000 + i) as f32)
                .collect();
            cache.write_latent(0, pos, &d).unwrap();
        }
        let prefix = cache.latent_layer_prefix(0, 3);
        assert_eq!(prefix.len(), 3 * KV_LATENT_DIM);
        // First element of position 2's vector.
        assert_eq!(prefix[2 * KV_LATENT_DIM], 2000.0);
    }

    #[test]
    fn rejects_wrong_latent_size() {
        let mut cache = KvCache::new(1, 4).unwrap();
        let err = cache.write_latent(0, 0, &[1.0; 100]).unwrap_err();
        assert!(err.to_string().contains("expected"));
    }

    #[test]
    fn rejects_pos_overflow() {
        let mut cache = KvCache::new(1, 4).unwrap();
        let data = vec![0.0f32; KV_LATENT_DIM];
        let err = cache.write_latent(0, 4, &data).unwrap_err();
        assert!(err.to_string().contains("context overflow"));
    }

    #[test]
    fn set_pos_watermark() {
        let mut cache = KvCache::new(1, 4).unwrap();
        cache.set_pos(3);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn write_at_capacity_boundary_ok() {
        let mut cache = KvCache::new(1, 4).unwrap();
        let lat = vec![1.0f32; KV_LATENT_DIM];
        let pe = vec![2.0f32; K_PE_DIM];
        cache.write_latent(0, 3, &lat).unwrap();
        cache.write_k_pe(0, 3, &pe).unwrap();
        assert_eq!(cache.read_latent(0, 3), lat.as_slice());
        assert_eq!(cache.read_k_pe(0, 3), pe.as_slice());
    }

    #[test]
    fn write_layer_overflow_errors() {
        let mut cache = KvCache::new(2, 4).unwrap();
        let data = vec![0.0f32; KV_LATENT_DIM];
        let err = cache.write_latent(2, 0, &data).unwrap_err();
        assert!(err.to_string().contains("layer 2"));
    }

    #[test]
    fn rejects_wrong_k_pe_size() {
        let mut cache = KvCache::new(1, 2).unwrap();
        let err = cache.write_k_pe(0, 0, &[0.0; 1]).unwrap_err();
        assert!(err.to_string().contains("expected"));
    }

    #[test]
    fn k_pe_pos_overflow_errors() {
        let mut cache = KvCache::new(1, 2).unwrap();
        let data = vec![0.0f32; K_PE_DIM];
        let err = cache.write_k_pe(0, 2, &data).unwrap_err();
        assert!(err.to_string().contains("context overflow"));
    }

    #[test]
    fn k_pe_layer_overflow_errors() {
        let mut cache = KvCache::new(1, 2).unwrap();
        let data = vec![0.0f32; K_PE_DIM];
        let err = cache.write_k_pe(1, 0, &data).unwrap_err();
        assert!(err.to_string().contains("layer 1"));
    }

    #[test]
    fn multi_layer_isolation_latent() {
        let mut cache = KvCache::new(3, 4).unwrap();
        let a = vec![1.0f32; KV_LATENT_DIM];
        let b = vec![2.0f32; KV_LATENT_DIM];
        cache.write_latent(0, 1, &a).unwrap();
        cache.write_latent(2, 1, &b).unwrap();
        assert!(cache.read_latent(0, 1).iter().all(|&v| v == 1.0));
        assert!(cache.read_latent(1, 1).iter().all(|&v| v == 0.0));
        assert!(cache.read_latent(2, 1).iter().all(|&v| v == 2.0));
    }

    #[test]
    fn multi_layer_isolation_k_pe() {
        let mut cache = KvCache::new(3, 4).unwrap();
        let a = vec![5.0f32; K_PE_DIM];
        cache.write_k_pe(1, 2, &a).unwrap();
        assert!(cache.read_k_pe(0, 2).iter().all(|&v| v == 0.0));
        assert!(cache.read_k_pe(1, 2).iter().all(|&v| v == 5.0));
        assert!(cache.read_k_pe(2, 2).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn k_pe_layer_prefix_per_layer() {
        let mut cache = KvCache::new(2, 3).unwrap();
        for layer in 0..2 {
            for pos in 0..3 {
                let v: Vec<f32> = (0..K_PE_DIM)
                    .map(|i| (layer * 100 + pos * 10 + i) as f32)
                    .collect();
                cache.write_k_pe(layer, pos, &v).unwrap();
            }
        }
        let prefix0 = cache.k_pe_layer_prefix(0, 3);
        let prefix1 = cache.k_pe_layer_prefix(1, 3);
        assert_eq!(prefix0.len(), 3 * K_PE_DIM);
        assert_eq!(prefix1.len(), 3 * K_PE_DIM);
        assert_eq!(prefix0[0], 0.0);
        assert_eq!(prefix1[0], 100.0);
        assert_eq!(prefix1[K_PE_DIM], 110.0);
    }

    #[test]
    fn empty_prefix_when_len_zero() {
        let cache = KvCache::new(2, 4).unwrap();
        assert_eq!(cache.latent_layer_prefix(0, 0).len(), 0);
        assert_eq!(cache.k_pe_layer_prefix(0, 0).len(), 0);
    }

    #[test]
    fn set_pos_to_ctx_size_ok() {
        let mut cache = KvCache::new(1, 4).unwrap();
        cache.set_pos(4);
        assert_eq!(cache.len(), 4);
        assert!(!cache.is_empty());
    }

    #[test]
    #[should_panic]
    fn set_pos_above_ctx_size_panics() {
        let mut cache = KvCache::new(1, 4).unwrap();
        cache.set_pos(5);
    }

    #[test]
    #[should_panic]
    fn read_latent_layer_oob_panics() {
        let cache = KvCache::new(1, 2).unwrap();
        let _ = cache.read_latent(1, 0);
    }

    #[test]
    #[should_panic]
    fn read_k_pe_pos_oob_panics() {
        let cache = KvCache::new(1, 2).unwrap();
        let _ = cache.read_k_pe(0, 2);
    }

    #[test]
    fn round_trip_overwrite() {
        let mut cache = KvCache::new(1, 4).unwrap();
        let a = vec![1.0f32; KV_LATENT_DIM];
        let b = vec![7.0f32; KV_LATENT_DIM];
        cache.write_latent(0, 0, &a).unwrap();
        cache.write_latent(0, 0, &b).unwrap();
        assert_eq!(cache.read_latent(0, 0), b.as_slice());
    }
}
