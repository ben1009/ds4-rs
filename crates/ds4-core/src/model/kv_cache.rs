//! MLA latent KV cache.
//!
//! See rfcs/0002-forward-pass.md §3.5. Instead of materialising K/V per head,
//! DS4 Flash caches a 512-dim latent vector (`kv_latent`) plus a 64-dim
//! decoupled RoPE key (`k_pe`) per token per layer. Up-projection happens
//! inside the attention kernel.
//!
//! Per-token footprint (f32): 512 + 64 = 576 floats = 2.25 KiB.
//! For ctx=4096 and 43 layers: ~396 MiB.
//!
//! The cache is pre-allocated to `[n_layer, ctx_size, dim]` at session creation
//! and written via O(1) slice mutations. No reallocation during generation.

/// In-memory MLA KV cache.
pub struct KvCache {
    /// Flat buffer: `[n_layer, ctx_size, 512]` in row-major order.
    latent: Vec<f32>,
    /// Flat buffer: `[n_layer, ctx_size, 64]` in row-major order.
    k_pe: Vec<f32>,
    /// Number of layers.
    n_layer: usize,
    /// Maximum context length (allocated capacity).
    ctx_size: usize,
    /// Watermark: number of tokens currently stored.
    pos: usize,
}

impl KvCache {
    /// Pre-allocate the cache to full `[n_layer, ctx_size, dim]`.
    pub fn new(n_layer: usize, ctx_size: usize) -> Self {
        let latent_len = n_layer * ctx_size * 512;
        let k_pe_len = n_layer * ctx_size * 64;
        tracing::info!(
            "KvCache: {n_layer} layers × {ctx_size} ctx = {} MiB",
            (latent_len + k_pe_len) * 4 / 1024 / 1024
        );
        Self {
            latent: vec![0.0f32; latent_len],
            k_pe: vec![0.0f32; k_pe_len],
            n_layer,
            ctx_size,
            pos: 0,
        }
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

    /// Write a 512-dim latent vector for `(layer, pos)`.
    ///
    /// Panics if `data.len() != 512` or `pos >= ctx_size`.
    pub fn write_latent(&mut self, layer: usize, pos: usize, data: &[f32]) {
        assert_eq!(data.len(), 512, "write_latent: expected 512 dims");
        assert!(
            layer < self.n_layer,
            "write_latent: layer {layer} >= {}",
            self.n_layer
        );
        assert!(
            pos < self.ctx_size,
            "write_latent: pos {pos} >= {}",
            self.ctx_size
        );
        let off = (layer * self.ctx_size + pos) * 512;
        self.latent[off..off + 512].copy_from_slice(data);
    }

    /// Write a 64-dim decoupled RoPE key for `(layer, pos)`.
    ///
    /// Panics if `data.len() != 64` or `pos >= ctx_size`.
    pub fn write_k_pe(&mut self, layer: usize, pos: usize, data: &[f32]) {
        assert_eq!(data.len(), 64, "write_k_pe: expected 64 dims");
        assert!(
            layer < self.n_layer,
            "write_k_pe: layer {layer} >= {}",
            self.n_layer
        );
        assert!(
            pos < self.ctx_size,
            "write_k_pe: pos {pos} >= {}",
            self.ctx_size
        );
        let off = (layer * self.ctx_size + pos) * 64;
        self.k_pe[off..off + 64].copy_from_slice(data);
    }

    /// Read the 512-dim latent vector for `(layer, pos)`.
    ///
    /// Panics if `pos >= ctx_size`.
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
        let off = (layer * self.ctx_size + pos) * 512;
        &self.latent[off..off + 512]
    }

    /// Read the 64-dim decoupled RoPE key for `(layer, pos)`.
    ///
    /// Panics if `pos >= ctx_size`.
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
        let off = (layer * self.ctx_size + pos) * 64;
        &self.k_pe[off..off + 64]
    }

    /// Return a slice of all latent vectors for `layer` up to `len` tokens.
    /// Shape is effectively `[len, 512]` contiguous.
    pub fn latent_layer_prefix(&self, layer: usize, len: usize) -> &[f32] {
        assert!(layer < self.n_layer);
        assert!(len <= self.ctx_size);
        let off = layer * self.ctx_size * 512;
        &self.latent[off..off + len * 512]
    }

    /// Return a slice of all k_pe vectors for `layer` up to `len` tokens.
    /// Shape is effectively `[len, 64]` contiguous.
    pub fn k_pe_layer_prefix(&self, layer: usize, len: usize) -> &[f32] {
        assert!(layer < self.n_layer);
        assert!(len <= self.ctx_size);
        let off = layer * self.ctx_size * 64;
        &self.k_pe[off..off + len * 64]
    }

    /// Advance the position watermark.
    ///
    /// Panics if `new_pos > ctx_size`.
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
        let cache = KvCache::new(2, 8);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.n_layer(), 2);
        assert_eq!(cache.ctx_size(), 8);
        assert!(cache.is_empty());
    }

    #[test]
    fn write_and_read_latent() {
        let mut cache = KvCache::new(2, 8);
        let data: Vec<f32> = (0..512).map(|i| i as f32).collect();
        cache.write_latent(1, 3, &data);
        let read = cache.read_latent(1, 3);
        assert_eq!(read, data.as_slice());
        // Other positions should still be zero.
        assert!(cache.read_latent(0, 3).iter().all(|&v| v == 0.0));
        assert!(cache.read_latent(1, 2).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn write_and_read_k_pe() {
        let mut cache = KvCache::new(2, 8);
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        cache.write_k_pe(0, 7, &data);
        let read = cache.read_k_pe(0, 7);
        assert_eq!(read, data.as_slice());
    }

    #[test]
    fn layer_prefix_slices() {
        let mut cache = KvCache::new(1, 4);
        for pos in 0..4 {
            let d: Vec<f32> = (0..512).map(|i| (pos * 1000 + i) as f32).collect();
            cache.write_latent(0, pos, &d);
        }
        let prefix = cache.latent_layer_prefix(0, 3);
        assert_eq!(prefix.len(), 3 * 512);
        // First element of position 2's vector.
        assert_eq!(prefix[2 * 512], 2000.0);
    }

    #[test]
    #[should_panic(expected = "write_latent: expected 512 dims")]
    fn rejects_wrong_latent_size() {
        let mut cache = KvCache::new(1, 4);
        cache.write_latent(0, 0, &[1.0; 100]);
    }

    #[test]
    fn set_pos_watermark() {
        let mut cache = KvCache::new(1, 4);
        cache.set_pos(3);
        assert_eq!(cache.len(), 3);
    }
}
