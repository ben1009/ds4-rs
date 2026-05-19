//! Per-layer raw KV ring buffer.
//!
//! Mirrors `ds4_layer_cache.raw_kv` plus `kv_cache_push_raw` in antirez/ds4
//! ds4.c (lines 6154-6470). Each layer keeps a fixed-capacity ring of the
//! most recent tokens' merged 512-dim KV rows: append on each step, evict
//! the oldest row once full (memmove down by one).
//!
//! Storage is one merged `HEAD_DIM = 512` row per token — the previous
//! `latent (448) + k_pe (64)` split has been collapsed to match the C
//! reference. The split sizes are retained as `pub const` so callers can
//! still slice the freshly-projected row before pushing it (RoPE only
//! rotates the last 64 dims).
//!
//! Per-token footprint (f32): 512 floats = 2 KiB.
//! For cap_raw=128 and 43 layers: ~11 MiB.

use anyhow::{Result, bail};

/// Width of one cached KV row. Matches `DS4_N_HEAD_DIM` in ds4.c.
pub const HEAD_DIM: usize = 512;
/// Sliding-window cap. Matches `DS4_N_SWA` in ds4.c.
pub const SWA: usize = 128;
/// Width of the non-positional ("nope") slice within a `HEAD_DIM` row.
pub const KV_LATENT_DIM: usize = 448;
/// Width of the decoupled RoPE key tail within a `HEAD_DIM` row.
pub const K_PE_DIM: usize = 64;

/// One layer's raw ring buffer. `[cap_raw, HEAD_DIM]` row-major with a
/// watermark `n_raw <= cap_raw`. On overflow the oldest row is evicted by
/// memmove (matches `kv_cache_push_raw` in ds4.c).
pub struct RawLayerCache {
    raw_kv: Vec<f32>,
    n_raw: usize,
    cap_raw: usize,
}

impl RawLayerCache {
    pub fn new(cap_raw: usize) -> Self {
        Self {
            raw_kv: vec![0.0f32; cap_raw * HEAD_DIM],
            n_raw: 0,
            cap_raw,
        }
    }

    /// Append a `HEAD_DIM`-wide row. When full, drops the oldest row and
    /// writes the new one at slot `cap_raw - 1`. Panics if `kv.len() != HEAD_DIM`.
    pub fn push(&mut self, kv: &[f32]) {
        assert_eq!(
            kv.len(),
            HEAD_DIM,
            "RawLayerCache::push: row width must be HEAD_DIM"
        );
        let slot = if self.n_raw < self.cap_raw {
            let s = self.n_raw;
            self.n_raw += 1;
            s
        } else {
            // memmove rows [1..cap_raw] down to [0..cap_raw-1]; new row at the tail.
            self.raw_kv
                .copy_within(HEAD_DIM..self.cap_raw * HEAD_DIM, 0);
            self.cap_raw - 1
        };
        let off = slot * HEAD_DIM;
        self.raw_kv[off..off + HEAD_DIM].copy_from_slice(kv);
    }

    /// Drop all rows. Buffer capacity preserved.
    pub fn clear(&mut self) {
        self.n_raw = 0;
    }

    pub fn n_raw(&self) -> usize {
        self.n_raw
    }

    pub fn cap_raw(&self) -> usize {
        self.cap_raw
    }

    /// Slice of the `n_raw` active rows: `[n_raw, HEAD_DIM]` row-major.
    pub fn rows(&self) -> &[f32] {
        &self.raw_kv[..self.n_raw * HEAD_DIM]
    }
}

/// Multi-layer KV cache: one `RawLayerCache` per transformer layer.
pub struct KvCache {
    layers: Vec<RawLayerCache>,
}

impl KvCache {
    /// `ctx_size` mirrors `ds4_default_raw_cap` in ds4.c:
    /// `cap_raw = min(SWA, ctx_size).max(1)`.
    pub fn new(n_layer: usize, ctx_size: usize) -> Result<Self> {
        let cap_raw = SWA.min(ctx_size).max(1);
        let total = n_layer
            .checked_mul(cap_raw)
            .and_then(|v| v.checked_mul(HEAD_DIM))
            .ok_or_else(|| anyhow::anyhow!("KvCache: buffer length overflow"))?;
        if total > isize::MAX as usize {
            bail!("KvCache: buffer too large");
        }
        tracing::info!(
            "KvCache: {n_layer} layers × cap_raw {cap_raw} × HEAD_DIM {HEAD_DIM} = {} MiB",
            total * 4 / 1024 / 1024
        );
        Ok(Self {
            layers: (0..n_layer).map(|_| RawLayerCache::new(cap_raw)).collect(),
        })
    }

    pub fn layer(&self, il: usize) -> &RawLayerCache {
        &self.layers[il]
    }

    pub fn layer_mut(&mut self, il: usize) -> &mut RawLayerCache {
        &mut self.layers[il]
    }

    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    pub fn cap_raw(&self) -> usize {
        self.layers.first().map_or(0, |l| l.cap_raw)
    }

    /// Clear every layer's ring (n_raw = 0 each). Preserves allocations.
    pub fn clear_all(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear();
        }
    }

    /// Capture the full ring state for every layer so a failed multi-step
    /// operation (e.g. mid-prefill error) can restore it.
    ///
    /// Snapshotting `n_raw` alone is not enough: once the ring wraps,
    /// pushes overwrite the buffer in place, so the only way to undo a
    /// partial run is to keep the bytes around. Allocates `n_layer *
    /// cap_raw * HEAD_DIM` floats — only meant for rollback paths.
    pub fn snapshot(&self) -> KvCacheSnapshot {
        KvCacheSnapshot {
            layers: self
                .layers
                .iter()
                .map(|l| (l.raw_kv.clone(), l.n_raw))
                .collect(),
        }
    }

    /// Restore a snapshot taken earlier with [`KvCache::snapshot`].
    /// Panics if the snapshot's shape doesn't match the cache.
    pub fn restore(&mut self, snap: KvCacheSnapshot) {
        assert_eq!(
            snap.layers.len(),
            self.layers.len(),
            "KvCache::restore: layer count mismatch"
        );
        for (l, (raw, n_raw)) in self.layers.iter_mut().zip(snap.layers) {
            assert_eq!(
                raw.len(),
                l.raw_kv.len(),
                "KvCache::restore: buffer size mismatch"
            );
            l.raw_kv = raw;
            l.n_raw = n_raw;
        }
    }
}

/// Owned snapshot of every layer's ring buffer + watermark, taken via
/// [`KvCache::snapshot`] and consumed by [`KvCache::restore`].
pub struct KvCacheSnapshot {
    layers: Vec<(Vec<f32>, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(seed: f32) -> Vec<f32> {
        (0..HEAD_DIM).map(|i| seed + i as f32 * 0.001).collect()
    }

    #[test]
    fn fresh_cache_is_empty() {
        let cache = KvCache::new(3, 64).unwrap();
        assert_eq!(cache.n_layer(), 3);
        assert_eq!(cache.cap_raw(), 64);
        for il in 0..3 {
            assert_eq!(cache.layer(il).n_raw(), 0);
            assert_eq!(cache.layer(il).cap_raw(), 64);
            assert!(cache.layer(il).rows().is_empty());
        }
    }

    #[test]
    fn cap_clamps_to_swa() {
        let cache = KvCache::new(1, 1024).unwrap();
        assert_eq!(cache.cap_raw(), SWA);
    }

    #[test]
    fn cap_clamps_to_ctx_when_below_swa() {
        let cache = KvCache::new(1, 32).unwrap();
        assert_eq!(cache.cap_raw(), 32);
    }

    #[test]
    fn cap_zero_ctx_floors_to_one() {
        let cache = KvCache::new(1, 0).unwrap();
        assert_eq!(cache.cap_raw(), 1);
    }

    #[test]
    fn push_under_cap_fills_in_order() {
        let mut layer = RawLayerCache::new(4);
        for i in 0..3 {
            layer.push(&row(i as f32));
        }
        assert_eq!(layer.n_raw(), 3);
        let rows = layer.rows();
        assert_eq!(rows.len(), 3 * HEAD_DIM);
        for i in 0..3 {
            let r = &rows[i * HEAD_DIM..(i + 1) * HEAD_DIM];
            assert_eq!(r, row(i as f32).as_slice());
        }
    }

    #[test]
    fn push_at_cap_evicts_oldest() {
        let mut layer = RawLayerCache::new(3);
        for i in 0..3 {
            layer.push(&row(i as f32));
        }
        // Now full: rows = [0, 1, 2]. Push row 3 -> rows shift to [1, 2, 3].
        layer.push(&row(3.0));
        assert_eq!(layer.n_raw(), 3);
        let rows = layer.rows();
        assert_eq!(&rows[0..HEAD_DIM], row(1.0).as_slice());
        assert_eq!(&rows[HEAD_DIM..2 * HEAD_DIM], row(2.0).as_slice());
        assert_eq!(&rows[2 * HEAD_DIM..3 * HEAD_DIM], row(3.0).as_slice());
    }

    #[test]
    fn push_repeatedly_past_cap_keeps_only_latest_window() {
        let mut layer = RawLayerCache::new(2);
        for i in 0..6 {
            layer.push(&row(i as f32));
        }
        let rows = layer.rows();
        assert_eq!(layer.n_raw(), 2);
        assert_eq!(&rows[0..HEAD_DIM], row(4.0).as_slice());
        assert_eq!(&rows[HEAD_DIM..2 * HEAD_DIM], row(5.0).as_slice());
    }

    #[test]
    fn cap_one_keeps_only_last_row() {
        let mut layer = RawLayerCache::new(1);
        layer.push(&row(0.0));
        layer.push(&row(1.0));
        layer.push(&row(2.0));
        assert_eq!(layer.n_raw(), 1);
        assert_eq!(layer.rows(), row(2.0).as_slice());
    }

    #[test]
    fn multi_layer_isolation() {
        let mut cache = KvCache::new(3, 8).unwrap();
        cache.layer_mut(0).push(&row(10.0));
        cache.layer_mut(2).push(&row(20.0));
        cache.layer_mut(2).push(&row(30.0));
        assert_eq!(cache.layer(0).n_raw(), 1);
        assert_eq!(cache.layer(1).n_raw(), 0);
        assert_eq!(cache.layer(2).n_raw(), 2);
        assert_eq!(&cache.layer(0).rows()[..HEAD_DIM], row(10.0).as_slice());
        assert_eq!(&cache.layer(2).rows()[..HEAD_DIM], row(20.0).as_slice());
        assert_eq!(
            &cache.layer(2).rows()[HEAD_DIM..2 * HEAD_DIM],
            row(30.0).as_slice()
        );
    }

    #[test]
    fn clear_resets_n_raw_without_realloc() {
        let mut layer = RawLayerCache::new(4);
        for i in 0..4 {
            layer.push(&row(i as f32));
        }
        let cap_before = layer.raw_kv.capacity();
        layer.clear();
        assert_eq!(layer.n_raw(), 0);
        assert_eq!(layer.cap_raw(), 4);
        assert!(layer.rows().is_empty());
        // Buffer not freed.
        assert_eq!(layer.raw_kv.capacity(), cap_before);
    }

    #[test]
    fn clear_all_zeros_each_layer() {
        let mut cache = KvCache::new(2, 8).unwrap();
        cache.layer_mut(0).push(&row(1.0));
        cache.layer_mut(1).push(&row(2.0));
        cache.layer_mut(1).push(&row(3.0));
        cache.clear_all();
        assert_eq!(cache.layer(0).n_raw(), 0);
        assert_eq!(cache.layer(1).n_raw(), 0);
    }

    #[test]
    fn push_after_clear_starts_at_slot_zero() {
        let mut layer = RawLayerCache::new(3);
        for i in 0..3 {
            layer.push(&row(i as f32));
        }
        layer.clear();
        layer.push(&row(99.0));
        assert_eq!(layer.n_raw(), 1);
        assert_eq!(layer.rows(), row(99.0).as_slice());
    }

    #[test]
    #[should_panic(expected = "row width must be HEAD_DIM")]
    fn push_wrong_width_panics() {
        let mut layer = RawLayerCache::new(2);
        layer.push(&[0.0f32; HEAD_DIM - 1]);
    }

    #[test]
    fn split_constants_match_head_dim() {
        assert_eq!(KV_LATENT_DIM + K_PE_DIM, HEAD_DIM);
    }

    #[test]
    fn snapshot_restore_round_trips_after_eviction() {
        // Push past cap so the ring evicts oldest rows. Snapshot mid-stream,
        // push more (overwriting cached bytes), then restore. The restored
        // contents must match the snapshot byte-for-byte — this is the
        // property `Session::prefill` rollback depends on.
        let mut cache = KvCache::new(2, 4).unwrap();
        for i in 0..6 {
            cache.layer_mut(0).push(&row(i as f32));
        }
        cache.layer_mut(1).push(&row(100.0));
        let snap = cache.snapshot();
        let saved_l0: Vec<f32> = cache.layer(0).rows().to_vec();
        let saved_l1: Vec<f32> = cache.layer(1).rows().to_vec();

        for i in 0..3 {
            cache.layer_mut(0).push(&row(50.0 + i as f32));
        }
        cache.layer_mut(1).push(&row(200.0));
        cache.layer_mut(1).push(&row(300.0));
        assert_ne!(cache.layer(0).rows(), saved_l0.as_slice());

        cache.restore(snap);
        assert_eq!(cache.layer(0).n_raw(), 4);
        assert_eq!(cache.layer(1).n_raw(), 1);
        assert_eq!(cache.layer(0).rows(), saved_l0.as_slice());
        assert_eq!(cache.layer(1).rows(), saved_l1.as_slice());
    }

    #[test]
    fn snapshot_of_fresh_cache_restores_empty() {
        let mut cache = KvCache::new(2, 4).unwrap();
        let snap = cache.snapshot();
        cache.layer_mut(0).push(&row(1.0));
        cache.layer_mut(1).push(&row(2.0));
        cache.restore(snap);
        assert_eq!(cache.layer(0).n_raw(), 0);
        assert_eq!(cache.layer(1).n_raw(), 0);
    }
}
