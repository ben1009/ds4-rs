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
//!
//! Per-layer streaming compressor state lives here too (see
//! [`CompressorState`]). Layers with `ratio == 0` (the dense first two)
//! carry no state. Ratio-4 layers read these emitted rows in the
//! mixed-attention path.

use anyhow::{Result, bail};

use crate::config::{INDEXER_HEAD_DIM, layer_compress_ratio};

/// Width of one cached KV row. Matches `DS4_N_HEAD_DIM` in ds4.c.
pub const HEAD_DIM: usize = 512;
/// Sliding-window cap. Matches `DS4_N_SWA` in ds4.c.
pub const SWA: usize = 128;
/// Width of the non-positional ("nope") slice within a `HEAD_DIM` row.
pub const KV_LATENT_DIM: usize = 448;
/// Width of the decoupled RoPE key tail within a `HEAD_DIM` row.
pub const K_PE_DIM: usize = 64;

/// Sentinel "no logit" score used by the streaming compressor's softmax.
/// Matches `DS4_NEG_INF` in ds4.c (`-1e30`); the pool helper compares
/// against `NEG_INF * 0.5` to detect an all-empty column.
pub const NEG_INF: f32 = -1.0e30;

/// Per-layer state for the streaming compressor (attention path).
///
/// Mirrors the `attn_state_kv / attn_state_score / attn_comp_kv / n_comp`
/// quartet inside `ds4_layer_cache` in antirez/ds4 ds4.c (lines 6154-6470).
/// Layers with `ratio == 0` (the dense first two layers in DS4 Flash) have
/// no compressor; `KvCache::compressor(il)` returns `None` for them.
///
/// Layout summary (`ratio != 0`):
/// * `coff = 2 if ratio == 4 else 1` — extra "compressed lane" only the ratio-4 path uses, doubling
///   the per-row state width.
/// * `width = coff * HEAD_DIM` — per-row width of `state_kv`/`state_score`.
/// * `state_kv`/`state_score` are `[coff*ratio, width]` row-major. Score is initialised to
///   `NEG_INF` so the softmax in `compressor_pool` ignores slots until they receive a real score.
/// * `comp_kv` is the emitted ring of `[comp_cap, HEAD_DIM]` compressed rows; `n_comp` is the
///   watermark.
pub struct CompressorState {
    pub ratio: u32,
    pub comp_cap: usize,
    pub state_kv: Vec<f32>,
    pub state_score: Vec<f32>,
    pub comp_kv: Vec<f32>,
    pub n_comp: usize,
}

impl CompressorState {
    /// Build a fresh state for one layer. `ratio == 0` is rejected; the
    /// caller owns the dense vs. compressed branch and must not invoke this
    /// for dense layers.
    ///
    /// Returns `Err` if any of the buffer-size products overflow `usize`,
    /// so `KvCache::new` can propagate the failure instead of panicking
    /// inside `vec!` on a pathological `ctx_size`.
    pub fn new(ratio: u32, ctx_size: usize) -> Result<Self> {
        assert!(ratio != 0, "CompressorState::new called with ratio = 0");
        let coff = if ratio == 4 { 2 } else { 1usize };
        let width = coff
            .checked_mul(HEAD_DIM)
            .ok_or_else(|| anyhow::anyhow!("CompressorState: width overflow"))?;
        let rows = coff
            .checked_mul(ratio as usize)
            .ok_or_else(|| anyhow::anyhow!("CompressorState: row count overflow"))?;
        let state_len = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("CompressorState: state buffer length overflow"))?;
        // ds4.c: `comp_cap = ctx_size / ratio + 2`. The +2 absorbs the
        // partial-window edge cases (a long-running session can outrun the
        // simple ctx/ratio bound by one or two emitted rows).
        let comp_cap = (ctx_size / ratio as usize)
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("CompressorState: comp_cap overflow"))?;
        let comp_kv_len = comp_cap
            .checked_mul(HEAD_DIM)
            .ok_or_else(|| anyhow::anyhow!("CompressorState: comp_kv length overflow"))?;
        Ok(Self {
            ratio,
            comp_cap,
            state_kv: vec![0.0f32; state_len],
            state_score: vec![NEG_INF; state_len],
            comp_kv: vec![0.0f32; comp_kv_len],
            n_comp: 0,
        })
    }

    pub fn coff(&self) -> usize {
        if self.ratio == 4 { 2 } else { 1 }
    }

    /// Width of one row in `state_kv`/`state_score` (`coff * HEAD_DIM`).
    pub fn width(&self) -> usize {
        self.coff() * HEAD_DIM
    }

    /// Reset to the freshly-built shape: zero KV, scores back to `NEG_INF`,
    /// `n_comp = 0`. Mirrors what `ds4_session_invalidate` does to the
    /// per-layer compressor fields.
    pub fn clear(&mut self) {
        self.state_kv.fill(0.0);
        self.state_score.fill(NEG_INF);
        // Leave `comp_kv` allocation alone; n_comp = 0 makes its contents
        // unreachable. (The C reference doesn't zero comp_kv on invalidate
        // either.)
        self.n_comp = 0;
    }

    /// Append one `HEAD_DIM`-wide compressed row to the emitted ring.
    /// Errors if the cap would be exceeded — ds4.c calls `ds4_die` here, but
    /// we surface it as an error so the forward pass can roll back.
    pub fn push_comp(&mut self, row: &[f32]) -> Result<()> {
        if row.len() != HEAD_DIM {
            bail!(
                "CompressorState::push_comp: row width {} != HEAD_DIM {HEAD_DIM}",
                row.len(),
            );
        }
        if self.n_comp >= self.comp_cap {
            bail!(
                "CompressorState::push_comp: capacity {} exceeded",
                self.comp_cap,
            );
        }
        let off = self.n_comp * HEAD_DIM;
        self.comp_kv[off..off + HEAD_DIM].copy_from_slice(row);
        self.n_comp += 1;
        Ok(())
    }

    /// Slice of the `n_comp` emitted rows: `[n_comp, HEAD_DIM]` row-major.
    pub fn comp_rows(&self) -> &[f32] {
        &self.comp_kv[..self.n_comp * HEAD_DIM]
    }
}

/// Width of the indexer's head dim (`DS4_N_INDEXER_HEAD_DIM` in ds4.c).
/// Re-exported here for convenience; the source of truth is
/// [`crate::config::INDEXER_HEAD_DIM`].
pub const IDX_DIM: usize = INDEXER_HEAD_DIM as usize;

/// Per-layer state for the streaming **indexer** compressor (ratio-4 layers
/// only).
///
/// Mirrors the `index_state_kv / index_state_score / index_comp_kv /
/// n_index_comp` quartet inside `ds4_layer_cache` in antirez/ds4 ds4.c
/// (lines 6332-6470). Same shape rules as [`CompressorState`] but at width
/// `INDEXER_HEAD_DIM = 128` rather than `HEAD_DIM = 512`, and only created
/// for layers with `ratio == 4` (the dense first two layers and ratio-128
/// layers carry no indexer).
///
/// Layout summary:
/// * `coff = 2` (only ratio == 4 has an indexer; the C reference still computes coff for symmetry,
///   but it's always 2 here).
/// * `width = coff * IDX_DIM = 2 * 128 = 256` — per-row width of `state_kv`/`state_score`.
/// * `state_kv`/`state_score` are `[coff*ratio, width] = [8, 256]` row-major. Score is initialised
///   to `NEG_INF` so the softmax in the pool helper ignores slots until they receive a real score.
/// * `comp_kv` is the emitted ring of `[comp_cap, IDX_DIM]` rows; `n_comp` is the watermark.
///
/// The emitted indexer rows are consumed by the mixed-attention path in
/// ratio-4 layers.
pub struct IndexerState {
    pub comp_cap: usize,
    pub state_kv: Vec<f32>,
    pub state_score: Vec<f32>,
    pub comp_kv: Vec<f32>,
    pub n_comp: usize,
}

impl IndexerState {
    /// Build a fresh indexer state for one ratio-4 layer.
    ///
    /// `ctx_size` controls the emitted-ring capacity (`comp_cap = ctx_size /
    /// 4 + 2`, mirroring `kv_cache_alloc` in ds4.c).
    pub fn new(ctx_size: usize) -> Result<Self> {
        // Always ratio == 4 for the indexer (only ratio-4 layers carry one).
        let ratio: usize = 4;
        let coff: usize = 2;
        let width = coff
            .checked_mul(IDX_DIM)
            .ok_or_else(|| anyhow::anyhow!("IndexerState: width overflow"))?;
        let rows = coff
            .checked_mul(ratio)
            .ok_or_else(|| anyhow::anyhow!("IndexerState: row count overflow"))?;
        let state_len = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("IndexerState: state buffer length overflow"))?;
        let comp_cap = (ctx_size / ratio)
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("IndexerState: comp_cap overflow"))?;
        let comp_kv_len = comp_cap
            .checked_mul(IDX_DIM)
            .ok_or_else(|| anyhow::anyhow!("IndexerState: comp_kv length overflow"))?;
        Ok(Self {
            comp_cap,
            state_kv: vec![0.0f32; state_len],
            state_score: vec![NEG_INF; state_len],
            comp_kv: vec![0.0f32; comp_kv_len],
            n_comp: 0,
        })
    }

    /// `coff * IDX_DIM` — the per-row width of `state_kv`/`state_score`.
    /// Always `2 * 128 = 256` for the indexer (only ratio-4 layers have one).
    pub fn width(&self) -> usize {
        2 * IDX_DIM
    }

    /// Reset KV to zero, scores to `NEG_INF`, watermark to zero.
    pub fn clear(&mut self) {
        self.state_kv.fill(0.0);
        self.state_score.fill(NEG_INF);
        self.n_comp = 0;
    }

    /// Append one `IDX_DIM`-wide row to the emitted ring. Errors if `comp_cap`
    /// would be exceeded — ds4.c calls `ds4_die` here, but we surface it as
    /// an error so the forward pass can roll back.
    pub fn push_comp(&mut self, row: &[f32]) -> Result<()> {
        if row.len() != IDX_DIM {
            bail!(
                "IndexerState::push_comp: row width {} != IDX_DIM {IDX_DIM}",
                row.len(),
            );
        }
        if self.n_comp >= self.comp_cap {
            bail!(
                "IndexerState::push_comp: capacity {} exceeded",
                self.comp_cap,
            );
        }
        let off = self.n_comp * IDX_DIM;
        self.comp_kv[off..off + IDX_DIM].copy_from_slice(row);
        self.n_comp += 1;
        Ok(())
    }

    /// Slice of the `n_comp` emitted rows: `[n_comp, IDX_DIM]` row-major.
    pub fn comp_rows(&self) -> &[f32] {
        &self.comp_kv[..self.n_comp * IDX_DIM]
    }
}

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

/// Multi-layer KV cache: one `RawLayerCache` per transformer layer plus a
/// matching slot of optional `CompressorState` and (for ratio-4 layers) an
/// optional `IndexerState`. Dense layers (`ratio == 0`) and ratio-128 layers
/// store `None` in the indexer slot to make the "this layer skips the
/// indexer" case visible at the type level.
pub struct KvCache {
    layers: Vec<RawLayerCache>,
    compressors: Vec<Option<CompressorState>>,
    indexers: Vec<Option<IndexerState>>,
}

impl KvCache {
    /// `ctx_size` mirrors `ds4_default_raw_cap` in ds4.c:
    /// `cap_raw = min(SWA, ctx_size).max(1)`.
    ///
    /// Per-layer compressor state is sized from `layer_compress_ratio(il)`:
    /// the dense layers (`il < 2`) get `None`, the rest carry a
    /// [`CompressorState`] sized with the full unclamped `ctx_size` (the
    /// compressor's emitted ring is independent of the SWA-clamped raw ring).
    /// Per-layer indexer state is built only for ratio-4 layers.
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
        let compressors = (0..n_layer)
            .map(|il| {
                let ratio = layer_compress_ratio(il as u32);
                if ratio == 0 {
                    Ok(None)
                } else {
                    CompressorState::new(ratio, ctx_size).map(Some)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let indexers = (0..n_layer)
            .map(|il| {
                if layer_compress_ratio(il as u32) == 4 {
                    IndexerState::new(ctx_size).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers: (0..n_layer).map(|_| RawLayerCache::new(cap_raw)).collect(),
            compressors,
            indexers,
        })
    }

    pub fn layer(&self, il: usize) -> &RawLayerCache {
        &self.layers[il]
    }

    pub fn layer_mut(&mut self, il: usize) -> &mut RawLayerCache {
        &mut self.layers[il]
    }

    /// Borrow the streaming compressor for layer `il`. Dense layers return
    /// `None`. Mirrors the `ratio != 0` gating in ds4.c.
    pub fn compressor(&self, il: usize) -> Option<&CompressorState> {
        self.compressors[il].as_ref()
    }

    /// Mutable counterpart of [`KvCache::compressor`].
    pub fn compressor_mut(&mut self, il: usize) -> Option<&mut CompressorState> {
        self.compressors[il].as_mut()
    }

    /// Borrow the indexer compressor state for layer `il`. Only ratio-4
    /// layers carry one; everything else returns `None`.
    pub fn indexer(&self, il: usize) -> Option<&IndexerState> {
        self.indexers[il].as_ref()
    }

    /// Mutable counterpart of [`KvCache::indexer`].
    pub fn indexer_mut(&mut self, il: usize) -> Option<&mut IndexerState> {
        self.indexers[il].as_mut()
    }

    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    pub fn cap_raw(&self) -> usize {
        self.layers.first().map_or(0, |l| l.cap_raw)
    }

    /// Clear every layer's ring and compressor state. Preserves allocations.
    pub fn clear_all(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear();
        }
        for c in self.compressors.iter_mut().flatten() {
            c.clear();
        }
        for i in self.indexers.iter_mut().flatten() {
            i.clear();
        }
    }

    /// Capture the full ring state for every layer so a failed multi-step
    /// operation (e.g. mid-prefill error) can restore it.
    ///
    /// Snapshotting `n_raw` alone is not enough: once the ring wraps,
    /// pushes overwrite the buffer in place, so the only way to undo a
    /// partial run is to keep the bytes around. The compressor state is
    /// snapshotted with the same byte-for-byte semantics — `state_kv` and
    /// `state_score` rows can be overwritten in place by `compressor_decode_one`,
    /// and `comp_kv` slots can be reused if `n_comp` rolls back.
    ///
    /// Use [`KvCache::snapshot_into`] in hot paths so the destination buffer
    /// is reused across calls — that's a `cap_raw * HEAD_DIM * n_layer * 4`
    /// bytes save per token (~11 MiB at the 128/512/43 default), plus the
    /// per-layer compressor state.
    pub fn snapshot(&self) -> KvCacheSnapshot {
        let mut snap = KvCacheSnapshot::with_shape(self);
        self.snapshot_into(&mut snap);
        snap
    }

    /// Snapshot the cache into a pre-allocated destination, reusing its
    /// buffers. The destination must have been built with
    /// [`KvCacheSnapshot::with_shape`] for a cache of the same shape; an
    /// out-of-shape destination panics.
    pub fn snapshot_into(&self, snap: &mut KvCacheSnapshot) {
        assert_eq!(
            snap.layers.len(),
            self.layers.len(),
            "KvCache::snapshot_into: layer count mismatch"
        );
        assert_eq!(
            snap.compressors.len(),
            self.compressors.len(),
            "KvCache::snapshot_into: compressor count mismatch"
        );
        assert_eq!(
            snap.indexers.len(),
            self.indexers.len(),
            "KvCache::snapshot_into: indexer count mismatch"
        );
        for (dst, src) in snap.layers.iter_mut().zip(self.layers.iter()) {
            assert_eq!(
                dst.0.len(),
                src.raw_kv.len(),
                "KvCache::snapshot_into: buffer size mismatch"
            );
            dst.0.copy_from_slice(&src.raw_kv);
            dst.1 = src.n_raw;
        }
        for (dst, src) in snap.compressors.iter_mut().zip(self.compressors.iter()) {
            match (dst.as_mut(), src.as_ref()) {
                (Some(snap_state), Some(state)) => {
                    assert_eq!(
                        snap_state.state_kv.len(),
                        state.state_kv.len(),
                        "KvCache::snapshot_into: compressor state_kv size mismatch"
                    );
                    assert_eq!(
                        snap_state.state_score.len(),
                        state.state_score.len(),
                        "KvCache::snapshot_into: compressor state_score size mismatch"
                    );
                    assert_eq!(
                        snap_state.comp_kv.len(),
                        state.comp_kv.len(),
                        "KvCache::snapshot_into: comp_kv size mismatch"
                    );
                    snap_state.state_kv.copy_from_slice(&state.state_kv);
                    snap_state.state_score.copy_from_slice(&state.state_score);
                    snap_state.comp_kv.copy_from_slice(&state.comp_kv);
                    snap_state.n_comp = state.n_comp;
                }
                (None, None) => {}
                _ => panic!("KvCache::snapshot_into: compressor presence mismatch"),
            }
        }
        for (dst, src) in snap.indexers.iter_mut().zip(self.indexers.iter()) {
            match (dst.as_mut(), src.as_ref()) {
                (Some(snap_state), Some(state)) => {
                    assert_eq!(
                        snap_state.state_kv.len(),
                        state.state_kv.len(),
                        "KvCache::snapshot_into: indexer state_kv size mismatch"
                    );
                    assert_eq!(
                        snap_state.state_score.len(),
                        state.state_score.len(),
                        "KvCache::snapshot_into: indexer state_score size mismatch"
                    );
                    assert_eq!(
                        snap_state.comp_kv.len(),
                        state.comp_kv.len(),
                        "KvCache::snapshot_into: indexer comp_kv size mismatch"
                    );
                    snap_state.state_kv.copy_from_slice(&state.state_kv);
                    snap_state.state_score.copy_from_slice(&state.state_score);
                    snap_state.comp_kv.copy_from_slice(&state.comp_kv);
                    snap_state.n_comp = state.n_comp;
                }
                (None, None) => {}
                _ => panic!("KvCache::snapshot_into: indexer presence mismatch"),
            }
        }
    }

    /// Restore an in-place snapshot. Buffers are copied back into the cache;
    /// the snapshot is left intact for reuse.
    pub fn restore(&mut self, snap: &KvCacheSnapshot) {
        assert_eq!(
            snap.layers.len(),
            self.layers.len(),
            "KvCache::restore: layer count mismatch"
        );
        assert_eq!(
            snap.compressors.len(),
            self.compressors.len(),
            "KvCache::restore: compressor count mismatch"
        );
        assert_eq!(
            snap.indexers.len(),
            self.indexers.len(),
            "KvCache::restore: indexer count mismatch"
        );
        for (l, (raw, n_raw)) in self.layers.iter_mut().zip(snap.layers.iter()) {
            assert_eq!(
                raw.len(),
                l.raw_kv.len(),
                "KvCache::restore: buffer size mismatch"
            );
            l.raw_kv.copy_from_slice(raw);
            l.n_raw = *n_raw;
        }
        for (slot, snap_slot) in self.compressors.iter_mut().zip(snap.compressors.iter()) {
            match (slot.as_mut(), snap_slot.as_ref()) {
                (Some(state), Some(snap_state)) => {
                    assert_eq!(
                        snap_state.state_kv.len(),
                        state.state_kv.len(),
                        "KvCache::restore: compressor state_kv size mismatch"
                    );
                    assert_eq!(
                        snap_state.state_score.len(),
                        state.state_score.len(),
                        "KvCache::restore: compressor state_score size mismatch"
                    );
                    assert_eq!(
                        snap_state.comp_kv.len(),
                        state.comp_kv.len(),
                        "KvCache::restore: comp_kv size mismatch"
                    );
                    state.state_kv.copy_from_slice(&snap_state.state_kv);
                    state.state_score.copy_from_slice(&snap_state.state_score);
                    state.comp_kv.copy_from_slice(&snap_state.comp_kv);
                    state.n_comp = snap_state.n_comp;
                }
                (None, None) => {}
                _ => panic!("KvCache::restore: compressor presence mismatch"),
            }
        }
        for (slot, snap_slot) in self.indexers.iter_mut().zip(snap.indexers.iter()) {
            match (slot.as_mut(), snap_slot.as_ref()) {
                (Some(state), Some(snap_state)) => {
                    assert_eq!(
                        snap_state.state_kv.len(),
                        state.state_kv.len(),
                        "KvCache::restore: indexer state_kv size mismatch"
                    );
                    assert_eq!(
                        snap_state.state_score.len(),
                        state.state_score.len(),
                        "KvCache::restore: indexer state_score size mismatch"
                    );
                    assert_eq!(
                        snap_state.comp_kv.len(),
                        state.comp_kv.len(),
                        "KvCache::restore: indexer comp_kv size mismatch"
                    );
                    state.state_kv.copy_from_slice(&snap_state.state_kv);
                    state.state_score.copy_from_slice(&snap_state.state_score);
                    state.comp_kv.copy_from_slice(&snap_state.comp_kv);
                    state.n_comp = snap_state.n_comp;
                }
                (None, None) => {}
                _ => panic!("KvCache::restore: indexer presence mismatch"),
            }
        }
    }
}

/// Reusable rollback snapshot of every layer's ring buffer + watermark plus
/// per-layer compressor and indexer state.
///
/// Build once via [`KvCacheSnapshot::with_shape`] and reuse across calls
/// with [`KvCache::snapshot_into`] / [`KvCache::restore`] to avoid the
/// per-token allocation that `KvCache::snapshot` would otherwise pay.
pub struct KvCacheSnapshot {
    layers: Vec<(Vec<f32>, usize)>,
    compressors: Vec<Option<CompressorSnapshot>>,
    indexers: Vec<Option<IndexerSnapshot>>,
}

impl KvCacheSnapshot {
    /// Allocate a snapshot sized to `cache`. Initial contents undefined; a
    /// fresh snapshot must be filled with [`KvCache::snapshot_into`] before
    /// being used as a restore source.
    pub fn with_shape(cache: &KvCache) -> Self {
        Self {
            layers: cache
                .layers
                .iter()
                .map(|l| (vec![0.0f32; l.raw_kv.len()], 0usize))
                .collect(),
            compressors: cache
                .compressors
                .iter()
                .map(|c| {
                    c.as_ref().map(|s| CompressorSnapshot {
                        state_kv: vec![0.0f32; s.state_kv.len()],
                        state_score: vec![NEG_INF; s.state_score.len()],
                        comp_kv: vec![0.0f32; s.comp_kv.len()],
                        n_comp: 0,
                    })
                })
                .collect(),
            indexers: cache
                .indexers
                .iter()
                .map(|c| {
                    c.as_ref().map(|s| IndexerSnapshot {
                        state_kv: vec![0.0f32; s.state_kv.len()],
                        state_score: vec![NEG_INF; s.state_score.len()],
                        comp_kv: vec![0.0f32; s.comp_kv.len()],
                        n_comp: 0,
                    })
                })
                .collect(),
        }
    }
}

struct CompressorSnapshot {
    state_kv: Vec<f32>,
    state_score: Vec<f32>,
    comp_kv: Vec<f32>,
    n_comp: usize,
}

struct IndexerSnapshot {
    state_kv: Vec<f32>,
    state_score: Vec<f32>,
    comp_kv: Vec<f32>,
    n_comp: usize,
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

        cache.restore(&snap);
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
        cache.restore(&snap);
        assert_eq!(cache.layer(0).n_raw(), 0);
        assert_eq!(cache.layer(1).n_raw(), 0);
    }

    // ---------------------------------------------------------------------
    // CompressorState
    // ---------------------------------------------------------------------

    #[test]
    fn compressor_state_ratio_4_shapes() {
        let s = CompressorState::new(4, 1024).unwrap();
        assert_eq!(s.ratio, 4);
        assert_eq!(s.coff(), 2);
        assert_eq!(s.width(), 2 * HEAD_DIM);
        // rows = coff * ratio = 2 * 4 = 8.
        assert_eq!(s.state_kv.len(), 8 * 2 * HEAD_DIM);
        assert_eq!(s.state_score.len(), 8 * 2 * HEAD_DIM);
        // comp_cap = ctx / ratio + 2 = 1024 / 4 + 2 = 258.
        assert_eq!(s.comp_cap, 258);
        assert_eq!(s.comp_kv.len(), 258 * HEAD_DIM);
        assert_eq!(s.n_comp, 0);
        // state_score must be initialised to NEG_INF, not zero — the pool
        // softmax keys off this.
        assert!(s.state_score.iter().all(|&v| v == NEG_INF));
        assert!(s.state_kv.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn compressor_state_ratio_128_shapes() {
        let s = CompressorState::new(128, 8192).unwrap();
        assert_eq!(s.coff(), 1);
        assert_eq!(s.width(), HEAD_DIM);
        assert_eq!(s.state_kv.len(), 128 * HEAD_DIM);
        assert_eq!(s.state_score.len(), 128 * HEAD_DIM);
        // comp_cap = 8192 / 128 + 2 = 66.
        assert_eq!(s.comp_cap, 66);
    }

    #[test]
    fn compressor_state_clear_resets_score_and_n_comp() {
        let mut s = CompressorState::new(4, 64).unwrap();
        for v in s.state_kv.iter_mut() {
            *v = 1.0;
        }
        for v in s.state_score.iter_mut() {
            *v = 0.5;
        }
        s.push_comp(&[2.0; HEAD_DIM]).unwrap();
        s.push_comp(&[3.0; HEAD_DIM]).unwrap();
        assert_eq!(s.n_comp, 2);

        s.clear();
        assert_eq!(s.n_comp, 0);
        assert!(s.state_kv.iter().all(|&v| v == 0.0));
        assert!(s.state_score.iter().all(|&v| v == NEG_INF));
    }

    #[test]
    fn compressor_state_push_comp_grows_n_comp() {
        let mut s = CompressorState::new(128, 1024).unwrap();
        let row_a: Vec<f32> = (0..HEAD_DIM).map(|i| i as f32).collect();
        s.push_comp(&row_a).unwrap();
        assert_eq!(s.n_comp, 1);
        assert_eq!(s.comp_rows().len(), HEAD_DIM);
        assert_eq!(&s.comp_rows()[..HEAD_DIM], row_a.as_slice());
    }

    #[test]
    fn compressor_state_push_comp_capacity_overflow_errors() {
        // ctx_size = 4 → comp_cap = 4/4 + 2 = 3. Push 3 rows fine, 4th errors.
        let mut s = CompressorState::new(4, 4).unwrap();
        assert_eq!(s.comp_cap, 3);
        for i in 0..3 {
            s.push_comp(&[i as f32; HEAD_DIM]).unwrap();
        }
        let err = s.push_comp(&[99.0; HEAD_DIM]).unwrap_err();
        assert!(err.to_string().contains("capacity"), "got: {err}");
        // n_comp must not have advanced past the cap.
        assert_eq!(s.n_comp, 3);
    }

    #[test]
    fn compressor_state_push_comp_wrong_width_errors() {
        let mut s = CompressorState::new(128, 256).unwrap();
        let err = s.push_comp(&[0.0; HEAD_DIM - 1]).unwrap_err();
        assert!(err.to_string().contains("HEAD_DIM"), "got: {err}");
    }

    // ---------------------------------------------------------------------
    // KvCache compressor wiring
    // ---------------------------------------------------------------------

    #[test]
    fn kvcache_compressor_some_for_compressed_layers() {
        // 6 layers: 0,1 dense; 2 ratio=4; 3 ratio=128; 4 ratio=4; 5 ratio=128.
        let cache = KvCache::new(6, 4096).unwrap();
        assert!(cache.compressor(0).is_none());
        assert!(cache.compressor(1).is_none());
        let l2 = cache.compressor(2).unwrap();
        assert_eq!(l2.ratio, 4);
        let l3 = cache.compressor(3).unwrap();
        assert_eq!(l3.ratio, 128);
        assert_eq!(cache.compressor(4).unwrap().ratio, 4);
        assert_eq!(cache.compressor(5).unwrap().ratio, 128);
    }

    #[test]
    fn kvcache_clear_all_resets_compressor() {
        let mut cache = KvCache::new(4, 64).unwrap();
        let s = cache.compressor_mut(2).unwrap();
        s.push_comp(&[1.0; HEAD_DIM]).unwrap();
        for v in s.state_score.iter_mut() {
            *v = 0.5;
        }
        cache.clear_all();
        let s = cache.compressor(2).unwrap();
        assert_eq!(s.n_comp, 0);
        assert!(s.state_score.iter().all(|&v| v == NEG_INF));
    }

    #[test]
    fn kvcache_snapshot_restore_round_trips_compressor() {
        let mut cache = KvCache::new(4, 64).unwrap();
        // Seed compressor on layer 2 (ratio 4).
        let s = cache.compressor_mut(2).unwrap();
        for (i, v) in s.state_kv.iter_mut().enumerate() {
            *v = i as f32 * 0.1;
        }
        for (i, v) in s.state_score.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 - 1.0;
        }
        s.push_comp(&[7.0; HEAD_DIM]).unwrap();
        s.push_comp(&[9.0; HEAD_DIM]).unwrap();

        let saved_kv = s.state_kv.clone();
        let saved_score = s.state_score.clone();
        let saved_comp = s.comp_rows().to_vec();
        let saved_n = s.n_comp;

        let snap = cache.snapshot();
        // Stomp the compressor state.
        let s = cache.compressor_mut(2).unwrap();
        s.state_kv.fill(0.0);
        s.state_score.fill(NEG_INF);
        s.n_comp = 0;

        cache.restore(&snap);
        let s = cache.compressor(2).unwrap();
        assert_eq!(s.n_comp, saved_n);
        assert_eq!(s.state_kv, saved_kv);
        assert_eq!(s.state_score, saved_score);
        assert_eq!(s.comp_rows(), saved_comp.as_slice());
    }
}
