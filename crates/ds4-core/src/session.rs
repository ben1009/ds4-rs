use std::sync::Arc;

use anyhow::{Result, bail};

use crate::{
    engine::Engine,
    model,
    model::kv_cache::{KvCache, KvCacheSnapshot},
    mtp::MtpState,
    speculative::SpecConfig,
};

/// An inference session holding mutable state.
pub struct Session {
    engine: Arc<Engine>,
    tokens: Vec<u32>,
    pos: u32,
    ctx_size: u32,
    logits: Vec<f32>,
    kv_cache: KvCache,
    /// Reusable rollback buffer. Sized once at session creation; refilled
    /// by `kv_cache.snapshot_into` on each `eval_token` so a mid-forward
    /// failure can roll the ring back without paying a per-token alloc.
    kv_snapshot: KvCacheSnapshot,
    /// Reusable scratch for the per-layer attention `heads` buffer
    /// (`n_head * head_dim` floats). Sized once at session creation; the
    /// forward pass takes ownership for the duration of one decode step
    /// via `mem::take` and puts it back, avoiding a per-token alloc.
    pub(crate) heads_scratch: Vec<f32>,
    /// Collapsed hidden state from the most recent forward pass.
    /// `[n_embd]` vector captured after HC reduction but before output norm.
    /// Used by MTP speculative decoding as the "previous hidden state" input.
    pub(crate) last_hidden: Vec<f32>,
    /// Optional MTP state for speculative decoding. Created when the engine
    /// has MTP weights loaded.
    pub(crate) mtp_state: Option<MtpState>,
}

impl Session {
    pub fn new(engine: Arc<Engine>, ctx_size: u32) -> Result<Self> {
        tracing::info!("Creating session with ctx_size={ctx_size}");
        let n_vocab = engine.config.n_vocab as usize;
        let n_embd = engine.config.n_embd as usize;
        let n_layer = engine.config.n_layer as usize;
        let q_dim = (engine.config.n_head as usize)
            .checked_mul(engine.config.head_dim as usize)
            .ok_or_else(|| anyhow::anyhow!("Session: Q dimension overflow"))?;
        let kv_cache = KvCache::new(n_layer, ctx_size as usize)?;
        let kv_snapshot = KvCacheSnapshot::with_shape(&kv_cache);
        let mtp_state = if engine.mtp_weights.is_some() {
            Some(MtpState::new(&engine.config, ctx_size)?)
        } else {
            None
        };
        Ok(Self {
            engine,
            tokens: Vec::new(),
            pos: 0,
            ctx_size,
            logits: vec![0.0; n_vocab],
            kv_cache,
            kv_snapshot,
            heads_scratch: vec![0.0f32; q_dim],
            last_hidden: vec![0.0f32; n_embd],
            mtp_state,
        })
    }

    /// Run prefill for the entire prompt. Returns logits for the last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        tracing::info!("Prefill: {} tokens", tokens.len());
        if tokens.is_empty() {
            return Ok(&self.logits);
        }
        let final_len = self
            .tokens
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("prefill would overflow token buffer length"))?;
        if final_len > self.ctx_size as usize {
            bail!(
                "prefill context overflow: final length {final_len} exceeds ctx_size {}",
                self.ctx_size
            );
        }

        // One KV snapshot for the entire prefill. It covers both per-token
        // mid-forward errors *and* multi-token rollback: an error at any
        // token rewinds the ring to its pre-prefill state, which is also
        // the only valid restore point (later snapshots would already
        // include rows that need to be undone).
        let start_len = self.tokens.len();
        let start_pos = self.pos;
        let logits_snapshot = self.logits.clone();
        self.kv_cache.snapshot_into(&mut self.kv_snapshot);
        self.tokens.reserve(tokens.len());
        for &token in tokens {
            if let Err(err) = self.eval_token_inner(token, false) {
                self.tokens.truncate(start_len);
                self.pos = start_pos;
                self.logits = logits_snapshot;
                self.kv_cache.restore(&self.kv_snapshot);
                return Err(err);
            }
        }
        Ok(&self.logits)
    }

    /// Evaluate one decode token. Returns logits for the next token.
    pub fn eval_token(&mut self, token: u32) -> Result<&[f32]> {
        self.eval_token_inner(token, true)
    }

    /// Evaluate one decode token without snapshotting the KV cache first.
    ///
    /// Use this during speculative verification where the caller manages its
    /// own snapshot and must not have `kv_snapshot` overwritten.
    pub(crate) fn eval_token_no_snapshot(&mut self, token: u32) -> Result<&[f32]> {
        self.eval_token_inner(token, false)
    }

    /// Perform one speculative decoding step using the MTP draft model.
    ///
    /// Returns logits for the next token after the last accepted one. Multiple
    /// tokens may be accepted per call; they are appended to `self.tokens`
    /// and `self.pos` is advanced accordingly.
    ///
    /// Falls back to standard single-token `eval_token` if MTP is not loaded.
    pub fn eval_token_speculative(
        &mut self,
        engine: &Engine,
        spec_config: &SpecConfig,
    ) -> Result<&[f32]> {
        let _accepted = crate::speculative::generate_speculative(self, engine, spec_config)?;
        // generate_speculative already updated self.tokens, self.pos, and
        // self.logits via eval_token calls inside it. The accepted tokens
        // are the ones that were evaluated. We just need to return logits.
        Ok(&self.logits)
    }

    /// Inner decode step. When `take_snapshot` is true, the KV ring is
    /// snapshotted on entry and restored on error — that's the public
    /// `eval_token` contract. When false, the caller (only `prefill`) has
    /// already taken a snapshot covering this entire batch and restores
    /// itself; we skip the per-token copy to avoid the O(N * cap_raw *
    /// HEAD_DIM) traffic on long prompts.
    fn eval_token_inner(&mut self, token: u32, take_snapshot: bool) -> Result<&[f32]> {
        if self.tokens.len() >= self.ctx_size as usize {
            bail!(
                "eval_token context overflow: pos {} >= ctx_size {}",
                self.tokens.len(),
                self.ctx_size
            );
        }

        // forward_decode pushes one row into each layer's KV ring as it
        // walks the layer stack (see model::forward), so a fallible op
        // past the first push leaves earlier layers with a ghost row
        // that doesn't correspond to any committed token. The ring's
        // append-and-shift means a later successful eval can't
        // compensate.
        if take_snapshot {
            self.kv_cache.snapshot_into(&mut self.kv_snapshot);
        }
        self.tokens.push(token);
        self.pos = (self.tokens.len() - 1) as u32;

        let engine = self.engine.clone();
        match model::forward::forward_decode(self, &engine) {
            Ok(logits) => {
                self.logits = logits;
                self.pos = self.tokens.len() as u32;
            }
            Err(err) => {
                self.tokens.pop();
                self.pos = self.tokens.len() as u32;
                if take_snapshot {
                    self.kv_cache.restore(&self.kv_snapshot);
                }
                return Err(err);
            }
        }

        Ok(&self.logits)
    }

    /// Discard all session state, keeping the allocated KV buffers for reuse.
    ///
    /// After this, `pos() == 0`, `tokens().is_empty()`, and every layer's
    /// ring watermark is back at zero — but the underlying `Vec<f32>`s are
    /// not reallocated. `logits` is zeroed so any subsequent access (e.g.
    /// via an empty prefill) does not return stale predictions.
    ///
    /// Mirrors `ds4_session_invalidate` in antirez/ds4 ds4.c.
    pub fn invalidate(&mut self) {
        self.tokens.clear();
        self.pos = 0;
        self.kv_cache.clear_all();
        self.logits.fill(0.0);
        if let Some(ref mut mtp) = self.mtp_state {
            mtp.clear();
        }
    }

    /// Truncate the token list to `len` and reset `pos` accordingly.
    ///
    /// Used by speculative decoding to undo tokens pushed during verification
    /// before re-evaluating only the accepted prefix.
    pub(crate) fn truncate_tokens(&mut self, len: usize) {
        self.tokens.truncate(len);
        self.pos = self.tokens.len() as u32;
    }

    /// Restore session tokens and position from a loaded cache.
    ///
    /// This is a low-level operation used by `kv_disk::load_session`. It sets
    /// the token list and position without running the forward pass — the KV
    /// cache is expected to already be populated from disk.
    pub(crate) fn restore_from_tokens(&mut self, tokens: Vec<u32>) -> Result<()> {
        if tokens.len() > self.ctx_size as usize {
            bail!(
                "restore_from_tokens: {} tokens exceeds ctx_size {}",
                tokens.len(),
                self.ctx_size
            );
        }
        self.pos = tokens.len() as u32;
        self.tokens = tokens;
        Ok(())
    }

    /// Rewind the session to position `target`, dropping any tokens past it.
    ///
    /// The raw KV ring discards rows older than `cap_raw`, so we cannot
    /// truncate the cache by absolute position the way the old split-storage
    /// implementation did. Instead this is a clear-and-replay: clear every
    /// layer's ring, truncate `tokens` to `target`, then re-run
    /// `forward_decode` for each remaining token to rebuild the ring.
    ///
    /// `target == self.tokens.len()` is a no-op. `target == 0` is equivalent
    /// to [`Session::invalidate`] (no replay needed). `target` past the end
    /// is silently clamped down, matching the C reference.
    ///
    /// If replay fails partway through, the session is left in a clean-but-
    /// empty state (equivalent to a fresh `invalidate()`) and the error is
    /// returned to the caller.
    pub fn rewind(&mut self, target: u32) -> Result<()> {
        let target = (target as usize).min(self.tokens.len());
        if target == self.tokens.len() {
            return Ok(());
        }
        if target == 0 {
            self.invalidate();
            return Ok(());
        }

        let replay: Vec<u32> = self.tokens[..target].to_vec();
        self.invalidate();
        for tok in replay {
            if let Err(err) = self.eval_token(tok) {
                self.invalidate();
                return Err(err);
            }
        }
        Ok(())
    }

    /// Pop the last token and re-evaluate it to recover logits.
    ///
    /// Used for exact prefix matches where the KV cache has all prompt tokens
    /// but no logits are available (logits aren't saved to disk). This
    /// overwrites the last KV row and returns logits for the next token.
    ///
    /// Returns `Err` if the session has no tokens.
    pub fn recompute_last_logits(&mut self) -> Result<&[f32]> {
        let last = *self
            .tokens
            .last()
            .ok_or_else(|| anyhow::anyhow!("recompute_last_logits: session is empty"))?;
        // Snapshot everything before modification for atomic rollback.
        let saved_pos = self.pos;
        let saved_logits = self.logits.clone();
        self.kv_cache.snapshot_into(&mut self.kv_snapshot);
        // Pop last token, decrement pos, and rewind KV watermark so
        // eval_token_inner overwrites the last KV row instead of appending.
        self.tokens.pop();
        self.pos -= 1;
        self.kv_cache.pop_last_row();
        // eval_token_inner pushes the token back and attempts forward.
        // On failure it pops and restores pos, but we must also
        // restore the token vec, logits, and KV to the pre-call state.
        if let Err(err) = self.eval_token_inner(last, false) {
            // eval_token_inner already popped `last` on error, so
            // tokens.len() == saved_len - 1. Push it back.
            self.tokens.push(last);
            self.pos = saved_pos;
            self.logits = saved_logits;
            self.kv_cache.restore(&self.kv_snapshot);
            return Err(err);
        }
        Ok(&self.logits)
    }

    /// Greedy argmax: select the token with highest logit.
    /// NaN values are ignored. Returns `None` if `logits` is empty.
    pub fn argmax(logits: &[f32]) -> Option<u32> {
        let (idx, _) = logits.iter().enumerate().fold(
            (None, f32::NEG_INFINITY),
            |(idx_max, val_max), (idx, &val)| {
                if val > val_max {
                    (Some(idx as u32), val)
                } else {
                    (idx_max, val_max)
                }
            },
        );
        idx
    }

    pub fn pos(&self) -> u32 {
        self.pos
    }

    pub fn ctx_size(&self) -> u32 {
        self.ctx_size
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Current logits from the most recent `prefill` or `eval_token`.
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Collapsed hidden state from the most recent forward pass.
    pub fn last_hidden(&self) -> &[f32] {
        &self.last_hidden
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }

    pub fn kv_snapshot(&self) -> &KvCacheSnapshot {
        &self.kv_snapshot
    }

    pub fn kv_snapshot_mut(&mut self) -> &mut KvCacheSnapshot {
        &mut self.kv_snapshot
    }

    /// Snapshot the KV cache into the reusable rollback buffer.
    pub fn snapshot_kv(&mut self) {
        self.kv_cache.snapshot_into(&mut self.kv_snapshot);
    }

    /// Restore the KV cache from the rollback snapshot.
    pub fn restore_kv(&mut self) {
        self.kv_cache.restore(&self.kv_snapshot);
    }
}

/// Return the number of leading tokens that match between `a` and `b`.
///
/// Compares element-by-element and returns the count of equal leading
/// tokens. The caller is responsible for ensuring both sequences were
/// encoded consistently (e.g., both with or both without BOS).
pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    // Build a minimal valid GGUF file on disk and load it through Engine so we
    // exercise the full engine / model / session plumbing end-to-end.
    fn write_minimal_gguf(path: &std::path::Path) {
        let mut buf: Vec<u8> = Vec::new();
        let u32le = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        let u64le = |buf: &mut Vec<u8>, v: u64| buf.extend_from_slice(&v.to_le_bytes());
        let strle = |buf: &mut Vec<u8>, s: &str| {
            u64le(buf, s.len() as u64);
            buf.extend_from_slice(s.as_bytes());
        };
        let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            strle(buf, k);
            u32le(buf, 4);
            u32le(buf, v);
        };
        let kv_arr_string = |buf: &mut Vec<u8>, k: &str, values: &[String]| {
            strle(buf, k);
            u32le(buf, 9); // Array
            u32le(buf, 8); // inner type: String
            u64le(buf, values.len() as u64);
            for v in values {
                strle(buf, v);
            }
        };

        u32le(&mut buf, crate::gguf::GGUF_MAGIC);
        u32le(&mut buf, 3);
        u64le(&mut buf, 0);

        // We have 8 metadata entries.
        let tokens: Vec<String> = crate::tokenizer::synthetic_byte_tokens();
        u64le(&mut buf, 8);

        kv_u32(&mut buf, "deepseek4.vocab_size", 256);
        kv_u32(&mut buf, "deepseek4.embedding_length", 16);
        kv_u32(&mut buf, "deepseek4.attention.head_count", 4);
        kv_u32(&mut buf, "deepseek4.attention.head_count_kv", 4);
        kv_u32(&mut buf, "deepseek4.block_count", 2);
        kv_u32(&mut buf, "deepseek4.expert_feed_forward_length", 32);
        kv_u32(&mut buf, "deepseek4.attention.q_lora_rank", 8);
        kv_arr_string(&mut buf, "tokenizer.ggml.tokens", &tokens);

        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    fn open_engine() -> Arc<Engine> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ds4-session-test-{}-{}.gguf",
            std::process::id(),
            seq,
        ));
        write_minimal_gguf(&path);
        let engine = Engine::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        engine
    }

    #[test]
    fn argmax_empty_is_none() {
        assert_eq!(Session::argmax(&[]), None);
    }

    #[test]
    fn argmax_single() {
        assert_eq!(Session::argmax(&[1.0]), Some(0));
        assert_eq!(Session::argmax(&[f32::NEG_INFINITY]), None);
    }

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(Session::argmax(&[0.1, 0.9, 0.2]), Some(1));
        assert_eq!(Session::argmax(&[-3.0, -1.0, -2.0]), Some(1));
    }

    #[test]
    fn argmax_skips_nan() {
        assert_eq!(Session::argmax(&[f32::NAN, 0.5, f32::NAN]), Some(1));
        assert_eq!(Session::argmax(&[f32::NAN, f32::NAN]), None);
    }

    #[test]
    fn argmax_ties_take_first() {
        assert_eq!(Session::argmax(&[0.5, 0.5, 0.5]), Some(0));
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_prefill_requires_forward_weights_and_rolls_back() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 128).unwrap();
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());

        let err = s.prefill(&[1, 2, 3]).unwrap_err();
        assert!(err.to_string().contains("token_embd.weight"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.logits.len(), engine.config.n_vocab as usize);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_eval_token_requires_forward_weights_and_rolls_back() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 128).unwrap();
        let err = s.eval_token(30).unwrap_err();
        assert!(err.to_string().contains("token_embd.weight"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.logits.len(), engine.config.n_vocab as usize);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_engine_accessor_returns_same() {
        let engine = open_engine();
        let s = Session::new(engine.clone(), 64).unwrap();
        // Can't compare Engine directly — just check it's reachable.
        assert_eq!(s.engine().config.n_vocab, engine.config.n_vocab);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_initial_state_is_empty() {
        let engine = open_engine();
        let s = Session::new(engine.clone(), 256).unwrap();
        assert_eq!(s.pos(), 0);
        assert_eq!(s.ctx_size(), 256);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().n_layer(), engine.config.n_layer as usize);
        for il in 0..s.kv_cache().n_layer() {
            assert_eq!(s.kv_cache().layer(il).n_raw(), 0);
        }
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_prefill_rejects_context_overflow_before_mutating() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 2).unwrap();
        let err = s.prefill(&[1, 2, 3]).unwrap_err();
        assert!(err.to_string().contains("context overflow"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_eval_token_failure_restores_seeded_kv_state() {
        // forward_decode pushes per-layer as it walks the layer stack, so a
        // mid-forward error leaves earlier layers with a ghost row unless
        // eval_token snapshots and restores. The minimal-GGUF Engine errors
        // at token_embd before any push, so this is a sanity check rather
        // than a true mid-forward reproducer — but it pins the contract.
        use crate::model::kv_cache::HEAD_DIM;
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();

        let seed: Vec<f32> = (0..HEAD_DIM).map(|i| 3.0 + i as f32 * 0.001).collect();
        for il in 0..engine.config.n_layer as usize {
            s.kv_cache_mut().layer_mut(il).push(&seed);
        }

        let err = s.eval_token(5).unwrap_err();
        assert!(err.to_string().contains("token_embd.weight"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        for il in 0..engine.config.n_layer as usize {
            assert_eq!(s.kv_cache().layer(il).n_raw(), 1);
            assert_eq!(s.kv_cache().layer(il).rows(), seed.as_slice());
        }
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_prefill_failure_restores_pre_existing_kv_state() {
        // The minimal-GGUF Engine has no `token_embd.weight`, so any
        // prefill / eval errors out before pushing into the ring. Pre-seed
        // the cache with synthetic rows, kick off a prefill, and verify the
        // seeded rows are still there after the rollback.
        use crate::model::kv_cache::HEAD_DIM;
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 16).unwrap();

        let seed: Vec<f32> = (0..HEAD_DIM).map(|i| 7.0 + i as f32 * 0.001).collect();
        s.kv_cache_mut().layer_mut(0).push(&seed);
        s.kv_cache_mut().layer_mut(1).push(&seed);

        let err = s.prefill(&[1, 2, 3]).unwrap_err();
        assert!(err.to_string().contains("token_embd.weight"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().layer(0).n_raw(), 1);
        assert_eq!(s.kv_cache().layer(1).n_raw(), 1);
        assert_eq!(s.kv_cache().layer(0).rows(), seed.as_slice());
        assert_eq!(s.kv_cache().layer(1).rows(), seed.as_slice());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_eval_without_prefill_rolls_back_on_forward_error() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 16).unwrap();
        let err = s.eval_token(7).unwrap_err();
        assert!(err.to_string().contains("token_embd.weight"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_eval_rejects_context_overflow_before_mutating() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 0).unwrap();
        let err = s.eval_token(7).unwrap_err();
        assert!(err.to_string().contains("context overflow"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_empty_prefill_keeps_state() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 16).unwrap();
        let logits = s.prefill(&[]).unwrap();
        assert_eq!(logits.len(), engine.config.n_vocab as usize);
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
    }

    #[test]
    fn argmax_with_infinity() {
        assert_eq!(Session::argmax(&[1.0, f32::INFINITY, 2.0]), Some(1));
        assert_eq!(
            Session::argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]),
            Some(2)
        );
    }

    #[test]
    fn argmax_all_neg_infinity_is_none() {
        assert_eq!(
            Session::argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
            None
        );
    }

    // -----------------------------------------------------------------------
    // Lifecycle: invalidate / rewind
    //
    // These ops manipulate session state directly (tokens, pos, KV ring)
    // and don't go through the forward path, so they're testable with
    // synthetic state from the minimal GGUF — no real model weights needed.
    // -----------------------------------------------------------------------

    fn synth_row(seed: f32) -> Vec<f32> {
        use crate::model::kv_cache::HEAD_DIM;
        (0..HEAD_DIM).map(|i| seed + i as f32 * 0.001).collect()
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn invalidate_clears_tokens_pos_kv_and_zeros_logits() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();

        // Seed synthetic state without going through forward.
        s.tokens.extend_from_slice(&[10, 20, 30]);
        s.pos = 3;
        for _ in 0..3 {
            s.kv_cache_mut().layer_mut(0).push(&synth_row(1.5));
        }
        for v in s.logits.iter_mut() {
            *v = 7.0;
        }

        s.invalidate();

        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().layer(0).n_raw(), 0);
        assert_eq!(s.logits.len(), engine.config.n_vocab as usize);
        assert!(s.logits.iter().all(|&v| v == 0.0));
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn invalidate_on_fresh_session_is_noop() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 4).unwrap();
        s.invalidate();
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().layer(0).n_raw(), 0);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn invalidate_keeps_kv_cap_unchanged() {
        // The whole point of invalidate over reallocating: the KV ring's
        // cap_raw survives the call so the next turn doesn't pay alloc cost.
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        let cap_before = s.kv_cache().cap_raw();
        s.tokens.extend_from_slice(&[1, 2]);
        s.pos = 2;
        s.kv_cache_mut().layer_mut(0).push(&synth_row(0.0));
        s.invalidate();
        assert_eq!(s.kv_cache().cap_raw(), cap_before);
        assert_eq!(s.kv_cache().n_layer(), engine.config.n_layer as usize);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_to_current_length_is_noop() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        s.tokens.extend_from_slice(&[7, 8]);
        s.pos = 2;
        for _ in 0..2 {
            s.kv_cache_mut().layer_mut(0).push(&synth_row(0.0));
        }

        s.rewind(2).unwrap();

        assert_eq!(s.pos(), 2);
        assert_eq!(s.tokens(), &[7, 8]);
        // No-op: ring untouched.
        assert_eq!(s.kv_cache().layer(0).n_raw(), 2);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_past_end_clamps_to_current_length() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        s.tokens.extend_from_slice(&[1, 2, 3]);
        s.pos = 3;
        for _ in 0..3 {
            s.kv_cache_mut().layer_mut(0).push(&synth_row(0.0));
        }

        s.rewind(99).unwrap();

        assert_eq!(s.pos(), 3);
        assert_eq!(s.tokens(), &[1, 2, 3]);
        assert_eq!(s.kv_cache().layer(0).n_raw(), 3);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_to_zero_acts_like_invalidate() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        s.tokens.extend_from_slice(&[1, 2, 3]);
        s.pos = 3;
        for _ in 0..3 {
            s.kv_cache_mut().layer_mut(0).push(&synth_row(0.0));
        }
        for v in s.logits.iter_mut() {
            *v = 5.0;
        }

        s.rewind(0).unwrap();

        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().layer(0).n_raw(), 0);
        assert!(s.logits.iter().all(|&v| v == 0.0));
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_partial_replays_through_forward_and_surfaces_error() {
        // The minimal GGUF lacks forward weights, so any replay step fails
        // inside `eval_token`. The contract is "leave session in a clean-but-
        // empty state": tokens cleared, pos=0, ring empty, error returned.
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        s.tokens.extend_from_slice(&[10, 20, 30, 40, 50]);
        s.pos = 5;
        for _ in 0..5 {
            s.kv_cache_mut().layer_mut(0).push(&synth_row(0.0));
        }

        let err = s.rewind(2).unwrap_err();
        assert!(err.to_string().contains("token_embd.weight"));
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().layer(0).n_raw(), 0);
    }

    // ── common_prefix_len ─────────────────────────────────────────────────

    #[test]
    fn common_prefix_len_identical() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn common_prefix_len_partial() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4, 5], &[1, 2, 3, 7, 8]), 3);
    }

    #[test]
    fn common_prefix_len_no_match() {
        assert_eq!(common_prefix_len(&[1, 2], &[3, 4]), 0);
    }

    #[test]
    fn common_prefix_len_empty() {
        assert_eq!(common_prefix_len(&[], &[1, 2]), 0);
        assert_eq!(common_prefix_len(&[1, 2], &[]), 0);
        assert_eq!(common_prefix_len(&[], &[]), 0);
    }

    #[test]
    fn common_prefix_len_one_longer() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4, 5]), 3);
    }
}
