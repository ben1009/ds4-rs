use std::sync::Arc;

use anyhow::{Result, bail};

use crate::{engine::Engine, model, model::kv_cache::KvCache};

/// An inference session holding mutable state.
pub struct Session {
    engine: Arc<Engine>,
    tokens: Vec<u32>,
    pos: u32,
    ctx_size: u32,
    logits: Vec<f32>,
    kv_cache: KvCache,
}

impl Session {
    pub fn new(engine: Arc<Engine>, ctx_size: u32) -> Result<Self> {
        tracing::info!("Creating session with ctx_size={ctx_size}");
        let n_vocab = engine.config.n_vocab as usize;
        let n_layer = engine.config.n_layer as usize;
        Ok(Self {
            engine,
            tokens: Vec::new(),
            pos: 0,
            ctx_size,
            logits: vec![0.0; n_vocab],
            kv_cache: KvCache::new(n_layer, ctx_size as usize)?,
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

        let start_len = self.tokens.len();
        let start_pos = self.pos;
        let start_kv_pos = self.kv_cache.len();
        self.tokens.reserve(tokens.len());
        for &token in tokens.iter().take(tokens.len()) {
            if let Err(err) = self.eval_token(token) {
                self.tokens.truncate(start_len);
                self.pos = start_pos;
                self.kv_cache.set_pos(start_kv_pos);
                return Err(err);
            }
        }
        Ok(&self.logits)
    }

    /// Evaluate one decode token. Returns logits for the next token.
    pub fn eval_token(&mut self, token: u32) -> Result<&[f32]> {
        if self.tokens.len() >= self.ctx_size as usize {
            bail!(
                "eval_token context overflow: pos {} >= ctx_size {}",
                self.tokens.len(),
                self.ctx_size
            );
        }

        self.tokens.push(token);
        self.pos = (self.tokens.len() - 1) as u32;

        let engine = self.engine.clone();
        let start_kv_pos = self.kv_cache.len();
        match model::forward::forward_decode(self, &engine) {
            Ok(logits) => {
                self.logits = logits;
                self.pos = self.tokens.len() as u32;
            }
            Err(err) => {
                self.tokens.pop();
                self.pos = self.tokens.len() as u32;
                self.kv_cache.set_pos(start_kv_pos);
                return Err(err);
            }
        }

        Ok(&self.logits)
    }

    /// Discard all session state, keeping the allocated KV buffer for reuse.
    ///
    /// After this, `pos() == 0`, `tokens().is_empty()`, and the KV cache
    /// watermark is back at zero — but the underlying KV `Vec<f32>`s are
    /// not reallocated, so the next prefill / eval reuses them in place.
    /// `logits` keeps its `n_vocab` length so callers that hold a stale
    /// reference can still observe a valid (but zeroed) buffer.
    ///
    /// Mirrors `ds4_session_invalidate` in antirez/ds4 ds4.c.
    pub fn invalidate(&mut self) {
        self.tokens.clear();
        self.pos = 0;
        self.kv_cache.set_pos(0);
        for v in self.logits.iter_mut() {
            *v = 0.0;
        }
    }

    /// Rewind the session to position `pos`, dropping any tokens / KV state
    /// past it. Mirrors `ds4_session_rewind` in antirez/ds4 ds4.c, which
    /// clamps the requested position into `[0, current_len]`.
    ///
    /// `pos` past the current length is silently clamped down — matching
    /// the C reference, where rewinding past the end is a no-op rather than
    /// an error. Rewinding to the current length is a no-op too.
    ///
    /// The KV cache watermark moves with the token list. The underlying
    /// `Vec<f32>` buffers are kept (no reallocation); subsequent writes
    /// overwrite the now-stale entries.
    pub fn rewind(&mut self, pos: u32) {
        let target = (pos as usize).min(self.tokens.len());
        self.tokens.truncate(target);
        self.pos = target as u32;
        self.kv_cache.set_pos(target);
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

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }
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

        // We have 7 metadata entries.
        let tokens: Vec<String> = (0u8..=255).map(|b| format!("<0x{b:02X}>")).collect();
        u64le(&mut buf, 7);

        kv_u32(&mut buf, "llama.vocab_size", 256);
        kv_u32(&mut buf, "llama.embedding_length", 16);
        kv_u32(&mut buf, "llama.attention.head_count", 4);
        kv_u32(&mut buf, "llama.attention.head_count_kv", 4);
        kv_u32(&mut buf, "llama.block_count", 2);
        kv_u32(&mut buf, "llama.feed_forward_length", 32);
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
        assert_eq!(s.kv_cache().len(), 0);
        assert_eq!(s.kv_cache().n_layer(), engine.config.n_layer as usize);
        assert_eq!(s.kv_cache().ctx_size(), 256);
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
    fn session_kv_cache_mut_round_trips_writes() {
        use crate::model::kv_cache::{K_PE_DIM, KV_LATENT_DIM};
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        let lat = vec![3.0f32; KV_LATENT_DIM];
        let pe = vec![4.0f32; K_PE_DIM];
        s.kv_cache_mut().write_latent(0, 0, &lat).unwrap();
        s.kv_cache_mut().write_k_pe(0, 0, &pe).unwrap();
        s.kv_cache_mut().set_pos(1);
        assert_eq!(s.kv_cache().len(), 1);
        assert_eq!(s.kv_cache().read_latent(0, 0), lat.as_slice());
        assert_eq!(s.kv_cache().read_k_pe(0, 0), pe.as_slice());
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
    // These ops manipulate session state directly (tokens, pos, KV
    // watermark) and don't go through the forward path, so they're
    // testable with synthetic state from the minimal GGUF — no real model
    // weights needed.
    // -----------------------------------------------------------------------

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn invalidate_clears_tokens_pos_kv_and_zeros_logits() {
        use crate::model::kv_cache::{K_PE_DIM, KV_LATENT_DIM};
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();

        // Seed synthetic state without going through forward.
        s.tokens.extend_from_slice(&[10, 20, 30]);
        s.pos = 3;
        let lat = vec![1.5f32; KV_LATENT_DIM];
        let pe = vec![2.5f32; K_PE_DIM];
        for p in 0..3 {
            s.kv_cache_mut().write_latent(0, p, &lat).unwrap();
            s.kv_cache_mut().write_k_pe(0, p, &pe).unwrap();
        }
        s.kv_cache_mut().set_pos(3);
        for v in s.logits.iter_mut() {
            *v = 7.0;
        }

        s.invalidate();

        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().len(), 0);
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
        assert_eq!(s.kv_cache().len(), 0);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn invalidate_keeps_kv_buffer_allocated() {
        // The whole point of invalidate over reallocating: the KV `Vec<f32>`
        // capacity (n_layer * ctx_size * dim) survives the call so the next
        // turn doesn't pay the alloc cost.
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        let cap_before = s.kv_cache().ctx_size();
        s.tokens.extend_from_slice(&[1, 2]);
        s.pos = 2;
        s.kv_cache_mut().set_pos(2);
        s.invalidate();
        assert_eq!(s.kv_cache().ctx_size(), cap_before);
        assert_eq!(s.kv_cache().n_layer(), engine.config.n_layer as usize);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_truncates_tokens_and_kv_watermark() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        s.tokens.extend_from_slice(&[10, 20, 30, 40, 50]);
        s.pos = 5;
        s.kv_cache_mut().set_pos(5);

        s.rewind(2);

        assert_eq!(s.pos(), 2);
        assert_eq!(s.tokens(), &[10, 20]);
        assert_eq!(s.kv_cache().len(), 2);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_to_zero_matches_invalidate_for_position_state() {
        // Both end up at pos=0 / empty tokens / kv watermark 0; only
        // logits behaviour diverges (rewind preserves last logits, invalidate
        // zeros them).
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        s.tokens.extend_from_slice(&[1, 2, 3]);
        s.pos = 3;
        s.kv_cache_mut().set_pos(3);

        s.rewind(0);

        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
        assert_eq!(s.kv_cache().len(), 0);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_past_end_clamps_to_current_length() {
        // Matches `ds4_session_rewind`: `pos > checkpoint.len -> pos = checkpoint.len`.
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        s.tokens.extend_from_slice(&[1, 2, 3]);
        s.pos = 3;
        s.kv_cache_mut().set_pos(3);

        s.rewind(99);

        assert_eq!(s.pos(), 3);
        assert_eq!(s.tokens(), &[1, 2, 3]);
        assert_eq!(s.kv_cache().len(), 3);
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
        s.kv_cache_mut().set_pos(2);

        s.rewind(2);

        assert_eq!(s.pos(), 2);
        assert_eq!(s.tokens(), &[7, 8]);
        assert_eq!(s.kv_cache().len(), 2);
    }

    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    #[test]
    fn rewind_then_grow_back_overwrites_stale_kv() {
        // After a rewind, the KV buffer past the new watermark is stale but
        // still allocated. Subsequent writes should land at the new offset
        // without surfacing the old data.
        use crate::model::kv_cache::KV_LATENT_DIM;
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 8).unwrap();
        let stale = vec![9.0f32; KV_LATENT_DIM];
        let fresh = vec![1.0f32; KV_LATENT_DIM];

        s.tokens.extend_from_slice(&[1, 2, 3]);
        s.pos = 3;
        for p in 0..3 {
            s.kv_cache_mut().write_latent(0, p, &stale).unwrap();
        }
        s.kv_cache_mut().set_pos(3);

        s.rewind(1);
        s.tokens.push(99);
        s.pos = 2;
        s.kv_cache_mut().write_latent(0, 1, &fresh).unwrap();
        s.kv_cache_mut().set_pos(2);

        assert_eq!(s.kv_cache().read_latent(0, 0), stale.as_slice());
        assert_eq!(s.kv_cache().read_latent(0, 1), fresh.as_slice());
    }
}
