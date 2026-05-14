use std::sync::Arc;

use anyhow::Result;

use crate::{engine::Engine, model::kv_cache::KvCache};

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
            kv_cache: KvCache::new(n_layer, ctx_size as usize),
        })
    }

    /// Run prefill for the entire prompt. Returns logits for the last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        tracing::info!("Prefill: {} tokens", tokens.len());
        // TODO: implement forward pass
        self.tokens.extend_from_slice(tokens);
        self.pos = self.tokens.len() as u32;
        self.logits.fill(0.0);
        Ok(&self.logits)
    }

    /// Evaluate one decode token. Returns logits for the next token.
    pub fn eval_token(&mut self, token: u32) -> Result<&[f32]> {
        // TODO: implement single-token forward pass
        self.tokens.push(token);
        self.pos += 1;
        self.logits.fill(0.0);
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
    fn session_prefill_tracks_tokens_and_pos() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 128).unwrap();
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());

        let logits = s.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), engine.config.n_vocab as usize);
        assert_eq!(s.pos(), 3);
        assert_eq!(s.tokens(), &[1, 2, 3]);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn session_eval_token_appends_and_increments_pos() {
        let engine = open_engine();
        let mut s = Session::new(engine.clone(), 128).unwrap();
        let _ = s.prefill(&[10, 20]).unwrap();
        let logits = s.eval_token(30).unwrap();
        assert_eq!(logits.len(), engine.config.n_vocab as usize);
        assert_eq!(s.pos(), 3);
        assert_eq!(s.tokens(), &[10, 20, 30]);
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
}
