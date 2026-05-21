//! On-disk KVC cache: save and load session state.
//!
//! The KVC file format is binary-compatible with antirez/ds4:
//!
//! ```text
//! [48 bytes]  KvcHeader
//! [4 bytes]   text_length (u32 LE)
//! [N bytes]   rendered text (UTF-8)
//! [variable]  DSV4 payload
//! ```
//!
//! The DSV4 payload contains a 13-field sub-header, token IDs, logits
//! (zero-filled), per-layer compressed/indexer row counts, and the full
//! per-layer KV tensor data.
//!
//! See antirez/ds4 ds4.c `ds4_session_save` / `ds4_session_load`.

use std::{
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use super::kv_cache::{HEAD_DIM, IDX_DIM, SWA};
use crate::{config::layer_compress_ratio, engine::Engine, session::Session};

// ── Constants ────────────────────────────────────────────────────────────────

/// KVC file magic: ASCII "KVC".
const KVC_MAGIC: &[u8; 3] = b"KVC";
/// KVC format version.
const KVC_VERSION: u8 = 1;
/// DSV4 payload magic: "DSV4" as a u32 LE = 0x34565344.
const DSV4_MAGIC: u32 = 0x3456_5344;
/// DSV4 payload version.
const DSV4_VERSION: u32 = 1;

// ── Public types ─────────────────────────────────────────────────────────────

/// Reason for saving the cache. Maps to `save_reason` in the KVC header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SaveReason {
    Cold = 0,
    Continued = 1,
    Evict = 2,
    Shutdown = 3,
    Manual = 4,
}

// ── KVC header (48 bytes) ────────────────────────────────────────────────────

/// 48-byte KVC file header. Serialized field-by-field in LE byte order.
struct KvcHeader {
    magic: [u8; 3],
    version: u8,
    routed_quant_bits: u8,
    save_reason: u8,
    reserved: [u8; 2],
    cached_token_count: u32,
    hit_count: u32,
    context_size: u32,
    reserved2: [u8; 4],
    creation_time: u64,
    last_used_time: u64,
    payload_bytes: u64,
}

fn write_header(w: &mut impl Write, h: &KvcHeader) -> Result<()> {
    w.write_all(&h.magic)?;
    w.write_all(&[h.version])?;
    w.write_all(&[h.routed_quant_bits])?;
    w.write_all(&[h.save_reason])?;
    w.write_all(&h.reserved)?;
    w.write_all(&h.cached_token_count.to_le_bytes())?;
    w.write_all(&h.hit_count.to_le_bytes())?;
    w.write_all(&h.context_size.to_le_bytes())?;
    w.write_all(&h.reserved2)?;
    w.write_all(&h.creation_time.to_le_bytes())?;
    w.write_all(&h.last_used_time.to_le_bytes())?;
    w.write_all(&h.payload_bytes.to_le_bytes())?;
    Ok(())
}

fn read_header(r: &mut impl Read) -> Result<KvcHeader> {
    let mut magic = [0u8; 3];
    r.read_exact(&mut magic)?;
    let mut version = [0u8; 1];
    r.read_exact(&mut version)?;
    let mut routed_quant_bits = [0u8; 1];
    r.read_exact(&mut routed_quant_bits)?;
    let mut save_reason = [0u8; 1];
    r.read_exact(&mut save_reason)?;
    let mut reserved = [0u8; 2];
    r.read_exact(&mut reserved)?;
    let cached_token_count = read_u32(r)?;
    let hit_count = read_u32(r)?;
    let context_size = read_u32(r)?;
    let mut reserved2 = [0u8; 4];
    r.read_exact(&mut reserved2)?;
    let creation_time = read_u64(r)?;
    let last_used_time = read_u64(r)?;
    let payload_bytes = read_u64(r)?;
    Ok(KvcHeader {
        magic,
        version: version[0],
        routed_quant_bits: routed_quant_bits[0],
        save_reason: save_reason[0],
        reserved,
        cached_token_count,
        hit_count,
        context_size,
        reserved2,
        creation_time,
        last_used_time,
        payload_bytes,
    })
}

// ── Serialization helpers ────────────────────────────────────────────────────

fn write_u32(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(Into::into)
}

fn write_f32_slice(w: &mut impl Write, data: &[f32]) -> Result<()> {
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32_slice(r: &mut impl Read, out: &mut [f32]) -> Result<()> {
    let mut buf = [0u8; 4];
    for val in out.iter_mut() {
        r.read_exact(&mut buf)?;
        *val = f32::from_le_bytes(buf);
    }
    Ok(())
}

fn skip_bytes(r: &mut impl Read, n: u64) -> Result<()> {
    let copied = std::io::copy(&mut r.take(n), &mut std::io::sink())?;
    if copied != n {
        bail!("KVC: unexpected EOF while skipping {n} bytes (got {copied})");
    }
    Ok(())
}

// ── Save ─────────────────────────────────────────────────────────────────────

/// Save the session's KV cache to a KVC file.
///
/// Binary-compatible with antirez/ds4 `ds4_session_save`. The file can be
/// loaded by the C reference and vice versa.
pub fn save_session(path: &Path, session: &Session, reason: SaveReason) -> Result<()> {
    let engine = session.engine();
    let config = &engine.config;
    let tokenizer = &engine.tokenizer;
    let kv = session.kv_cache();
    let tokens = session.tokens();

    let n_layer = config.n_layer as usize;
    let n_vocab = config.n_vocab as usize;

    // Verify all layers have the same n_raw (the raw_live assumption).
    let raw_live = kv.layer(0).n_raw();
    for il in 1..n_layer {
        assert_eq!(
            kv.layer(il).n_raw(),
            raw_live,
            "save_session: n_raw mismatch at layer {il}"
        );
    }

    let rendered = tokenizer.decode_tokens(tokens);
    let rendered_bytes = rendered.as_bytes();

    let payload_bytes = compute_payload_size(n_layer, n_vocab, tokens.len(), kv)?;

    let file = std::fs::File::create(path)
        .with_context(|| format!("KVC: cannot create {}", path.display()))?;
    let mut w = BufWriter::new(file);

    // ── KVC header (48 bytes) ──
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    write_header(
        &mut w,
        &KvcHeader {
            magic: *KVC_MAGIC,
            version: KVC_VERSION,
            routed_quant_bits: 2, // DS4 Flash uses IQ2_XXS / Q2_K
            save_reason: reason as u8,
            reserved: [0; 2],
            cached_token_count: tokens.len() as u32,
            hit_count: 0,
            context_size: session.ctx_size(),
            reserved2: [0; 4],
            creation_time: now,
            last_used_time: now,
            payload_bytes: payload_bytes as u64,
        },
    )?;

    // ── Text section ──
    write_u32(&mut w, rendered_bytes.len() as u32)?;
    w.write_all(rendered_bytes)?;

    // ── DSV4 payload ──
    write_dsv4_payload(&mut w, kv, tokens, n_layer, n_vocab, raw_live)?;

    w.flush()?;
    Ok(())
}

fn compute_payload_size(
    n_layer: usize,
    n_vocab: usize,
    token_count: usize,
    kv: &super::kv_cache::KvCache,
) -> Result<usize> {
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| anyhow::anyhow!("compute_payload_size: overflow"))
    };
    let add = |a: usize, b: usize| {
        a.checked_add(b)
            .ok_or_else(|| anyhow::anyhow!("compute_payload_size: overflow"))
    };

    let sub_header = 13 * 4;
    let token_ids = mul(token_count, 4)?;
    let logits = mul(n_vocab, 4)?;
    let comp_row_counts = mul(n_layer, 4)?;
    let idx_row_counts = mul(n_layer, 4)?;

    let mut per_layer: usize = 0;
    for il in 0..n_layer {
        let n_raw = kv.layer(il).n_raw();
        per_layer = add(per_layer, mul(mul(n_raw, HEAD_DIM)?, 4)?)?;

        let ratio = layer_compress_ratio(il as u32);
        if ratio != 0 {
            let comp = kv.compressor(il).unwrap();
            per_layer = add(per_layer, mul(mul(comp.n_comp, HEAD_DIM)?, 4)?)?;
            per_layer = add(per_layer, mul(comp.state_kv.len(), 4)?)?;
            per_layer = add(per_layer, mul(comp.state_score.len(), 4)?)?;
        }
        if ratio == 4 {
            let idx = kv.indexer(il).unwrap();
            per_layer = add(per_layer, mul(mul(idx.n_comp, IDX_DIM)?, 4)?)?;
            per_layer = add(per_layer, mul(idx.state_kv.len(), 4)?)?;
            per_layer = add(per_layer, mul(idx.state_score.len(), 4)?)?;
        }
    }

    let fixed = add(
        add(sub_header, token_ids)?,
        add(logits, add(comp_row_counts, idx_row_counts)?)?,
    )?;
    add(fixed, per_layer)
}

fn write_dsv4_payload(
    w: &mut impl Write,
    kv: &super::kv_cache::KvCache,
    tokens: &[u32],
    n_layer: usize,
    n_vocab: usize,
    raw_live: usize,
) -> Result<()> {
    // Sub-header: 13 x u32
    write_u32(w, DSV4_MAGIC)?;
    write_u32(w, DSV4_VERSION)?;
    write_u32(w, kv.cap_raw() as u32)?; // ctx_size placeholder (cap_raw is derived from it)
    write_u32(w, 0)?; // prefill_cap
    write_u32(w, kv.cap_raw() as u32)?; // raw_cap
    write_u32(w, SWA as u32)?; // raw_window
    write_u32(w, 0)?; // comp_cap (per-layer, not global)
    write_u32(w, tokens.len() as u32)?; // token_count
    write_u32(w, n_layer as u32)?; // layer_count
    write_u32(w, HEAD_DIM as u32)?; // head_dim
    write_u32(w, IDX_DIM as u32)?; // indexer_head_dim
    write_u32(w, n_vocab as u32)?; // vocab_size
    write_u32(w, raw_live as u32)?; // raw_live

    // Token IDs
    for &tok in tokens {
        write_u32(w, tok)?;
    }

    // Logits: zero-filled for binary layout compatibility
    let zeros = [0u8; 4096];
    let mut remaining = n_vocab * 4;
    while remaining > 0 {
        let chunk = remaining.min(zeros.len());
        w.write_all(&zeros[..chunk])?;
        remaining -= chunk;
    }

    // Compressed row counts per layer
    for il in 0..n_layer {
        let n_comp = kv.compressor(il).map_or(0, |c| c.n_comp as u32);
        write_u32(w, n_comp)?;
    }

    // Indexer row counts per layer
    for il in 0..n_layer {
        let n_idx = kv.indexer(il).map_or(0, |c| c.n_comp as u32);
        write_u32(w, n_idx)?;
    }

    // Per-layer data
    for il in 0..n_layer {
        let layer = kv.layer(il);
        let n_raw = layer.n_raw();
        write_f32_slice(w, &layer.rows()[..n_raw * HEAD_DIM])?;

        let ratio = layer_compress_ratio(il as u32);

        if ratio != 0 {
            let comp = kv.compressor(il).unwrap();
            write_f32_slice(w, comp.comp_rows())?;
            write_f32_slice(w, &comp.state_kv)?;
            write_f32_slice(w, &comp.state_score)?;
        }

        if ratio == 4 {
            let idx = kv.indexer(il).unwrap();
            write_f32_slice(w, idx.comp_rows())?;
            write_f32_slice(w, &idx.state_kv)?;
            write_f32_slice(w, &idx.state_score)?;
        }
    }

    Ok(())
}

// ── Load ─────────────────────────────────────────────────────────────────────

/// Load a session from a KVC file, restoring the full KV cache state.
///
/// Returns a new `Session` with tokens, position, and KV cache restored.
/// The engine must match the one used when saving.
pub fn load_session(path: &Path, engine: &Arc<Engine>) -> Result<Session> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("KVC: cannot open {}", path.display()))?;
    let mut r = BufReader::new(file);

    // ── KVC header ──
    let header = read_header(&mut r)?;
    if header.magic != *KVC_MAGIC {
        bail!("KVC: bad magic: {:?}", header.magic);
    }
    if header.version != KVC_VERSION {
        bail!("KVC: unsupported version {}", header.version);
    }

    let config = &engine.config;
    let n_layer = config.n_layer as usize;
    let n_vocab = config.n_vocab as usize;

    // ── Text section ──
    let text_len = read_u32(&mut r)? as usize;
    let mut text_bytes = vec![0u8; text_len];
    r.read_exact(&mut text_bytes)?;
    let rendered_text =
        String::from_utf8(text_bytes).context("KVC: rendered text is not valid UTF-8")?;

    // ── DSV4 payload ──
    let session = read_dsv4_payload(&mut r, engine, n_layer, n_vocab, &header)?;

    // Validate rendered text matches decoded tokens (non-fatal).
    let decoded = engine.tokenizer.decode_tokens(session.tokens());
    if decoded != rendered_text {
        tracing::warn!("KVC: rendered text mismatch (tokens are authoritative)");
    }

    Ok(session)
}

// ── Prefix matching ─────────────────────────────────────────────────────────

/// Lightweight scan: read just the token IDs from a KVC file.
///
/// Returns `(token_ids, context_size, total_token_count)` without loading KV
/// state. `total_token_count` is the count from the file header, which may
/// exceed `token_ids.len()` when using [`read_token_ids_limited`].
pub fn read_token_ids(path: &Path) -> Result<(Vec<u32>, u32, usize)> {
    read_token_ids_inner(path, None)
}

fn read_token_ids_inner(
    path: &Path,
    read_at_most: Option<usize>,
) -> Result<(Vec<u32>, u32, usize)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("KVC: cannot open {}", path.display()))?;
    let mut r = BufReader::new(file);

    let header = read_header(&mut r)?;
    if header.magic != *KVC_MAGIC {
        bail!("KVC: bad magic: {:?}", header.magic);
    }
    if header.version != KVC_VERSION {
        bail!("KVC: unsupported version {}", header.version);
    }

    // Skip text section
    let text_len = read_u32(&mut r)? as u64;
    skip_bytes(&mut r, text_len)?;

    // DSV4 sub-header: 13 x u32. Read magic + token_count, skip the rest.
    let dsv4_magic = read_u32(&mut r)?;
    if dsv4_magic != DSV4_MAGIC {
        bail!("KVC/DSV4: bad magic: 0x{dsv4_magic:08X}");
    }
    // Skip fields 1..7 (version, ctx_size, prefill_cap, raw_cap, raw_window, comp_cap)
    skip_bytes(&mut r, 6 * 4)?;
    let token_count = read_u32(&mut r)?;
    // Hard cap: even if context_size in the header is corrupt, don't allocate
    // more than ~4 MB for token IDs (1M tokens × 4 bytes).
    const MAX_TOKEN_COUNT: u32 = 1_000_000;
    if token_count > header.context_size || token_count > MAX_TOKEN_COUNT {
        bail!(
            "KVC: token_count {token_count} exceeds context_size {} or hard limit {MAX_TOKEN_COUNT}",
            header.context_size
        );
    }
    // Skip remaining sub-header fields: layer_count, head_dim, indexer_head_dim, vocab_size,
    // raw_live (5 × u32 = 20 bytes)
    skip_bytes(&mut r, 20)?;

    // Read token IDs
    let count = usize::try_from(token_count)
        .map_err(|_| anyhow::anyhow!("KVC: token_count {token_count} overflows usize"))?;
    let read_count = read_at_most.map_or(count, |max| count.min(max));
    let read_bytes = read_count
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("KVC: read byte count overflow"))?;
    let mut bytes = vec![0u8; read_bytes];
    r.read_exact(&mut bytes)?;
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Skip any remaining tokens we didn't need
    if read_count < count {
        let skip = (count - read_count)
            .checked_mul(4)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| anyhow::anyhow!("KVC: skip byte count overflow"))?;
        skip_bytes(&mut r, skip)?;
    }

    Ok((tokens, header.context_size, count))
}

/// Like [`read_token_ids`] but reads at most `max` tokens, skipping the rest.
/// Used by [`find_prefix_match`] to avoid reading full token sequences.
pub fn read_token_ids_limited(path: &Path, max: usize) -> Result<(Vec<u32>, u32, usize)> {
    read_token_ids_inner(path, Some(max))
}

/// Scan `cache_dir` for KVC files and find the one whose token IDs
/// form the longest prefix of `query_tokens`.
///
/// Returns `Some((path, common_len))` for the best match, or `None`
/// if no file's tokens share a prefix with `query_tokens`.
/// Errors on individual files are logged and skipped.
///
/// Note: this performs a linear scan of the directory, reading only
/// the header and token IDs from each file. For large cache directories,
/// a more efficient lookup (e.g., index file or prefix tree) could be
/// added later.
pub fn find_prefix_match(
    cache_dir: &Path,
    query_tokens: &[u32],
) -> Option<(std::path::PathBuf, usize)> {
    use crate::session::common_prefix_len;

    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return None,
    };

    entries
        .flatten()
        .par_bridge()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.extension().is_some_and(|e| e == "kvc") {
                return None;
            }
            match read_token_ids_limited(&path, query_tokens.len()) {
                Ok((cached_tokens, _ctx, total)) => {
                    let common = common_prefix_len(&cached_tokens, query_tokens);
                    // The entire cached sequence must be a prefix of the query.
                    // Allow exact matches (total == query_tokens.len()) — the caller
                    // handles the empty-suffix case by re-evaluating the last token.
                    if common > 0 && common == cached_tokens.len() && total <= query_tokens.len() {
                        Some((path, common))
                    } else {
                        None
                    }
                }
                Err(e) => {
                    tracing::debug!("KVC: skipping {}: {e}", path.display());
                    None
                }
            }
        })
        .reduce_with(|a, b| if a.1 >= b.1 { a } else { b })
}

/// Load the best prefix-matching KVC session from `cache_dir` for
/// `query_tokens`. Returns `Some((session, suffix_tokens))` if a
/// match was found, or `None` if no cached session shares a prefix.
pub fn load_prefix_match(
    cache_dir: &Path,
    query_tokens: &[u32],
    engine: &Arc<Engine>,
) -> Result<Option<(Session, Vec<u32>)>> {
    let (path, common_len) = match find_prefix_match(cache_dir, query_tokens) {
        Some(v) => v,
        None => return Ok(None),
    };

    tracing::info!(
        "KVC prefix match: {} ({} common tokens)",
        path.display(),
        common_len
    );

    let session = load_session(&path, engine)?;
    let suffix = query_tokens[common_len..].to_vec();
    Ok(Some((session, suffix)))
}

fn read_dsv4_payload(
    r: &mut impl Read,
    engine: &Arc<Engine>,
    n_layer: usize,
    n_vocab: usize,
    header: &KvcHeader,
) -> Result<Session> {
    // Sub-header: 13 x u32
    let dsv4_magic = read_u32(r)?;
    if dsv4_magic != DSV4_MAGIC {
        bail!("KVC/DSV4: bad magic: 0x{dsv4_magic:08X}");
    }
    let _dsv4_version = read_u32(r)?;
    let ctx_size = read_u32(r)?;
    let _prefill_cap = read_u32(r)?;
    let raw_cap = read_u32(r)?;
    let _raw_window = read_u32(r)?;
    let _comp_cap = read_u32(r)?;
    let token_count = read_u32(r)?;
    let file_layer_count = read_u32(r)?;
    let file_head_dim = read_u32(r)?;
    let file_idx_head_dim = read_u32(r)?;
    let file_vocab_size = read_u32(r)?;
    let raw_live = read_u32(r)?;

    // Validate sub-header
    if file_layer_count as usize != n_layer {
        bail!("DSV4: layer_count mismatch: file={file_layer_count}, engine={n_layer}");
    }
    if file_head_dim as usize != HEAD_DIM {
        bail!("DSV4: head_dim mismatch: file={file_head_dim}, engine={HEAD_DIM}");
    }
    if file_idx_head_dim as usize != IDX_DIM {
        bail!("DSV4: indexer_head_dim mismatch: file={file_idx_head_dim}, engine={IDX_DIM}");
    }
    if file_vocab_size as usize != n_vocab {
        bail!("DSV4: vocab_size mismatch: file={file_vocab_size}, engine={n_vocab}");
    }
    if token_count != header.cached_token_count {
        bail!(
            "DSV4: token_count mismatch: payload={token_count}, header={}",
            header.cached_token_count
        );
    }
    if raw_live > raw_cap {
        bail!("DSV4: raw_live ({raw_live}) > raw_cap ({raw_cap})");
    }
    // Validate raw_live fits in the session's allocated ring.
    let expected_cap_raw = SWA.min(ctx_size as usize).max(1);
    if raw_live as usize > expected_cap_raw {
        bail!("DSV4: raw_live ({raw_live}) > expected cap_raw ({expected_cap_raw})");
    }

    // Create session with the saved ctx_size from the header.
    let ctx_size = header.context_size;
    let mut session = Session::new(engine.clone(), ctx_size)?;

    // Validate token_count before allocating
    if token_count > header.context_size {
        bail!(
            "DSV4: token_count {token_count} exceeds context_size {}",
            header.context_size
        );
    }

    // Read token IDs
    let mut tokens = Vec::with_capacity(token_count as usize);
    for _ in 0..token_count {
        let tok = read_u32(r)?;
        if tok >= file_vocab_size {
            bail!("DSV4: token ID {tok} >= vocab_size {file_vocab_size}");
        }
        tokens.push(tok);
    }

    // Skip logits
    skip_bytes(r, (n_vocab as u64) * 4)?;

    // Read compressed row counts per layer
    let mut comp_counts = Vec::with_capacity(n_layer);
    for _ in 0..n_layer {
        comp_counts.push(read_u32(r)? as usize);
    }

    // Read indexer row counts per layer
    let mut idx_counts = Vec::with_capacity(n_layer);
    for _ in 0..n_layer {
        idx_counts.push(read_u32(r)? as usize);
    }

    // Per-layer data
    for il in 0..n_layer {
        let ratio = layer_compress_ratio(il as u32);

        // Raw KV rows — read one row at a time into a stack buffer
        let mut row = [0.0f32; HEAD_DIM];
        for _ in 0..raw_live as usize {
            read_f32_slice(r, &mut row)?;
            session.kv_cache_mut().layer_mut(il).push(&row);
        }

        // Compressed KV rows + state (layers 2+)
        if ratio != 0 {
            let n_comp = comp_counts[il];
            let comp = session.kv_cache_mut().compressor_mut(il).unwrap();
            if n_comp > comp.comp_cap {
                bail!(
                    "DSV4: n_comp {n_comp} exceeds capacity {} at layer {il}",
                    comp.comp_cap
                );
            }

            // Read state directly into the compressor's buffers
            read_f32_slice(r, &mut comp.state_kv)?;
            read_f32_slice(r, &mut comp.state_score)?;

            // Read compressed rows one at a time
            for _ in 0..n_comp {
                read_f32_slice(r, &mut row)?;
                comp.push_comp(&row)?;
            }
        }

        // Indexer data (ratio-4 layers only)
        if ratio == 4 {
            let n_idx = idx_counts[il];
            let idx = session.kv_cache_mut().indexer_mut(il).unwrap();
            if n_idx > idx.comp_cap {
                bail!(
                    "DSV4: n_idx {n_idx} exceeds capacity {} at layer {il}",
                    idx.comp_cap
                );
            }

            // Read state directly into the indexer's buffers
            read_f32_slice(r, &mut idx.state_kv)?;
            read_f32_slice(r, &mut idx.state_score)?;

            // Read indexer rows one at a time
            let mut idx_row = [0.0f32; IDX_DIM];
            for _ in 0..n_idx {
                read_f32_slice(r, &mut idx_row)?;
                idx.push_comp(&idx_row)?;
            }
        }
    }

    session.restore_from_tokens(tokens)?;
    Ok(session)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF builder for test engines. Reuses the pattern from
    /// session.rs and layer.rs tests.
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
            u32le(buf, 9);
            u32le(buf, 8);
            u64le(buf, values.len() as u64);
            for v in values {
                strle(buf, v);
            }
        };
        u32le(&mut buf, crate::gguf::GGUF_MAGIC);
        u32le(&mut buf, 3);
        u64le(&mut buf, 0);
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

    fn open_engine() -> (Arc<Engine>, std::path::PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ds4-kv-disk-test-{}-{}.gguf",
            std::process::id(),
            seq,
        ));
        write_minimal_gguf(&path);
        let engine = Engine::open(&path).unwrap();
        (engine, path)
    }

    #[test]
    fn dsv4_magic_is_correct() {
        assert_eq!(DSV4_MAGIC, 0x3456_5344);
    }

    #[test]
    fn header_is_48_bytes() {
        let mut buf = Vec::new();
        let now = 1700000000u64;
        write_header(
            &mut buf,
            &KvcHeader {
                magic: *KVC_MAGIC,
                version: KVC_VERSION,
                routed_quant_bits: 2,
                save_reason: 0,
                reserved: [0; 2],
                cached_token_count: 10,
                hit_count: 0,
                context_size: 2048,
                reserved2: [0; 4],
                creation_time: now,
                last_used_time: now,
                payload_bytes: 100,
            },
        )
        .unwrap();
        assert_eq!(buf.len(), 48);
    }

    #[test]
    fn save_load_round_trip_empty_session() {
        let (engine, _path) = open_engine();
        let session = Session::new(engine.clone(), 2048).unwrap();

        let kvc_path =
            std::env::temp_dir().join(format!("ds4-kvc-test-empty-{}.kvc", std::process::id()));
        save_session(&kvc_path, &session, SaveReason::Manual).unwrap();

        let loaded = load_session(&kvc_path, &engine).unwrap();
        assert_eq!(loaded.tokens(), session.tokens());
        assert_eq!(loaded.pos(), session.pos());
        assert_eq!(loaded.ctx_size(), session.ctx_size());

        let _ = std::fs::remove_file(&kvc_path);
    }

    #[test]
    fn save_load_round_trip_with_tokens() {
        let (engine, _path) = open_engine();
        let mut session = Session::new(engine.clone(), 2048).unwrap();

        // Manually set token list via restore_from_tokens (no forward pass
        // needed — the GGUF test helper doesn't include model weights).
        session.restore_from_tokens(vec![42, 100, 7]).unwrap();

        let kvc_path =
            std::env::temp_dir().join(format!("ds4-kvc-test-tokens-{}.kvc", std::process::id()));
        save_session(&kvc_path, &session, SaveReason::Shutdown).unwrap();

        let loaded = load_session(&kvc_path, &engine).unwrap();
        assert_eq!(loaded.tokens(), session.tokens());
        assert_eq!(loaded.pos(), session.pos());

        // Verify raw KV state was restored (all zero since no forward pass).
        for il in 0..engine.config.n_layer as usize {
            assert_eq!(
                loaded.kv_cache().layer(il).n_raw(),
                session.kv_cache().layer(il).n_raw(),
                "n_raw mismatch at layer {il}"
            );
        }

        let _ = std::fs::remove_file(&kvc_path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let (engine, _path) = open_engine();
        let kvc_path =
            std::env::temp_dir().join(format!("ds4-kvc-test-badmagic-{}.kvc", std::process::id()));

        // Write a full 48-byte header with bad magic.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"XXX"); // bad magic
        buf.push(KVC_VERSION);
        buf.push(2); // quant_bits
        buf.push(0); // reason
        buf.extend_from_slice(&[0u8; 2]); // reserved
        buf.extend_from_slice(&0u32.to_le_bytes()); // tokens
        buf.extend_from_slice(&0u32.to_le_bytes()); // hits
        buf.extend_from_slice(&2048u32.to_le_bytes()); // ctx_size
        buf.extend_from_slice(&[0u8; 4]); // reserved2
        buf.extend_from_slice(&0u64.to_le_bytes()); // created
        buf.extend_from_slice(&0u64.to_le_bytes()); // last_used
        buf.extend_from_slice(&0u64.to_le_bytes()); // payload_bytes
        std::fs::write(&kvc_path, &buf).unwrap();

        let result = load_session(&kvc_path, &engine);
        assert!(result.is_err(), "expected error for bad magic");
        if let Err(err) = result {
            assert!(err.to_string().contains("bad magic"), "got: {err}");
        }

        let _ = std::fs::remove_file(&kvc_path);
    }

    #[test]
    fn payload_bytes_matches_actual_size() {
        let (engine, _path) = open_engine();
        let session = Session::new(engine.clone(), 2048).unwrap();

        let kvc_path =
            std::env::temp_dir().join(format!("ds4-kvc-test-payloadsz-{}.kvc", std::process::id()));
        save_session(&kvc_path, &session, SaveReason::Manual).unwrap();

        // Read back header and verify payload_bytes matches actual file layout.
        let mut f = std::fs::File::open(&kvc_path).unwrap();
        let header = read_header(&mut f).unwrap();
        let text_len = read_u32(&mut f).unwrap() as u64;
        let mut text_buf = vec![0u8; text_len as usize];
        f.read_exact(&mut text_buf).unwrap();

        // File size = 48 (header) + 4 (text_len) + text_len + payload_bytes
        let file_len = std::fs::metadata(&kvc_path).unwrap().len();
        let expected = 48u64 + 4 + text_len + header.payload_bytes;
        assert_eq!(
            file_len, expected,
            "file size ({file_len}) != 48 + 4 + {text_len} + payload_bytes ({})",
            header.payload_bytes
        );

        let _ = std::fs::remove_file(&kvc_path);
    }

    // ── read_token_ids ────────────────────────────────────────────────────

    #[test]
    fn read_token_ids_round_trip() {
        let (engine, _path) = open_engine();
        let mut session = Session::new(engine.clone(), 2048).unwrap();
        session
            .restore_from_tokens(vec![10, 20, 30, 40, 50])
            .unwrap();

        let kvc_path =
            std::env::temp_dir().join(format!("ds4-kvc-token-ids-{}.kvc", std::process::id()));
        save_session(&kvc_path, &session, SaveReason::Manual).unwrap();

        let (tokens, ctx_size, total_count) = read_token_ids(&kvc_path).unwrap();
        assert_eq!(tokens, vec![10, 20, 30, 40, 50]);
        assert_eq!(ctx_size, 2048);
        assert_eq!(total_count, 5);

        let _ = std::fs::remove_file(&kvc_path);
    }

    #[test]
    fn read_token_ids_rejects_bad_magic() {
        let path =
            std::env::temp_dir().join(format!("ds4-kvc-bad-magic-ids-{}.kvc", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            let mut header = [0u8; 48];
            header[..3].copy_from_slice(b"XXX");
            f.write_all(&header).unwrap();
        }
        assert!(read_token_ids(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    // ── find_prefix_match ─────────────────────────────────────────────────

    #[test]
    fn find_prefix_match_finds_longest() {
        let (engine, _path) = open_engine();
        let dir = std::env::temp_dir().join(format!("ds4-kvc-prefix-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut s1 = Session::new(engine.clone(), 2048).unwrap();
        s1.restore_from_tokens(vec![1, 2, 3]).unwrap();
        save_session(&dir.join("a.kvc"), &s1, SaveReason::Manual).unwrap();

        let mut s2 = Session::new(engine.clone(), 2048).unwrap();
        s2.restore_from_tokens(vec![1, 2, 3, 4, 5]).unwrap();
        save_session(&dir.join("b.kvc"), &s2, SaveReason::Manual).unwrap();

        let result = find_prefix_match(&dir, &[1, 2, 3, 4, 5, 6, 7]);
        let (path, common) = result.unwrap();
        assert_eq!(common, 5);
        assert_eq!(path.file_name().unwrap(), "b.kvc");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_prefix_match_skips_non_prefix() {
        let (engine, _path) = open_engine();
        let dir =
            std::env::temp_dir().join(format!("ds4-kvc-prefix-no-match-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut s = Session::new(engine.clone(), 2048).unwrap();
        s.restore_from_tokens(vec![10, 20, 30]).unwrap();
        save_session(&dir.join("a.kvc"), &s, SaveReason::Manual).unwrap();

        let result = find_prefix_match(&dir, &[1, 2, 3]);
        assert!(result.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_prefix_match_allows_exact_match() {
        let (engine, _path) = open_engine();
        let dir = std::env::temp_dir().join(format!("ds4-kvc-prefix-exact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut s = Session::new(engine.clone(), 2048).unwrap();
        s.restore_from_tokens(vec![1, 2, 3]).unwrap();
        save_session(&dir.join("a.kvc"), &s, SaveReason::Manual).unwrap();

        // Exact match is now allowed
        let result = find_prefix_match(&dir, &[1, 2, 3]);
        let (path, len) = result.unwrap();
        assert_eq!(len, 3);
        assert!(path.ends_with("a.kvc"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_prefix_match_skips_longer_cached() {
        let (engine, _path) = open_engine();
        let dir =
            std::env::temp_dir().join(format!("ds4-kvc-prefix-longer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Cached session has MORE tokens than the query — not a prefix
        let mut s = Session::new(engine.clone(), 2048).unwrap();
        s.restore_from_tokens(vec![1, 2, 3, 4, 5]).unwrap();
        save_session(&dir.join("a.kvc"), &s, SaveReason::Manual).unwrap();

        let result = find_prefix_match(&dir, &[1, 2, 3]);
        assert!(result.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_prefix_match_empty_dir() {
        let dir = std::env::temp_dir().join(format!("ds4-kvc-prefix-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let result = find_prefix_match(&dir, &[1, 2, 3]);
        assert!(result.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── load_prefix_match ─────────────────────────────────────────────────

    #[test]
    fn load_prefix_match_returns_suffix() {
        let (engine, _path) = open_engine();
        let dir = std::env::temp_dir().join(format!("ds4-kvc-load-prefix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut s = Session::new(engine.clone(), 2048).unwrap();
        s.restore_from_tokens(vec![10, 20, 30, 40]).unwrap();
        save_session(&dir.join("cached.kvc"), &s, SaveReason::Manual).unwrap();

        let query = vec![10, 20, 30, 40, 50, 60];
        let result = load_prefix_match(&dir, &query, &engine).unwrap();
        let (session, suffix) = result.unwrap();

        assert_eq!(session.tokens(), &[10, 20, 30, 40]);
        assert_eq!(suffix, vec![50, 60]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_prefix_match_exact_returns_empty_suffix() {
        let (engine, _path) = open_engine();
        let dir = std::env::temp_dir().join(format!("ds4-kvc-load-exact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut s = Session::new(engine.clone(), 2048).unwrap();
        s.restore_from_tokens(vec![10, 20, 30]).unwrap();
        save_session(&dir.join("cached.kvc"), &s, SaveReason::Manual).unwrap();

        // Exact match returns session with empty suffix
        let query = vec![10, 20, 30];
        let (loaded, suffix) = load_prefix_match(&dir, &query, &engine).unwrap().unwrap();
        assert_eq!(loaded.tokens(), &[10, 20, 30]);
        assert!(suffix.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
