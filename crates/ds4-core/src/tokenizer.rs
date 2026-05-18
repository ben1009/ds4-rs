//! BPE tokenizer for DeepSeek V4 Flash.
//!
//! Mirrors the JoyAI BPE path in antirez/ds4 ds4.c:
//! 1. The JoyAI pre-tokenizer splits raw input text into pieces using a fixed cascade of
//!    UTF-8-aware rules (`bpe_tokenize_text`).
//! 2. Each piece is GPT-2 byte-encoded (`byte_encode` / `gpt2_byte_to_codepoint`) so raw bytes
//!    become printable Unicode codepoints; the GGUF vocab and merges are stored in this encoded
//!    form.
//! 3. Each encoded piece is BPE-merged using the GGUF merge ranks; the final symbols are looked up
//!    in the vocab. Symbols missing from the vocab fall back to per-byte lookup over the encoded
//!    UTF-8 bytes.
//!
//! Decode reverses the byte encoding: tokens are emitted in encoded form,
//! `append_token_bytes` walks the token's UTF-8 codepoints and maps each one
//! back to its raw byte.
use std::collections::HashMap;

use anyhow::Result;

use crate::gguf::Value;

/// Number of distinct codepoints produced by `gpt2_byte_to_codepoint`.
/// 33–126 ∪ 161–172 ∪ 174–255 are kept as-is (188 codepoints); the remaining
/// 68 non-printable bytes are remapped to codepoints 256..324. The
/// codepoint range is therefore 0..324; `cp_to_byte` is sized by this max.
const N_BYTE_CP_MAX: usize = 324;

/// Cached GPT-2 byte → printable codepoint table. Built once on first use;
/// `byte_encode` and the test/synthetic helpers all read from it.
fn byte_to_cp() -> &'static [u32; 256] {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(build_byte_to_cp)
}

/// GPT-2 byte → printable codepoint table.
/// `byte_to_cp[b]` gives the encoded codepoint for raw byte `b`.
fn build_byte_to_cp() -> [u32; 256] {
    let mut out = [0u32; 256];
    let mut n: u32 = 0;
    for b in 0u32..256 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
        if printable {
            out[b as usize] = b;
        } else {
            out[b as usize] = 256 + n;
            n += 1;
        }
    }
    out
}

/// Inverse of `build_byte_to_cp`: codepoint → raw byte. Indexed by
/// codepoint; entries outside the GPT-2 byte map (cp ≥ N_BYTE_CP_MAX or
/// unmapped values inside the range) hold `-1`. Replaces the per-decode
/// HashMap lookup with an O(1) array index on the hot path.
fn build_cp_to_byte(byte_to_cp: &[u32; 256]) -> [i16; N_BYTE_CP_MAX] {
    let mut out = [-1i16; N_BYTE_CP_MAX];
    for (b, &cp) in byte_to_cp.iter().enumerate() {
        out[cp as usize] = b as i16;
    }
    out
}

/// BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    /// Raw vocab strings, in the GPT-2 byte-encoded form used by the GGUF.
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// Merge ranks: encoded merge string `"left right"` → rank index.
    /// Lower rank = earlier in the merge file = applied first.
    merge_rank: HashMap<String, usize>,
    /// codepoint → raw byte (inverse of GPT-2 byte encoding) for decode.
    /// Indexed by codepoint; `-1` = no mapping (codepoint is part of a
    /// special token, not a byte-encoded character).
    cp_to_byte: [i16; N_BYTE_CP_MAX],
    bos_token: u32,
    eos_token: u32,
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata.
    pub fn from_metadata(metadata: &std::collections::HashMap<String, Value>) -> Result<Self> {
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_array())
            .ok_or_else(|| anyhow::anyhow!("Missing tokenizer.ggml.tokens"))?
            .iter()
            .map(|v| {
                v.to_string_val()
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow::anyhow!("Invalid token value"))
            })
            .collect::<Result<Vec<_>>>()?;

        let merges_raw: Vec<String> = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string_val().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        let mut merge_rank: HashMap<String, usize> = HashMap::with_capacity(merges_raw.len());
        for (i, m) in merges_raw.iter().enumerate() {
            // Merges are stored as "left right". A rank of 0 = highest
            // priority. Skip malformed entries (no space) silently — they
            // can't apply during encode anyway.
            if m.contains(' ') {
                merge_rank.insert(m.clone(), i);
            }
        }

        let bos_token = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32())
            .unwrap_or(0);
        let eos_token = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32())
            .unwrap_or(1);

        let byte_to_cp = byte_to_cp();
        let cp_to_byte = build_cp_to_byte(byte_to_cp);

        // Validate that every raw byte has a single-codepoint token in the
        // vocab. Missing byte tokens mean the vocab is incompatible — bail
        // rather than silently dropping bytes during encode.
        let mut buf = [0u8; 4];
        for b in 0u32..256 {
            let cp = byte_to_cp[b as usize];
            let ch = char::from_u32(cp)
                .ok_or_else(|| anyhow::anyhow!("Invalid GPT-2 byte codepoint {cp}"))?;
            let s = ch.encode_utf8(&mut buf);
            if !token_to_id.contains_key(s) {
                anyhow::bail!(
                    "Byte token for raw byte 0x{b:02X} (codepoint U+{cp:04X}) missing from vocabulary"
                );
            }
        }

        tracing::info!(
            "Tokenizer: {} tokens, {} merges (joyai-llm)",
            tokens.len(),
            merge_rank.len()
        );

        Ok(Self {
            tokens,
            token_to_id,
            merge_rank,
            cp_to_byte,
            bos_token,
            eos_token,
        })
    }

    /// Encode text into token IDs. Prepends BOS if `add_bos` is true.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            out.push(self.bos_token);
        }
        for piece in joyai_split(text) {
            self.encode_piece(piece, &mut out);
        }
        out
    }

    /// BPE-encode one pre-tokenized piece (raw text) into the output stream.
    fn encode_piece(&self, raw_piece: &str, out: &mut Vec<u32>) {
        if raw_piece.is_empty() {
            return;
        }
        let encoded = byte_encode(raw_piece.as_bytes());

        // Initial symbol list: one entry per UTF-8 char of `encoded`, stored
        // as (byte_offset, byte_len). Working on byte indices avoids building
        // owned Strings inside the merge loop.
        let mut sym: Vec<(usize, usize)> = encoded
            .char_indices()
            .map(|(i, ch)| (i, ch.len_utf8()))
            .collect();

        // Repeatedly merge the lowest-rank adjacent pair until none apply.
        // Quadratic in symbol count; pieces are short (single words /
        // identifiers) so this matches `bpe_emit_piece` in ds4.c without the
        // extra heap-bookkeeping the byte-level path needs. The lookup key
        // is built into a single reused `String` buffer to keep allocations
        // out of the inner loop.
        let mut key = String::new();
        loop {
            let mut best_i: Option<usize> = None;
            let mut best_rank = usize::MAX;
            for i in 0..sym.len().saturating_sub(1) {
                let (lo_a, len_a) = sym[i];
                let (lo_b, len_b) = sym[i + 1];
                key.clear();
                key.push_str(&encoded[lo_a..lo_a + len_a]);
                key.push(' ');
                key.push_str(&encoded[lo_b..lo_b + len_b]);
                if let Some(&rank) = self.merge_rank.get(&key)
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_i = Some(i);
                }
            }
            let Some(i) = best_i else {
                break;
            };
            let (lo_a, len_a) = sym[i];
            let (_lo_b, len_b) = sym[i + 1];
            sym[i] = (lo_a, len_a + len_b);
            sym.remove(i + 1);
        }

        // Emit token IDs. If a merged symbol isn't in the vocab, fall back
        // to per-codepoint lookup. The encoded string contains codepoints
        // outside the ASCII range (the GPT-2 byte map remaps non-printable
        // bytes to U+0100..U+0143, which are 2-byte UTF-8), so we must
        // iterate codepoints, not raw UTF-8 bytes — splitting a multi-byte
        // codepoint yields invalid UTF-8 and the lookup silently drops.
        let mut buf = [0u8; 4];
        for (lo, len) in sym {
            let s = &encoded[lo..lo + len];
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
            } else {
                for ch in s.chars() {
                    let bs = ch.encode_utf8(&mut buf);
                    if let Some(&id) = self.token_to_id.get(bs) {
                        out.push(id);
                    }
                }
            }
        }
    }

    /// Decode a token ID to its (encoded) string representation.
    pub fn decode(&self, token_id: u32) -> &str {
        self.tokens
            .get(token_id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Append the raw (decoded) bytes for one token ID to `out`. Reverses
    /// the GPT-2 byte encoding so the streamed output is in the user's
    /// original encoding (typically UTF-8).
    pub fn append_token_bytes(&self, token_id: u32, out: &mut Vec<u8>) {
        let s = self.decode(token_id);
        for ch in s.chars() {
            let cp = ch as u32;
            let mapped = if (cp as usize) < N_BYTE_CP_MAX {
                let v = self.cp_to_byte[cp as usize];
                if v >= 0 { Some(v as u8) } else { None }
            } else {
                None
            };
            match mapped {
                Some(b) => out.push(b),
                None => {
                    // Codepoint outside the GPT-2 byte-map range — emit it as
                    // its UTF-8 bytes. This shouldn't normally happen for a
                    // healthy DS4 vocab, but special-token strings (e.g.
                    // `<｜begin▁of▁sentence｜>`) sit outside the byte map and
                    // we fall through to their raw UTF-8.
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }

    /// Decode a sequence of token IDs to text.
    pub fn decode_tokens(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            self.append_token_bytes(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn bos_token(&self) -> u32 {
        self.bos_token
    }

    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte encoding
// ---------------------------------------------------------------------------

/// Map raw bytes to a printable UTF-8 string per the GPT-2 / DS4 convention
/// (`byte_encode` in ds4.c). Each input byte becomes exactly one Unicode
/// codepoint in the encoded output.
fn byte_encode(bytes: &[u8]) -> String {
    let table = byte_to_cp();
    let mut out = String::with_capacity(bytes.len() * 2);
    let mut buf = [0u8; 4];
    for &b in bytes {
        let cp = table[b as usize];
        // Every codepoint produced by the table is a valid Unicode scalar;
        // unwrap is sound here.
        let ch = char::from_u32(cp).expect("byte codepoint is valid");
        out.push_str(ch.encode_utf8(&mut buf));
    }
    out
}

fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if (c & 0xe0) == 0xc0 {
        2
    } else if (c & 0xf0) == 0xe0 {
        3
    } else if (c & 0xf8) == 0xf0 {
        4
    } else {
        1
    }
}

/// Test-only helper: build the canonical 256 GPT-2 byte-encoded token
/// strings, in raw-byte order. Synthetic GGUF fixtures use this so
/// `Tokenizer::from_metadata` succeeds during engine/session tests that
/// don't otherwise care about the tokenizer.
#[cfg(test)]
pub(crate) fn synthetic_byte_tokens() -> Vec<String> {
    let table = byte_to_cp();
    let mut out = Vec::with_capacity(256);
    let mut buf = [0u8; 4];
    for b in 0u32..256 {
        let cp = table[b as usize];
        let ch = char::from_u32(cp).expect("byte codepoint is valid");
        out.push(ch.encode_utf8(&mut buf).to_string());
    }
    out
}

// ---------------------------------------------------------------------------
// JoyAI pre-tokenizer
// ---------------------------------------------------------------------------

#[inline]
fn ascii_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}
#[inline]
fn ascii_digit(c: u8) -> bool {
    c.is_ascii_digit()
}
#[inline]
fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}
#[inline]
fn ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}
#[inline]
fn joyai_ascii_punct_symbol(c: u8) -> bool {
    matches!(c, 0x21..=0x2f | 0x3a..=0x40 | 0x5b..=0x60 | 0x7b..=0x7e)
}

#[inline]
fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

/// Step to the next UTF-8 character boundary at or after `pos`. Mirrors
/// `next_utf8_char` in ds4.c — if the leading-byte width runs past `len`,
/// advance by 1 to make progress on malformed input.
fn next_utf8_char(bytes: &[u8], pos: usize) -> usize {
    let n = utf8_len_from_first_byte(bytes[pos]);
    if pos + n > bytes.len() {
        pos + 1
    } else {
        pos + n
    }
}

/// Decode the codepoint at `pos`, returning `(cp, next_pos)`. Mirrors
/// `utf8_peek_one`.
fn utf8_peek_one(bytes: &[u8], pos: usize) -> (u32, usize) {
    let c0 = bytes[pos];
    let n_raw = utf8_len_from_first_byte(c0);
    let n = if pos + n_raw > bytes.len() { 1 } else { n_raw };
    let next = pos + n;
    let cp = match n {
        1 => c0 as u32,
        2 => (((c0 & 0x1f) as u32) << 6) | ((bytes[pos + 1] & 0x3f) as u32),
        3 => {
            (((c0 & 0x0f) as u32) << 12)
                | (((bytes[pos + 1] & 0x3f) as u32) << 6)
                | ((bytes[pos + 2] & 0x3f) as u32)
        }
        _ => {
            (((c0 & 0x07) as u32) << 18)
                | (((bytes[pos + 1] & 0x3f) as u32) << 12)
                | (((bytes[pos + 2] & 0x3f) as u32) << 6)
                | ((bytes[pos + 3] & 0x3f) as u32)
        }
    };
    (cp, next)
}

/// Mirrors `joyai_letter_like_at`: ASCII letters, plus any non-ASCII byte.
/// CJK / hira / kata are isolated by their own rule earlier in the cascade.
fn joyai_letter_like_at(bytes: &[u8], pos: usize) -> bool {
    let c = bytes[pos];
    if c < 128 { ascii_alpha(c) } else { true }
}

fn joyai_consume_letters(bytes: &[u8], pos: usize) -> usize {
    let mut p = pos;
    while p < bytes.len() && joyai_letter_like_at(bytes, p) {
        p = next_utf8_char(bytes, p);
    }
    p
}

fn joyai_cjk_at(bytes: &[u8], pos: usize) -> bool {
    if bytes[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(bytes, pos);
    utf8_is_cjk_hira_kata(cp)
}

/// Iterator yielding pre-tokenized pieces from `text`. Each piece is a
/// `&str` borrow into the input; the cascade matches `bpe_tokenize_text`
/// in ds4.c byte-for-byte.
fn joyai_split(text: &str) -> JoyaiSplit<'_> {
    JoyaiSplit { text, pos: 0 }
}

struct JoyaiSplit<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Iterator for JoyaiSplit<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let bytes = self.text.as_bytes();
        let len = bytes.len();
        if self.pos >= len {
            return None;
        }
        let start = self.pos;
        let mut pos = self.pos;
        let c = bytes[pos];

        if ascii_digit(c) {
            // \p{N}{1,3}
            let mut n = 0;
            while pos < len && ascii_digit(bytes[pos]) && n < 3 {
                pos += 1;
                n += 1;
            }
        } else if joyai_cjk_at(bytes, pos) {
            // [CJK/Hiragana/Katakana]+
            loop {
                pos = next_utf8_char(bytes, pos);
                if pos >= len || !joyai_cjk_at(bytes, pos) {
                    break;
                }
            }
        } else if joyai_ascii_punct_symbol(c) && pos + 1 < len && ascii_alpha(bytes[pos + 1]) {
            // [P/S][A-Za-z]+
            pos += 1;
            while pos < len && ascii_alpha(bytes[pos]) {
                pos += 1;
            }
        } else if joyai_letter_like_at(bytes, pos) {
            // \p{L}+
            pos = joyai_consume_letters(bytes, pos);
        } else if !ascii_newline(c)
            && !joyai_ascii_punct_symbol(c)
            && pos + 1 < len
            && joyai_letter_like_at(bytes, pos + 1)
        {
            // [^\r\n\p{P}\p{S}] then \p{L}+ — the leading byte joins the run.
            pos += 1;
            pos = joyai_consume_letters(bytes, pos);
        } else if c == b' ' && pos + 1 < len && joyai_ascii_punct_symbol(bytes[pos + 1]) {
            // " " then [P/S]+ then [\r\n]*
            pos += 1;
            while pos < len && joyai_ascii_punct_symbol(bytes[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(bytes[pos]) {
                pos += 1;
            }
        } else if joyai_ascii_punct_symbol(c) {
            // [P/S]+ then [\r\n]*
            while pos < len && joyai_ascii_punct_symbol(bytes[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(bytes[pos]) {
                pos += 1;
            }
        } else if ascii_space(c) {
            // Whitespace runs. If the run contains a newline, split on the
            // last newline. Otherwise, if the trailing space precedes a
            // letter or punct (and the run is at least 2 spaces), give that
            // single trailing space to the next piece.
            let mut p = pos;
            let mut last_newline_end = 0usize;
            while p < len && ascii_space(bytes[p]) {
                let sc = bytes[p];
                p += 1;
                if ascii_newline(sc) {
                    last_newline_end = p;
                }
            }
            if last_newline_end != 0 {
                pos = last_newline_end;
            } else if p < len
                && p > pos + 1
                && (joyai_letter_like_at(bytes, p) || joyai_ascii_punct_symbol(bytes[p]))
            {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(bytes, pos);
        }

        // Any rule that didn't advance falls back to one UTF-8 char so we
        // can't loop forever on weird input.
        if pos == start {
            pos = next_utf8_char(bytes, pos);
        }

        self.pos = pos;
        Some(&self.text[start..pos])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tokenizer from a list of (left, right) merge pairs and a list
    /// of extra (non-merge) tokens. The vocab is initialised with all 256
    /// GPT-2 byte codepoints in canonical order so byte fallback works.
    fn tok(merges: &[(&str, &str)], extra: &[&str]) -> Tokenizer {
        let byte_to_cp = byte_to_cp();
        let cp_to_byte = build_cp_to_byte(byte_to_cp);

        // Canonical byte tokens first: one single-codepoint token per byte
        // value in raw-byte order. This means tokens[b] is the encoded token
        // for raw byte b — convenient for tests that want predictable IDs.
        let mut tokens: Vec<String> = Vec::with_capacity(256);
        let mut buf = [0u8; 4];
        for b in 0u32..256 {
            let cp = byte_to_cp[b as usize];
            let ch = char::from_u32(cp).unwrap();
            tokens.push(ch.encode_utf8(&mut buf).to_string());
        }
        for (l, r) in merges {
            tokens.push(format!("{l}{r}"));
        }
        for t in extra {
            tokens.push((*t).to_string());
        }

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merge_rank: HashMap<String, usize> = HashMap::new();
        for (i, (l, r)) in merges.iter().enumerate() {
            merge_rank.insert(format!("{l} {r}"), i);
        }

        Tokenizer {
            tokens,
            token_to_id,
            merge_rank,
            cp_to_byte,
            bos_token: 0,
            eos_token: 1,
        }
    }

    /// Look up the token ID for the single-byte vocab entry that encodes
    /// raw byte `b`. Replaces the old `byte_token_id` array (which was
    /// derivable from the vocab and only used in tests).
    fn byte_id(t: &Tokenizer, b: u8) -> u32 {
        let s = enc(b);
        t.token_to_id[&s]
    }

    /// Encoded form of a single raw byte (one-character string).
    fn enc(b: u8) -> String {
        let table = byte_to_cp();
        let cp = table[b as usize];
        let mut buf = [0u8; 4];
        char::from_u32(cp)
            .unwrap()
            .encode_utf8(&mut buf)
            .to_string()
    }

    fn enc_str(s: &str) -> String {
        byte_encode(s.as_bytes())
    }

    // ----- byte encoding -----

    #[test]
    fn byte_encoding_keeps_printable_ascii() {
        let table = byte_to_cp();
        for b in *b"Az5!~" {
            assert_eq!(table[b as usize], b as u32);
        }
    }

    #[test]
    fn byte_encoding_remaps_nonprintable_bytes() {
        let table = byte_to_cp();
        // 0x00 and 0x20 (space) are remapped to codepoints >= 256.
        assert!(table[0] >= 256);
        assert!(table[0x20] >= 256);
        assert!(table[0x7f] >= 256);
    }

    #[test]
    fn byte_encode_is_invertible() {
        // Every byte 0..=255 round-trips through encode → cp_to_byte.
        let byte_to_cp = byte_to_cp();
        let cp_to_byte = build_cp_to_byte(byte_to_cp);
        for b in 0u8..=255 {
            let cp = byte_to_cp[b as usize];
            assert_eq!(cp_to_byte[cp as usize], b as i16);
        }
    }

    // ----- JoyAI pre-tokenizer -----

    fn pieces(text: &str) -> Vec<&str> {
        joyai_split(text).collect()
    }

    #[test]
    fn joyai_digit_run_caps_at_three() {
        assert_eq!(pieces("12345"), vec!["123", "45"]);
    }

    #[test]
    fn joyai_letters_run_together() {
        assert_eq!(pieces("hello"), vec!["hello"]);
    }

    #[test]
    fn joyai_punct_then_letters_one_piece() {
        // ".foo" → ".foo" (one piece, leading punct attaches to letters).
        assert_eq!(pieces(".foo"), vec![".foo"]);
    }

    #[test]
    fn joyai_punct_run_swallows_trailing_newlines() {
        // "{};\n\n" splits as a single punct+newline run.
        assert_eq!(pieces("{};\n\n"), vec!["{};\n\n"]);
    }

    #[test]
    fn joyai_leading_space_attaches_to_word() {
        // " hello" → " hello"
        assert_eq!(pieces(" hello"), vec![" hello"]);
    }

    #[test]
    fn joyai_indent_then_word_keeps_three_spaces_then_space_word() {
        // "    int" → "   " then " int" (matches the C-source comment).
        assert_eq!(pieces("    int"), vec!["   ", " int"]);
    }

    #[test]
    fn joyai_whitespace_splits_on_last_newline() {
        // "  \n  foo" → "  \n" then "  foo" (well, " foo" after leading
        // space attaches; " " stays as its own piece because there's only
        // one space before the letter so the rule doesn't fire).
        // Actually: after the newline we have "  foo" — two spaces then
        // letters, and the leading-space rule fires (p > pos+1).
        let p = pieces("  \n  foo");
        assert_eq!(p[0], "  \n");
        assert_eq!(p[p.len() - 1], " foo");
    }

    #[test]
    fn joyai_cjk_run_isolates_from_punct() {
        // CJK at the start of a piece consumes only CJK; trailing ASCII
        // punctuation breaks the run. Matches `bpe_tokenize_text` in ds4.c
        // — note `joyai_letter_like_at` treats *any* non-ASCII byte as a
        // letter, so CJK is only isolated when it's the *first* byte of a
        // piece. Once inside the letter rule, CJK and Latin letters pool
        // together.
        let p = pieces("世界!");
        assert_eq!(p, vec!["世界", "!"]);
    }

    #[test]
    fn joyai_pieces_concat_to_input() {
        // Across a varied input the pieces must reassemble into the
        // original text.
        let text = "let x = 42; // hello, 世界!\n  return x + 1\n";
        let joined: String = joyai_split(text).collect();
        assert_eq!(joined, text);
    }

    // ----- BPE encode -----

    #[test]
    fn encode_no_merges_produces_one_token_per_byte() {
        let t = tok(&[], &[]);
        let out = t.encode("Hi", false);
        // "Hi" → JoyAI splits into one piece "Hi"; BPE has no merges so
        // each byte (= each encoded codepoint) becomes its own token.
        assert_eq!(out, vec![byte_id(&t, b'H'), byte_id(&t, b'i')]);
    }

    #[test]
    fn encode_chained_merge() {
        // Build merges that fold "ABC" → "AB" → "ABC".
        let merges = &[
            (enc(b'A'), enc(b'B')),
            (format!("{}{}", enc(b'A'), enc(b'B')), enc(b'C')),
        ];
        let merges_refs: Vec<(&str, &str)> = merges
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let t = tok(&merges_refs, &[]);

        let abc = format!("{}{}{}", enc(b'A'), enc(b'B'), enc(b'C'));
        let abc_id = t.token_to_id[&abc];
        assert_eq!(t.encode("ABC", false), vec![abc_id]);
    }

    #[test]
    fn encode_with_bos_prepends() {
        let t = tok(&[], &[]);
        let out = t.encode("A", true);
        assert_eq!(out, vec![t.bos_token(), byte_id(&t, b'A')]);
    }

    #[test]
    fn encode_empty_no_bos() {
        let t = tok(&[], &[]);
        assert!(t.encode("", false).is_empty());
    }

    #[test]
    fn encode_empty_with_bos() {
        let t = tok(&[], &[]);
        assert_eq!(t.encode("", true), vec![t.bos_token()]);
    }

    #[test]
    fn lower_rank_merge_wins() {
        // Rank 0: (B,C)→BC. Rank 1: (A,B)→AB.
        // "ABC" should merge BC first, leaving [A, BC].
        let merges = &[(enc(b'B'), enc(b'C')), (enc(b'A'), enc(b'B'))];
        let merges_refs: Vec<(&str, &str)> = merges
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let t = tok(&merges_refs, &[]);
        let bc = t.token_to_id[&format!("{}{}", enc(b'B'), enc(b'C'))];
        let a = byte_id(&t, b'A');
        assert_eq!(t.encode("ABC", false), vec![a, bc]);
    }

    #[test]
    fn repeated_pair_merges_all_occurrences() {
        // (A,B)→AB; "AB AB" should produce [AB, " ", AB] — though the JoyAI
        // splitter keeps the leading space attached to the second piece, so
        // we get ["AB", " AB"] and AB doesn't appear in the second piece's
        // BPE. Use a single piece without a space instead.
        let merges = &[(enc(b'A'), enc(b'B'))];
        let merges_refs: Vec<(&str, &str)> = merges
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let t = tok(&merges_refs, &[]);
        let ab = t.token_to_id[&format!("{}{}", enc(b'A'), enc(b'B'))];
        assert_eq!(t.encode("ABAB", false), vec![ab, ab]);
    }

    // ----- decode -----

    #[test]
    fn decode_byte_token_round_trips() {
        let t = tok(&[], &[]);
        let mut out = Vec::new();
        t.append_token_bytes(byte_id(&t, b'X'), &mut out);
        assert_eq!(out, b"X");
    }

    #[test]
    fn decode_multibyte_utf8_via_byte_tokens() {
        let t = tok(&[], &[]);
        // "世" is 3 bytes 0xE4 0xB8 0x96.
        let mut out = Vec::new();
        for &b in "世".as_bytes() {
            t.append_token_bytes(byte_id(&t, b), &mut out);
        }
        assert_eq!(out, "世".as_bytes());
    }

    #[test]
    fn decode_unknown_token_id_emits_nothing() {
        let t = tok(&[], &[]);
        let mut out = Vec::new();
        t.append_token_bytes(u32::MAX, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn decode_special_token_falls_through_as_utf8() {
        // Tokens whose codepoints aren't in the GPT-2 byte map (e.g. real
        // CJK characters used as special tokens) should emit their UTF-8
        // bytes verbatim.
        let t = tok(&[], &["<｜foo｜>"]);
        let id = t.token_to_id["<｜foo｜>"];
        let mut out = Vec::new();
        t.append_token_bytes(id, &mut out);
        // The "<", "|", ">" are in the GPT-2 byte map and decode to those
        // bytes; the "｜" (FULLWIDTH VERTICAL LINE, U+FF5C) is outside the
        // map and falls through as its 3 UTF-8 bytes.
        assert_eq!(out, "<｜foo｜>".as_bytes());
    }

    // ----- round-trip -----

    #[test]
    fn round_trip_ascii() {
        let t = tok(&[], &[]);
        let text = "Hello, world!";
        let ids = t.encode(text, false);
        assert_eq!(t.decode_tokens(&ids), text);
    }

    #[test]
    fn round_trip_whitespace_and_newlines() {
        let t = tok(&[], &[]);
        let text = "  \t \n  ";
        let ids = t.encode(text, false);
        assert_eq!(t.decode_tokens(&ids), text);
    }

    #[test]
    fn round_trip_multibyte_utf8() {
        let t = tok(&[], &[]);
        let text = "héllo 世界 🌍";
        let ids = t.encode(text, false);
        assert_eq!(t.decode_tokens(&ids), text);
    }

    #[test]
    fn round_trip_repeated_chars() {
        let t = tok(&[], &[]);
        let text = "aaaaaaaa";
        let ids = t.encode(text, false);
        assert_eq!(t.decode_tokens(&ids), text);
    }

    // ----- from_metadata -----

    #[test]
    fn from_metadata_builds_tokenizer() {
        // Build a vocab: 256 byte tokens (in encoded form) + one merged
        // token for the encoded "Hi" pair.
        let byte_to_cp = byte_to_cp();
        let mut buf = [0u8; 4];
        let mut tokens_v: Vec<Value> = (0u32..256)
            .map(|b| {
                let cp = byte_to_cp[b as usize];
                let s = char::from_u32(cp)
                    .unwrap()
                    .encode_utf8(&mut buf)
                    .to_string();
                Value::String(s)
            })
            .collect();
        let h_enc = enc(b'H');
        let i_enc = enc(b'i');
        tokens_v.push(Value::String(format!("{h_enc}{i_enc}")));
        let merges = vec![Value::String(format!("{h_enc} {i_enc}"))];

        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens_v));
        m.insert("tokenizer.ggml.merges".to_string(), Value::Array(merges));
        m.insert("tokenizer.ggml.bos_token_id".to_string(), Value::U32(7));
        m.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U32(11));

        let t = Tokenizer::from_metadata(&m).unwrap();
        assert_eq!(t.bos_token(), 7);
        assert_eq!(t.eos_token(), 11);
        assert_eq!(t.vocab_size(), 257);
        assert_eq!(t.encode("Hi", false), vec![256]);
    }

    #[test]
    fn from_metadata_missing_byte_token_fails() {
        // Vocab with all but one byte codepoint.
        let byte_to_cp = byte_to_cp();
        let mut buf = [0u8; 4];
        let mut tokens_v: Vec<Value> = Vec::with_capacity(255);
        for b in 0u32..256 {
            if b == 0xFF {
                continue;
            }
            let cp = byte_to_cp[b as usize];
            let s = char::from_u32(cp)
                .unwrap()
                .encode_utf8(&mut buf)
                .to_string();
            tokens_v.push(Value::String(s));
        }
        let mut m = std::collections::HashMap::new();
        m.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(tokens_v.clone()),
        );
        assert!(Tokenizer::from_metadata(&m).is_err());

        // Adding the missing entry makes load succeed.
        let cp = byte_to_cp[0xFF];
        let s = char::from_u32(cp)
            .unwrap()
            .encode_utf8(&mut buf)
            .to_string();
        tokens_v.push(Value::String(s));
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens_v));
        assert!(Tokenizer::from_metadata(&m).is_ok());
    }

    #[test]
    fn from_metadata_missing_tokens_key_fails() {
        let m = std::collections::HashMap::new();
        assert!(Tokenizer::from_metadata(&m).is_err());
    }

    #[test]
    fn from_metadata_default_bos_eos_when_absent() {
        let byte_to_cp = byte_to_cp();
        let mut buf = [0u8; 4];
        let tokens_v: Vec<Value> = (0u32..256)
            .map(|b| {
                let cp = byte_to_cp[b as usize];
                let s = char::from_u32(cp)
                    .unwrap()
                    .encode_utf8(&mut buf)
                    .to_string();
                Value::String(s)
            })
            .collect();
        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens_v));
        let t = Tokenizer::from_metadata(&m).unwrap();
        assert_eq!(t.bos_token(), 0);
        assert_eq!(t.eos_token(), 1);
    }

    #[test]
    fn from_metadata_ignores_malformed_merge_entries() {
        let byte_to_cp = byte_to_cp();
        let mut buf = [0u8; 4];
        let mut tokens_v: Vec<Value> = (0u32..256)
            .map(|b| {
                let cp = byte_to_cp[b as usize];
                let s = char::from_u32(cp)
                    .unwrap()
                    .encode_utf8(&mut buf)
                    .to_string();
                Value::String(s)
            })
            .collect();
        let h_enc = enc(b'H');
        let i_enc = enc(b'i');
        tokens_v.push(Value::String(format!("{h_enc}{i_enc}")));

        // First merge is malformed (no space); second is valid.
        let merges = vec![
            Value::String("nospace".to_string()),
            Value::String(format!("{h_enc} {i_enc}")),
        ];
        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens_v));
        m.insert("tokenizer.ggml.merges".to_string(), Value::Array(merges));
        let t = Tokenizer::from_metadata(&m).unwrap();
        assert_eq!(t.encode("Hi", false), vec![256]);
    }

    #[test]
    fn from_metadata_works_without_merges() {
        let byte_to_cp = byte_to_cp();
        let mut buf = [0u8; 4];
        let tokens_v: Vec<Value> = (0u32..256)
            .map(|b| {
                let cp = byte_to_cp[b as usize];
                let s = char::from_u32(cp)
                    .unwrap()
                    .encode_utf8(&mut buf)
                    .to_string();
                Value::String(s)
            })
            .collect();
        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens_v));
        let t = Tokenizer::from_metadata(&m).unwrap();
        let ids = t.encode("hi", false);
        assert_eq!(t.decode_tokens(&ids), "hi");
    }

    #[test]
    fn vocab_size_matches() {
        let t = tok(&[], &["a", "b", "c"]);
        assert_eq!(t.vocab_size(), 256 + 3);
    }

    // Reference: byte_encode result for a known string should match the
    // formula in `gpt2_byte_to_codepoint`.
    #[test]
    fn enc_str_known_vector() {
        // Space (0x20) is non-printable in this scheme. It maps to the first
        // remapped codepoint, 256 + n where n = position-among-non-printables.
        // 0..=32 are positions 0..32 (33 values), so 0x20 == 32 → cp 256+32=288 (U+0120 = "Ġ").
        let s = enc_str(" ");
        assert_eq!(s, "Ġ");
    }
}
