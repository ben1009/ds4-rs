use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use anyhow::Result;

use crate::gguf::Value;

/// BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    tokens: Vec<String>,
    /// Merge pairs (left_id, right_id) -> (rank, merged_id).
    merge_rank: HashMap<(u32, u32), (usize, u32)>,
    /// Pre-computed byte token IDs for fast encode.
    byte_to_id: [u32; 256],
    /// Reverse of `byte_to_id`, indexed by token ID. `Some(b)` means the
    /// token decodes to a single raw byte via byte-fallback; `None` means
    /// it's a regular text token.
    id_to_byte: Vec<Option<u8>>,
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

        let merges: Vec<(String, String)> = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        v.to_string_val().and_then(|s| {
                            s.split_once(' ')
                                .map(|(l, r)| (l.to_string(), r.to_string()))
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut token_to_id = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        // Build merge rank: (left_id, right_id) -> (rank, merged_id)
        let mut merge_rank = HashMap::new();
        for (i, (l, r)) in merges.iter().enumerate() {
            if let (Some(&lid), Some(&rid)) = (token_to_id.get(l), token_to_id.get(r)) {
                let merged = format!("{l}{r}");
                if let Some(&mid) = token_to_id.get(&merged) {
                    merge_rank.insert((lid, rid), (i, mid));
                }
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

        // Pre-compute byte token IDs and the reverse map. Missing byte tokens
        // mean the vocab is incompatible — bail rather than silently collapsing
        // bytes to id 0.
        let mut byte_to_id = [0u32; 256];
        let mut id_to_byte: Vec<Option<u8>> = vec![None; tokens.len()];
        for b in 0u8..=255 {
            let s = format!("<0x{b:02X}>");
            match token_to_id.get(&s) {
                Some(&id) => {
                    byte_to_id[b as usize] = id;
                    id_to_byte[id as usize] = Some(b);
                }
                None => anyhow::bail!("Byte token {s} missing from vocabulary"),
            }
        }

        tracing::info!(
            "Tokenizer: {} tokens, {} merges",
            tokens.len(),
            merge_rank.len()
        );

        Ok(Self {
            tokens,
            merge_rank,
            byte_to_id,
            id_to_byte,
            bos_token,
            eos_token,
        })
    }

    /// Encode text into token IDs using BPE.
    /// If `add_bos` is true, prepend the BOS token.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        if text.is_empty() && !add_bos {
            return Vec::new();
        }

        let mut pieces: Vec<u32> = text.bytes().map(|b| self.byte_to_id[b as usize]).collect();
        let n = pieces.len();

        if n >= 2 {
            // Doubly-linked list over piece indices; usize::MAX = end.
            let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
            let mut next: Vec<usize> = (0..n).map(|i| i + 1).collect();
            next[n - 1] = usize::MAX;

            // Min-heap by (rank, left-index) so equal-rank ties break leftward,
            // matching GPT-2 / llama.cpp BPE convention. Stale entries are
            // filtered when popped by re-checking the pair.
            let mut heap: BinaryHeap<(Reverse<usize>, Reverse<usize>)> = BinaryHeap::new();
            for i in 0..n - 1 {
                if let Some(&(rank, _)) = self.merge_rank.get(&(pieces[i], pieces[i + 1])) {
                    heap.push((Reverse(rank), Reverse(i)));
                }
            }

            while let Some((Reverse(rank), Reverse(i))) = heap.pop() {
                if pieces[i] == u32::MAX {
                    continue;
                }
                let j = next[i];
                if j == usize::MAX {
                    continue;
                }
                let Some(&(r, merged_id)) = self.merge_rank.get(&(pieces[i], pieces[j])) else {
                    continue;
                };
                if r != rank {
                    continue;
                }

                pieces[i] = merged_id;
                let k = next[j];
                next[i] = k;
                if k != usize::MAX {
                    prev[k] = i;
                }
                pieces[j] = u32::MAX;

                let p = prev[i];
                if p != usize::MAX
                    && let Some(&(nr, _)) = self.merge_rank.get(&(pieces[p], pieces[i]))
                {
                    heap.push((Reverse(nr), Reverse(p)));
                }
                if k != usize::MAX
                    && let Some(&(nr, _)) = self.merge_rank.get(&(pieces[i], pieces[k]))
                {
                    heap.push((Reverse(nr), Reverse(i)));
                }
            }

            let mut compacted = Vec::with_capacity(n);
            let mut idx = 0usize;
            while idx != usize::MAX {
                if pieces[idx] != u32::MAX {
                    compacted.push(pieces[idx]);
                }
                idx = next[idx];
            }
            pieces = compacted;
        }

        if add_bos {
            let mut result = Vec::with_capacity(pieces.len() + 1);
            result.push(self.bos_token);
            result.extend_from_slice(&pieces);
            result
        } else {
            pieces
        }
    }

    /// Decode a token ID to its string representation.
    pub fn decode(&self, token_id: u32) -> &str {
        self.tokens
            .get(token_id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Append the bytes for one token ID to `out`, decoding byte-fallback
    /// tokens (`<0xNN>`) into their raw byte. Lets the caller buffer and flush
    /// only at UTF-8 boundaries when streaming.
    pub fn append_token_bytes(&self, token_id: u32, out: &mut Vec<u8>) {
        let idx = token_id as usize;
        if let Some(Some(b)) = self.id_to_byte.get(idx) {
            out.push(*b);
            return;
        }
        out.extend_from_slice(self.decode(token_id).as_bytes());
    }

    /// Decode a sequence of token IDs to text, converting byte-fallback tokens.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(merges: &[(&str, &str)], extra: &[&str]) -> Tokenizer {
        let mut tokens: Vec<String> = (0u8..=255).map(|b| format!("<0x{b:02X}>")).collect();
        for (l, r) in merges {
            tokens.push(format!("{l}{r}"));
        }
        for t in extra {
            tokens.push((*t).to_string());
        }
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::new();
        for (i, (l, r)) in merges.iter().enumerate() {
            let lid = token_to_id[*l];
            let rid = token_to_id[*r];
            let mid = token_to_id[&format!("{l}{r}")];
            merge_rank.insert((lid, rid), (i, mid));
        }
        let mut byte_to_id = [0u32; 256];
        let mut id_to_byte: Vec<Option<u8>> = vec![None; tokens.len()];
        for b in 0u8..=255 {
            let id = token_to_id[&format!("<0x{b:02X}>")];
            byte_to_id[b as usize] = id;
            id_to_byte[id as usize] = Some(b);
        }
        Tokenizer {
            tokens,
            merge_rank,
            byte_to_id,
            id_to_byte,
            bos_token: 0,
            eos_token: 1,
        }
    }

    #[test]
    fn chained_merge() {
        // A=<0x41>, B=<0x42>, C=<0x43>; (A,B)->AB, (AB,C)->ABC
        let t = tok(&[("<0x41>", "<0x42>"), ("<0x41><0x42>", "<0x43>")], &[]);
        let out = t.encode("ABC", false);
        let abc_id = t
            .tokens
            .iter()
            .position(|s| s == "<0x41><0x42><0x43>")
            .unwrap() as u32;
        assert_eq!(out, vec![abc_id]);
    }

    #[test]
    fn no_merges_keeps_bytes() {
        let t = tok(&[], &[]);
        let out = t.encode("A", false);
        assert_eq!(out, vec![t.byte_to_id[b'A' as usize]]);
    }

    #[test]
    fn leftmost_wins_on_rank_tie() {
        // Two disjoint merges with the same rank: when both pairs appear,
        // the leftmost must merge first (GPT-2 / llama.cpp convention).
        // Build a vocab where (A,B) and (C,D) are both rank 0.
        let t = tok(&[("<0x41>", "<0x42>"), ("<0x43>", "<0x44>")], &[]);
        // Force equal rank: the helper numbers merges by index, so they
        // already differ. Patch merge_rank directly.
        let mut t = t;
        let ab = t.tokens.iter().position(|s| s == "<0x41><0x42>").unwrap() as u32;
        let cd = t.tokens.iter().position(|s| s == "<0x43><0x44>").unwrap() as u32;
        let a = t.byte_to_id[b'A' as usize];
        let b = t.byte_to_id[b'B' as usize];
        let c = t.byte_to_id[b'C' as usize];
        let d = t.byte_to_id[b'D' as usize];
        t.merge_rank.insert((a, b), (0, ab));
        t.merge_rank.insert((c, d), (0, cd));
        let out = t.encode("ABCD", false);
        assert_eq!(out, vec![ab, cd]);
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
    fn encode_with_bos_prepends() {
        let t = tok(&[], &[]);
        let out = t.encode("A", true);
        assert_eq!(out, vec![t.bos_token(), t.byte_to_id[b'A' as usize]]);
    }

    #[test]
    fn encode_single_byte_no_merges_needed() {
        // Single-byte input skips the heap loop entirely.
        let t = tok(&[("<0x41>", "<0x42>")], &[]);
        let out = t.encode("A", false);
        assert_eq!(out, vec![t.byte_to_id[b'A' as usize]]);
    }

    #[test]
    fn decode_out_of_range_returns_empty() {
        let t = tok(&[], &[]);
        assert_eq!(t.decode(u32::MAX), "");
    }

    #[test]
    fn decode_tokens_converts_byte_fallback() {
        let t = tok(&[], &[]);
        // "Hi" -> bytes 0x48 0x69 -> decoded back to "Hi".
        let ids = vec![t.byte_to_id[b'H' as usize], t.byte_to_id[b'i' as usize]];
        assert_eq!(t.decode_tokens(&ids), "Hi");
    }

    #[test]
    fn decode_tokens_preserves_literal_tokens() {
        let t = tok(&[], &["hello"]);
        let id = t.tokens.iter().position(|s| s == "hello").unwrap() as u32;
        assert_eq!(t.decode_tokens(&[id]), "hello");
    }

    #[test]
    fn append_token_bytes_byte_fallback_and_literal() {
        let t = tok(&[], &["xy"]);
        let mut out = Vec::new();
        t.append_token_bytes(t.byte_to_id[b'A' as usize], &mut out);
        let xy_id = t.tokens.iter().position(|s| s == "xy").unwrap() as u32;
        t.append_token_bytes(xy_id, &mut out);
        assert_eq!(out, b"Axy");
    }

    #[test]
    fn vocab_size_matches() {
        let t = tok(&[], &["a", "b", "c"]);
        // 256 byte tokens + 3 extras
        assert_eq!(t.vocab_size(), 259);
    }

    #[test]
    fn eos_and_bos_accessors() {
        let t = tok(&[], &[]);
        assert_eq!(t.bos_token(), 0);
        assert_eq!(t.eos_token(), 1);
    }

    #[test]
    fn round_trip_encode_decode_bytes() {
        let t = tok(&[], &[]);
        let text = "Hello, world!";
        let ids = t.encode(text, false);
        assert_eq!(t.decode_tokens(&ids), text);
    }

    #[test]
    fn from_metadata_builds_tokenizer() {
        // Merges pair "<0x48>" and "<0x69>" → merged token is their concat
        // "<0x48><0x69>" (that's how `format!("{l}{r}")` builds it inside
        // from_metadata). So the encoder will emit a single merged id for
        // the bytes of "Hi".
        let mut tokens: Vec<Value> = (0u8..=255)
            .map(|b| Value::String(format!("<0x{b:02X}>")))
            .collect();
        tokens.push(Value::String("<0x48><0x69>".to_string()));
        let merges = vec![Value::String("<0x48> <0x69>".to_string())];

        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
        m.insert("tokenizer.ggml.merges".to_string(), Value::Array(merges));
        m.insert("tokenizer.ggml.bos_token_id".to_string(), Value::U32(7));
        m.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U32(11));

        let t = Tokenizer::from_metadata(&m).unwrap();
        assert_eq!(t.bos_token(), 7);
        assert_eq!(t.eos_token(), 11);
        assert_eq!(t.vocab_size(), 257);

        let merged_id = 256u32;
        assert_eq!(t.encode("Hi", false), vec![merged_id]);
    }

    #[test]
    fn from_metadata_missing_byte_token_fails() {
        // Build a vocab that is missing one byte-fallback token.
        let mut tokens: Vec<Value> = (0u8..=254)
            .map(|b| Value::String(format!("<0x{b:02X}>")))
            .collect();
        // Skip 0xFF — should cause construction to fail.
        let mut m = std::collections::HashMap::new();
        m.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(tokens.clone()),
        );

        assert!(Tokenizer::from_metadata(&m).is_err());

        // And when we include all 256, it succeeds.
        tokens.push(Value::String("<0xFF>".to_string()));
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
        assert!(Tokenizer::from_metadata(&m).is_ok());
    }

    #[test]
    fn from_metadata_missing_tokens_key_fails() {
        let m = std::collections::HashMap::new();
        assert!(Tokenizer::from_metadata(&m).is_err());
    }

    #[test]
    fn round_trip_whitespace() {
        let t = tok(&[], &[]);
        let text = "  \t \n  ";
        let ids = t.encode(text, false);
        assert_eq!(t.decode_tokens(&ids), text);
    }

    #[test]
    fn round_trip_single_char() {
        let t = tok(&[], &[]);
        let ids = t.encode("x", false);
        assert_eq!(ids.len(), 1);
        assert_eq!(t.decode_tokens(&ids), "x");
    }

    #[test]
    fn round_trip_repeated_chars() {
        let t = tok(&[], &[]);
        let text = "aaaaaaaa";
        let ids = t.encode(text, false);
        assert_eq!(ids.len(), 8);
        assert_eq!(t.decode_tokens(&ids), text);
    }

    #[test]
    fn round_trip_multibyte_utf8() {
        let t = tok(&[], &[]);
        let text = "héllo 世界 🌍";
        let ids = t.encode(text, false);
        assert_eq!(ids.len(), text.len()); // one id per byte without merges
        assert_eq!(t.decode_tokens(&ids), text);
    }

    #[test]
    fn repeated_pair_merges_all_occurrences() {
        // (A,B) -> AB; "ABAB" should produce two AB tokens.
        let t = tok(&[("<0x41>", "<0x42>")], &[]);
        let ab = t.tokens.iter().position(|s| s == "<0x41><0x42>").unwrap() as u32;
        let out = t.encode("ABAB", false);
        assert_eq!(out, vec![ab, ab]);
    }

    #[test]
    fn lower_rank_merge_wins_over_higher() {
        // Rank 0: (B,C)->BC. Rank 1: (A,B)->AB.
        // For "ABC", BC must merge first, then no further merges apply.
        let t = tok(&[("<0x42>", "<0x43>"), ("<0x41>", "<0x42>")], &[]);
        let bc = t.tokens.iter().position(|s| s == "<0x42><0x43>").unwrap() as u32;
        let a = t.byte_to_id[b'A' as usize];
        let out = t.encode("ABC", false);
        assert_eq!(out, vec![a, bc]);
    }

    #[test]
    fn long_chain_of_merges() {
        // (A,B)->AB, (AB,C)->ABC, (ABC,D)->ABCD, (ABCD,E)->ABCDE
        let t = tok(
            &[
                ("<0x41>", "<0x42>"),
                ("<0x41><0x42>", "<0x43>"),
                ("<0x41><0x42><0x43>", "<0x44>"),
                ("<0x41><0x42><0x43><0x44>", "<0x45>"),
            ],
            &[],
        );
        let abcde = t
            .tokens
            .iter()
            .position(|s| s == "<0x41><0x42><0x43><0x44><0x45>")
            .unwrap() as u32;
        assert_eq!(t.encode("ABCDE", false), vec![abcde]);
    }

    #[test]
    fn append_token_bytes_decodes_byte_fallback() {
        let t = tok(&[], &[]);
        let mut out = Vec::new();
        t.append_token_bytes(t.byte_to_id[0xE4], &mut out);
        t.append_token_bytes(t.byte_to_id[0xB8], &mut out);
        t.append_token_bytes(t.byte_to_id[0x96], &mut out);
        assert_eq!(out, "世".as_bytes());
    }

    #[test]
    fn decode_tokens_handles_invalid_utf8_lossily() {
        let t = tok(&[], &[]);
        // Lone 0xFF is not valid UTF-8; from_utf8_lossy should produce the
        // replacement character rather than panic.
        let ids = vec![t.byte_to_id[0xFF]];
        let s = t.decode_tokens(&ids);
        assert!(s.contains('\u{FFFD}'));
    }

    #[test]
    fn decode_unknown_id_returns_empty_string() {
        let t = tok(&[], &[]);
        assert_eq!(t.decode(t.vocab_size() as u32 + 100), "");
    }

    #[test]
    fn decode_tokens_skips_unknown_ids_silently() {
        let t = tok(&[], &[]);
        let oob = t.vocab_size() as u32;
        // append_token_bytes for an out-of-range id falls through to decode(),
        // which yields "" — the byte stream is just empty for that id.
        let ids = vec![
            t.byte_to_id[b'X' as usize],
            oob,
            t.byte_to_id[b'Y' as usize],
        ];
        assert_eq!(t.decode_tokens(&ids), "XY");
    }

    #[test]
    fn from_metadata_default_bos_eos_when_absent() {
        let tokens: Vec<Value> = (0u8..=255)
            .map(|b| Value::String(format!("<0x{b:02X}>")))
            .collect();
        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
        let t = Tokenizer::from_metadata(&m).unwrap();
        assert_eq!(t.bos_token(), 0);
        assert_eq!(t.eos_token(), 1);
    }

    #[test]
    fn from_metadata_ignores_malformed_merge_entries() {
        let mut tokens: Vec<Value> = (0u8..=255)
            .map(|b| Value::String(format!("<0x{b:02X}>")))
            .collect();
        tokens.push(Value::String("<0x48><0x69>".to_string()));
        // First merge is malformed (no space); second is valid.
        let merges = vec![
            Value::String("nospace".to_string()),
            Value::String("<0x48> <0x69>".to_string()),
        ];
        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
        m.insert("tokenizer.ggml.merges".to_string(), Value::Array(merges));
        let t = Tokenizer::from_metadata(&m).unwrap();
        assert_eq!(t.encode("Hi", false), vec![256]);
    }

    #[test]
    fn from_metadata_works_without_merges() {
        let tokens: Vec<Value> = (0u8..=255)
            .map(|b| Value::String(format!("<0x{b:02X}>")))
            .collect();
        let mut m = std::collections::HashMap::new();
        m.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
        let t = Tokenizer::from_metadata(&m).unwrap();
        let ids = t.encode("hi", false);
        assert_eq!(t.decode_tokens(&ids), "hi");
    }

    #[test]
    fn encode_with_bos_and_merges_emits_bos_then_merged_ids() {
        let t = tok(&[("<0x41>", "<0x42>")], &[]);
        let ab = t.tokens.iter().position(|s| s == "<0x41><0x42>").unwrap() as u32;
        let ids = t.encode("ABAB", true);
        assert_eq!(ids, vec![t.bos_token(), ab, ab]);
    }
}
