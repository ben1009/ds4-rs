use anyhow::Result;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

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
                            s.split_once(' ').map(|(l, r)| (l.to_string(), r.to_string()))
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
                if p != usize::MAX {
                    if let Some(&(nr, _)) = self.merge_rank.get(&(pieces[p], pieces[i])) {
                        heap.push((Reverse(nr), Reverse(p)));
                    }
                }
                if k != usize::MAX {
                    if let Some(&(nr, _)) = self.merge_rank.get(&(pieces[i], pieces[k])) {
                        heap.push((Reverse(nr), Reverse(i)));
                    }
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
        let abc_id = t.tokens.iter().position(|s| s == "<0x41><0x42><0x43>").unwrap() as u32;
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
}
