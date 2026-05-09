use anyhow::Result;
use std::collections::HashMap;

use crate::gguf::Value;

/// BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    tokens: Vec<String>,
    #[allow(dead_code)]
    scores: Vec<f32>,
    token_to_id: HashMap<String, u32>,
    /// Merge pairs (left_id, right_id) -> (rank, merged_id).
    merge_rank: HashMap<(u32, u32), (usize, u32)>,
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

        let scores: Vec<f32> = metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.to_array())
            .ok_or_else(|| anyhow::anyhow!("Missing tokenizer.ggml.scores"))?
            .iter()
            .map(|v| {
                v.to_f32()
                    .ok_or_else(|| anyhow::anyhow!("Invalid score value"))
            })
            .collect::<Result<Vec<_>>>()?;

        let merges: Vec<(String, String)> = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        v.to_string_val().and_then(|s| {
                            let parts: Vec<&str> = s.splitn(2, ' ').collect();
                            if parts.len() == 2 {
                                Some((parts[0].to_string(), parts[1].to_string()))
                            } else {
                                None
                            }
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

        tracing::info!(
            "Tokenizer: {} tokens, {} merges",
            tokens.len(),
            merge_rank.len()
        );

        Ok(Self {
            tokens,
            scores,
            token_to_id,
            merge_rank,
            bos_token,
            eos_token,
        })
    }

    /// Encode text into token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with byte-level token IDs
        let mut pieces: Vec<u32> = text
            .bytes()
            .map(|b| {
                let s = format!("<0x{b:02X}>");
                *self.token_to_id.get(&s).unwrap_or(&0)
            })
            .collect();

        // Iteratively merge the highest-priority adjacent pair
        loop {
            if pieces.len() < 2 {
                break;
            }

            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..pieces.len() - 1 {
                if let Some(&(rank, _)) = self.merge_rank.get(&(pieces[i], pieces[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break;
            }

            let (_, merged_id) = self.merge_rank[&(pieces[best_idx], pieces[best_idx + 1])];
            pieces[best_idx] = merged_id;
            pieces.remove(best_idx + 1);
        }

        pieces
    }

    /// Decode a token ID to its string representation.
    pub fn decode(&self, token_id: u32) -> &str {
        self.tokens
            .get(token_id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Decode a sequence of token IDs to text, converting byte-fallback tokens.
    pub fn decode_tokens(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            let tok = self.decode(id);
            if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&tok[3..5], 16) {
                    bytes.push(byte);
                    continue;
                }
            }
            bytes.extend_from_slice(tok.as_bytes());
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
