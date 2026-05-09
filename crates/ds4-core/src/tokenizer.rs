use anyhow::Result;
use std::collections::HashMap;

use crate::gguf::Value;

/// BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    tokens: Vec<String>,
    #[allow(dead_code)]
    scores: Vec<f32>,
    token_to_id: HashMap<String, u32>,
    /// Merge pairs mapped to their priority (index in merges list).
    merge_rank: HashMap<(String, String), usize>,
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

        // Build merge rank: pair -> priority (lower = higher priority)
        let merge_rank: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, (l, r))| ((l.clone(), r.clone()), i))
            .collect();

        let mut token_to_id = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
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
    /// Follows the standard BPE algorithm: iteratively merge the highest-priority
    /// adjacent pair until no more merges apply.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with byte-level tokens
        let mut pieces: Vec<String> = text.bytes().map(|b| format!("<0x{b:02X}>")).collect();

        // Iteratively merge the highest-priority adjacent pair
        loop {
            if pieces.len() < 2 {
                break;
            }

            // Find the adjacent pair with the lowest merge rank (highest priority)
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..pieces.len() - 1 {
                let pair = (&pieces[i], &pieces[i + 1]);
                if let Some(&rank) = self.merge_rank.get(&(pair.0.clone(), pair.1.clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break; // No more merges possible
            }

            // Merge the best pair
            let merged = format!("{}{}", pieces[best_idx], pieces[best_idx + 1]);
            pieces[best_idx] = merged;
            pieces.remove(best_idx + 1);
        }

        // Convert pieces to token IDs
        pieces
            .iter()
            .filter_map(|t| {
                self.token_to_id.get(t).copied().or_else(|| {
                    tracing::warn!("Token not in vocabulary: {t:?}");
                    None
                })
            })
            .collect()
    }

    /// Decode a token ID to its string representation.
    pub fn decode(&self, token_id: u32) -> &str {
        self.tokens
            .get(token_id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Decode a sequence of token IDs to text.
    pub fn decode_tokens(&self, token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .map(|&id| self.decode(id))
            .collect::<String>()
            .replace("<0x0A>", "\n")
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
