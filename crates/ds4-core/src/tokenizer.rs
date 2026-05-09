use anyhow::Result;
use std::collections::HashMap;

use crate::gguf::Value;

/// BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    tokens: Vec<String>,
    #[allow(dead_code)]
    scores: Vec<f32>,
    token_to_id: HashMap<String, u32>,
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
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
            merges.len()
        );

        Ok(Self {
            tokens,
            scores,
            token_to_id,
            merges,
            bos_token,
            eos_token,
        })
    }

    /// Encode text into token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with byte-level tokens
        let mut tokens: Vec<String> = text.bytes().map(|b| format!("<0x{b:02X}>")).collect();

        // Apply merges iteratively
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the best merge (lowest score = highest priority)
            let mut best_merge_idx = usize::MAX;
            let mut best_pair_idx = 0;
            for i in 0..tokens.len() - 1 {
                let pair = format!("{} {}", tokens[i], tokens[i + 1]);
                if let Some(&merge_idx) = self.token_to_id.get(&pair) {
                    if (merge_idx as usize) < best_merge_idx {
                        best_merge_idx = merge_idx as usize;
                        best_pair_idx = i;
                    }
                }
            }

            if best_merge_idx == usize::MAX {
                break;
            }

            let merged = format!("{}{}", tokens[best_pair_idx], tokens[best_pair_idx + 1]);
            tokens[best_pair_idx] = merged;
            tokens.remove(best_pair_idx + 1);
        }

        tokens
            .iter()
            .filter_map(|t| self.token_to_id.get(t).copied())
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
