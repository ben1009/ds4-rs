use serde::{Deserialize, Serialize};

// ── OpenAI Chat Completions ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct OpenaiChatRequest {
    pub model: Option<String>,
    pub messages: Vec<OpenaiMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> u32 {
    512
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenaiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OpenaiChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenaiChoice>,
    pub usage: OpenaiUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenaiChoice {
    pub index: u32,
    pub message: OpenaiMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenaiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Anthropic Messages ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub system: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub content: Vec<AnthropicContent>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamStart {
    #[serde(rename = "type")]
    pub event_type: String,
    pub message: AnthropicStreamMessage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    pub event_type: String,
    pub delta: AnthropicDelta,
}

#[derive(Debug, Serialize)]
pub struct AnthropicDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamStop {
    #[serde(rename = "type")]
    pub event_type: String,
    pub stop_reason: String,
}

// ── Shared ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GenerationEvent {
    Token(String),
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Error(String),
}
