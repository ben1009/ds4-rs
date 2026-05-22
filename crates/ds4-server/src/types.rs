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

// ── OpenAI streaming chunk types ────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct OpenaiChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenaiChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct OpenaiChunkChoice {
    pub index: u32,
    pub delta: OpenaiDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenaiDelta {
    pub content: Option<String>,
}

// ── OpenAI Completions (raw text) ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct OpenaiCompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct OpenaiCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenaiCompletionChoice>,
    pub usage: OpenaiUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenaiCompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenaiCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenaiCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct OpenaiCompletionChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

// ── OpenAI Models ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct OpenaiModelList {
    pub object: String,
    pub data: Vec<OpenaiModel>,
}

#[derive(Debug, Serialize)]
pub struct OpenaiModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
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
    pub usage: AnthropicUsage,
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
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamContentBlockStart {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: u32,
    pub content_block: AnthropicContentBlock,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamContentBlockStop {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: u32,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamMessageDelta {
    #[serde(rename = "type")]
    pub event_type: String,
    pub delta: AnthropicMessageDelta,
    pub usage: AnthropicStreamUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageDelta {
    pub stop_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStreamUsage {
    pub output_tokens: u32,
}

// ── Shared ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GenerationEvent {
    Token(String),
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
        finish_reason: &'static str,
    },
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_request_defaults() {
        let json = r#"{"messages":[{"role":"user","content":"Hi"}]}"#;
        let req: OpenaiChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 512);
        assert!(!req.stream);
        assert!(req.model.is_none());
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn openai_request_all_fields() {
        let json = r#"{"model":"ds4","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"}],"max_tokens":100,"stream":true}"#;
        let req: OpenaiChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("ds4"));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, 100);
        assert!(req.stream);
    }

    #[test]
    fn openai_response_serializes() {
        let resp = OpenaiChatResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion".into(),
            created: 1000,
            model: "ds4".into(),
            choices: vec![OpenaiChoice {
                index: 0,
                message: OpenaiMessage {
                    role: "assistant".into(),
                    content: "Hello!".into(),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: OpenaiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"id\":\"chatcmpl-123\""));
        assert!(json.contains("\"finish_reason\":\"stop\""));
        assert!(json.contains("\"total_tokens\":15"));
    }

    #[test]
    fn anthropic_request_defaults() {
        let json = r#"{"messages":[{"role":"user","content":"Hi"}]}"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 512);
        assert!(!req.stream);
        assert!(req.system.is_none());
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn anthropic_request_with_system() {
        let json = r#"{"model":"ds4","system":"Be helpful.","messages":[{"role":"user","content":"Hi"}],"max_tokens":200,"stream":true}"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system.as_deref(), Some("Be helpful."));
        assert_eq!(req.max_tokens, 200);
        assert!(req.stream);
    }

    #[test]
    fn anthropic_response_serializes_with_type_field() {
        let resp = AnthropicResponse {
            id: "msg_abc".into(),
            msg_type: "message".into(),
            role: "assistant".into(),
            content: vec![AnthropicContent {
                content_type: "text".into(),
                text: "Hi there".into(),
            }],
            model: "ds4".into(),
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 3,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"stop_reason\":\"end_turn\""));
        assert!(json.contains("\"input_tokens\":10"));
    }

    #[test]
    fn anthropic_stream_types_serialize() {
        let start = AnthropicStreamStart {
            event_type: "message_start".into(),
            message: AnthropicStreamMessage {
                id: "msg_1".into(),
                msg_type: "message".into(),
                role: "assistant".into(),
                model: "ds4".into(),
                usage: AnthropicUsage {
                    input_tokens: 10,
                    output_tokens: 0,
                },
            },
        };
        let json = serde_json::to_string(&start).unwrap();
        assert!(json.contains("\"type\":\"message_start\""));
        assert!(json.contains("\"input_tokens\":10"));

        let delta = AnthropicStreamDelta {
            event_type: "content_block_delta".into(),
            delta: AnthropicDelta {
                delta_type: "text_delta".into(),
                text: "Hello".into(),
            },
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains("\"type\":\"content_block_delta\""));
        assert!(json.contains("\"type\":\"text_delta\""));

        let stop = AnthropicStreamStop {
            event_type: "message_stop".into(),
        };
        let json = serde_json::to_string(&stop).unwrap();
        assert!(json.contains("\"type\":\"message_stop\""));

        let msg_delta = AnthropicStreamMessageDelta {
            event_type: "message_delta".into(),
            delta: AnthropicMessageDelta {
                stop_reason: "end_turn".into(),
            },
            usage: AnthropicStreamUsage { output_tokens: 5 },
        };
        let json = serde_json::to_string(&msg_delta).unwrap();
        assert!(json.contains("\"type\":\"message_delta\""));
        assert!(json.contains("\"stop_reason\":\"end_turn\""));
        assert!(json.contains("\"output_tokens\":5"));
    }

    #[test]
    fn generation_event_debug() {
        let e = GenerationEvent::Token("hi".into());
        assert!(format!("{e:?}").contains("Token"));
        let e = GenerationEvent::Done {
            prompt_tokens: 5,
            completion_tokens: 3,
            finish_reason: "stop",
        };
        assert!(format!("{e:?}").contains("Done"));
        let e = GenerationEvent::Error("fail".into());
        assert!(format!("{e:?}").contains("Error"));
    }

    #[test]
    fn openai_completion_request_defaults() {
        let json = r#"{"prompt":"Once upon a time"}"#;
        let req: OpenaiCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "Once upon a time");
        assert_eq!(req.max_tokens, 512);
        assert!(!req.stream);
        assert!(req.model.is_none());
    }

    #[test]
    fn openai_completion_response_serializes() {
        let resp = OpenaiCompletionResponse {
            id: "cmpl-123".into(),
            object: "text_completion".into(),
            created: 1000,
            model: "ds4".into(),
            choices: vec![OpenaiCompletionChoice {
                index: 0,
                text: "...there was a model.".into(),
                finish_reason: Some("stop".into()),
            }],
            usage: OpenaiUsage {
                prompt_tokens: 5,
                completion_tokens: 6,
                total_tokens: 11,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "text_completion");
        assert_eq!(parsed["choices"][0]["text"], "...there was a model.");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["usage"]["total_tokens"], 11);
    }

    #[test]
    fn openai_model_list_serializes() {
        let list = OpenaiModelList {
            object: "list".into(),
            data: vec![OpenaiModel {
                id: "ds4flash".into(),
                object: "model".into(),
                created: 0,
                owned_by: "ds4".into(),
            }],
        };
        let json = serde_json::to_string(&list).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "list");
        assert_eq!(parsed["data"][0]["id"], "ds4flash");
        assert_eq!(parsed["data"][0]["owned_by"], "ds4");
    }
}
