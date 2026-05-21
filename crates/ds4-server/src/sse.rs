use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_core::Stream;
use futures_util::{StreamExt, stream};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::types::{GenerationEvent, OpenaiChatChunk, OpenaiChunkChoice, OpenaiDelta};

pub fn stream_from_receiver(
    request_id: String,
    model: String,
    created: u64,
    rx: mpsc::Receiver<GenerationEvent>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(rx).map(move |event| -> Vec<Result<Event, Infallible>> {
        match event {
            GenerationEvent::Token(text) => {
                let chunk = OpenaiChatChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.clone(),
                    choices: vec![OpenaiChunkChoice {
                        index: 0,
                        delta: OpenaiDelta {
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                vec![Ok(Event::default()
                    .data(serde_json::to_string(&chunk).unwrap_or_default()))]
            }
            GenerationEvent::Done { .. } => {
                let chunk = OpenaiChatChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.clone(),
                    choices: vec![OpenaiChunkChoice {
                        index: 0,
                        delta: OpenaiDelta { content: None },
                        finish_reason: Some("stop".into()),
                    }],
                };
                vec![
                    Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap_or_default())),
                    Ok(Event::default().data("[DONE]")),
                ]
            }
            GenerationEvent::Error(msg) => {
                vec![Ok(Event::default()
                    .data(serde_json::json!({"error": msg}).to_string()))]
            }
        }
    }).flat_map(|events| stream::iter(events));

    Sse::new(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk() -> OpenaiChatChunk {
        OpenaiChatChunk {
            id: "chatcmpl-test".into(),
            object: "chat.completion.chunk".into(),
            created: 1000,
            model: "ds4".into(),
            choices: vec![OpenaiChunkChoice {
                index: 0,
                delta: OpenaiDelta {
                    content: Some("Hello".into()),
                },
                finish_reason: None,
            }],
        }
    }

    #[test]
    fn token_chunk_format() {
        let chunk = sample_chunk();
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "chat.completion.chunk");
        assert_eq!(parsed["choices"][0]["delta"]["content"], "Hello");
        assert!(parsed["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn done_chunk_format() {
        let chunk = OpenaiChatChunk {
            id: "chatcmpl-test".into(),
            object: "chat.completion.chunk".into(),
            created: 1000,
            model: "ds4".into(),
            choices: vec![OpenaiChunkChoice {
                index: 0,
                delta: OpenaiDelta { content: None },
                finish_reason: Some("stop".into()),
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert!(parsed["choices"][0]["delta"]["content"].is_null());
    }

    #[test]
    fn delta_with_none_content() {
        let delta = OpenaiDelta { content: None };
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, r#"{"content":null}"#);
    }

    #[test]
    fn delta_with_content() {
        let delta = OpenaiDelta {
            content: Some("Hi".into()),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, r#"{"content":"Hi"}"#);
    }

    #[test]
    fn chunk_choice_with_stop() {
        let choice = OpenaiChunkChoice {
            index: 0,
            delta: OpenaiDelta { content: None },
            finish_reason: Some("stop".into()),
        };
        let json = serde_json::to_string(&choice).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["finish_reason"], "stop");
    }

    #[test]
    fn chunk_choice_with_content() {
        let choice = OpenaiChunkChoice {
            index: 0,
            delta: OpenaiDelta {
                content: Some("token".into()),
            },
            finish_reason: None,
        };
        let json = serde_json::to_string(&choice).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["delta"]["content"], "token");
        assert!(parsed["finish_reason"].is_null());
    }

    #[test]
    fn error_event_format() {
        let data = serde_json::json!({"error": "something failed"}).to_string();
        let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["error"], "something failed");
    }

    #[test]
    fn token_empty_string() {
        let delta = OpenaiDelta {
            content: Some("".into()),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, r#"{"content":""}"#);
    }

    #[test]
    fn token_unicode() {
        let delta = OpenaiDelta {
            content: Some("你好".into()),
        };
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["content"], "你好");
    }
}
