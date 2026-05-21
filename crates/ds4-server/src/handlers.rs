use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{AppState, chat_template, generation::InferenceRequest, sse, types::*};

fn request_id() -> String {
    format!("chatcmpl-{}", Uuid::new_v4().as_simple())
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── OpenAI /v1/chat/completions ──────────────────────────────────────────────

pub async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<OpenaiChatRequest>,
) -> Response {
    let messages: Vec<(&str, &str)> = req
        .messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();

    let prompt = chat_template::render(&messages);
    let prompt_tokens = state.engine.tokenizer.encode(&prompt, true);

    let (tx, rx) = mpsc::channel(64);
    let request = InferenceRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        response_tx: tx,
    };

    if let Err(e) = state.inference.submit(request).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response();
    }

    if req.stream {
        sse::stream_from_receiver(rx).into_response()
    } else {
        let mut content = String::new();
        let mut prompt_tok_count = 0u32;
        let mut completion_tok_count = 0u32;
        let mut stream = rx;

        while let Some(event) = stream.recv().await {
            match event {
                GenerationEvent::Token(text) => content.push_str(&text),
                GenerationEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                } => {
                    prompt_tok_count = prompt_tokens;
                    completion_tok_count = completion_tokens;
                }
                GenerationEvent::Error(msg) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": msg})),
                    )
                        .into_response();
                }
            }
        }

        let response = OpenaiChatResponse {
            id: request_id(),
            object: "chat.completion".into(),
            created: timestamp(),
            model: req.model.unwrap_or_else(|| "ds4".into()),
            choices: vec![OpenaiChoice {
                index: 0,
                message: OpenaiMessage {
                    role: "assistant".into(),
                    content,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: OpenaiUsage {
                prompt_tokens: prompt_tok_count,
                completion_tokens: completion_tok_count,
                total_tokens: prompt_tok_count + completion_tok_count,
            },
        };

        Json(response).into_response()
    }
}

// ── Anthropic /v1/messages ───────────────────────────────────────────────────

pub async fn anthropic_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> Response {
    let mut messages: Vec<(&str, &str)> = Vec::new();

    if let Some(ref system) = req.system {
        messages.push(("system", system));
    }

    for m in &req.messages {
        messages.push((m.role.as_str(), m.content.as_str()));
    }

    let prompt = chat_template::render(&messages);
    let prompt_tokens = state.engine.tokenizer.encode(&prompt, true);

    let (tx, rx) = mpsc::channel(64);
    let request = InferenceRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        response_tx: tx,
    };

    if let Err(e) = state.inference.submit(request).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response();
    }

    let msg_id = format!("msg_{}", Uuid::new_v4().as_simple());
    let model = req.model.unwrap_or_else(|| "ds4".into());

    if req.stream {
        let stream = async_stream::stream! {
            let start = AnthropicStreamStart {
                event_type: "message_start".into(),
                message: AnthropicStreamMessage {
                    id: msg_id.clone(),
                    msg_type: "message".into(),
                    role: "assistant".into(),
                    model: model.clone(),
                },
            };
            yield Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default()
                    .event("message_start")
                    .data(serde_json::to_string(&start).unwrap()),
            );

            let mut stream = rx;
            while let Some(event) = stream.recv().await {
                match event {
                    GenerationEvent::Token(text) => {
                        let delta = AnthropicStreamDelta {
                            event_type: "content_block_delta".into(),
                            delta: AnthropicDelta {
                                delta_type: "text_delta".into(),
                                text,
                            },
                        };
                        yield Ok(
                            axum::response::sse::Event::default()
                                .event("content_block_delta")
                                .data(serde_json::to_string(&delta).unwrap()),
                        );
                    }
                    GenerationEvent::Done { .. } => {
                        let stop = AnthropicStreamStop {
                            event_type: "message_stop".into(),
                            stop_reason: "end_turn".into(),
                        };
                        yield Ok(
                            axum::response::sse::Event::default()
                                .event("message_stop")
                                .data(serde_json::to_string(&stop).unwrap()),
                        );
                    }
                    GenerationEvent::Error(msg) => {
                        yield Ok(
                            axum::response::sse::Event::default()
                                .event("error")
                                .data(serde_json::json!({"error": msg}).to_string()),
                        );
                    }
                }
            }
        };

        axum::response::sse::Sse::new(stream).into_response()
    } else {
        let mut content = String::new();
        let mut prompt_tok_count = 0u32;
        let mut completion_tok_count = 0u32;
        let mut stream = rx;

        while let Some(event) = stream.recv().await {
            match event {
                GenerationEvent::Token(text) => content.push_str(&text),
                GenerationEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                } => {
                    prompt_tok_count = prompt_tokens;
                    completion_tok_count = completion_tokens;
                }
                GenerationEvent::Error(msg) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": msg})),
                    )
                        .into_response();
                }
            }
        }

        let response = AnthropicResponse {
            id: msg_id,
            msg_type: "message".into(),
            role: "assistant".into(),
            content: vec![AnthropicContent {
                content_type: "text".into(),
                text: content,
            }],
            model,
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: prompt_tok_count,
                output_tokens: completion_tok_count,
            },
        };

        Json(response).into_response()
    }
}
