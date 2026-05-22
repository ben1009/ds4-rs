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
    // Template already starts with BOS-equivalent (<｜begin▁of▁sentence｜>),
    // so don't add another BOS via encode.
    let prompt_tokens = state.engine.tokenizer.encode(&prompt, false);

    let (tx, rx) = mpsc::channel(64);
    let request = InferenceRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        response_tx: tx,
    };

    if let Err(e) = state.inference.submit(request).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}})),
        )
            .into_response();
    }

    if req.stream {
        let req_id = request_id();
        let model_name = req.model.unwrap_or_else(|| "ds4".into());
        let created = timestamp();
        sse::stream_from_receiver(req_id, model_name, created, rx).into_response()
    } else {
        let mut content = String::new();
        let mut prompt_tok_count = 0u32;
        let mut completion_tok_count = 0u32;
        let mut finish_reason = "stop";
        let mut stream = rx;

        while let Some(event) = stream.recv().await {
            match event {
                GenerationEvent::Token(text) => content.push_str(&text),
                GenerationEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason: reason,
                } => {
                    prompt_tok_count = prompt_tokens;
                    completion_tok_count = completion_tokens;
                    finish_reason = reason;
                }
                GenerationEvent::Error(msg) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(
                            serde_json::json!({"error": {"message": msg, "type": "server_error"}}),
                        ),
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
                finish_reason: Some(finish_reason.into()),
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

// ── OpenAI /v1/completions (raw text) ────────────────────────────────────────

pub async fn openai_completions(
    State(state): State<AppState>,
    Json(req): Json<OpenaiCompletionRequest>,
) -> Response {
    // No chat template — encode the raw prompt with BOS prepended.
    let prompt_tokens = state.engine.tokenizer.encode(&req.prompt, true);

    let (tx, rx) = mpsc::channel(64);
    let request = InferenceRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        response_tx: tx,
    };

    if let Err(e) = state.inference.submit(request).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}})),
        )
            .into_response();
    }

    if req.stream {
        let req_id = format!("cmpl-{}", Uuid::new_v4().as_simple());
        let model_name = req.model.unwrap_or_else(|| "ds4".into());
        let created = timestamp();
        sse::stream_completions_from_receiver(req_id, model_name, created, rx).into_response()
    } else {
        let mut text = String::new();
        let mut prompt_tok_count = 0u32;
        let mut completion_tok_count = 0u32;
        let mut finish_reason = "stop";
        let mut stream = rx;

        while let Some(event) = stream.recv().await {
            match event {
                GenerationEvent::Token(t) => text.push_str(&t),
                GenerationEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason: reason,
                } => {
                    prompt_tok_count = prompt_tokens;
                    completion_tok_count = completion_tokens;
                    finish_reason = reason;
                }
                GenerationEvent::Error(msg) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(
                            serde_json::json!({"error": {"message": msg, "type": "server_error"}}),
                        ),
                    )
                        .into_response();
                }
            }
        }

        let response = OpenaiCompletionResponse {
            id: format!("cmpl-{}", Uuid::new_v4().as_simple()),
            object: "text_completion".into(),
            created: timestamp(),
            model: req.model.unwrap_or_else(|| "ds4".into()),
            choices: vec![OpenaiCompletionChoice {
                index: 0,
                text,
                finish_reason: Some(finish_reason.into()),
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

// ── OpenAI /v1/models ────────────────────────────────────────────────────────

pub async fn openai_models(State(state): State<AppState>) -> Response {
    let response = OpenaiModelList {
        object: "list".into(),
        data: vec![OpenaiModel {
            id: state.model_id.clone(),
            object: "model".into(),
            created: 0,
            owned_by: "ds4".into(),
        }],
    };
    Json(response).into_response()
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
    let prompt_tokens = state.engine.tokenizer.encode(&prompt, false);
    let input_token_count = prompt_tokens.len() as u32;

    let (tx, rx) = mpsc::channel(64);
    let request = InferenceRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        response_tx: tx,
    };

    if let Err(e) = state.inference.submit(request).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"type": "error", "error": {"type": "api_error", "message": e.to_string()}})),
        )
            .into_response();
    }

    let msg_id = format!("msg_{}", Uuid::new_v4().as_simple());
    let model = req.model.unwrap_or_else(|| "ds4".into());

    if req.stream {
        let stream = async_stream::stream! {
            // message_start
            let start = AnthropicStreamStart {
                event_type: "message_start".into(),
                message: AnthropicStreamMessage {
                    id: msg_id.clone(),
                    msg_type: "message".into(),
                    role: "assistant".into(),
                    model: model.clone(),
                    usage: AnthropicUsage {
                        input_tokens: input_token_count,
                        output_tokens: 0,
                    },
                },
            };
            yield Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default()
                    .event("message_start")
                    .data(serde_json::to_string(&start).unwrap_or_default()),
            );

            // content_block_start
            let block_start = AnthropicStreamContentBlockStart {
                event_type: "content_block_start".into(),
                index: 0,
                content_block: AnthropicContentBlock {
                    block_type: "text".into(),
                    text: String::new(),
                },
            };
            yield Ok(
                axum::response::sse::Event::default()
                    .event("content_block_start")
                    .data(serde_json::to_string(&block_start).unwrap_or_default()),
            );

            let mut completion_tokens = 0u32;
            let mut stream = rx;
            while let Some(event) = stream.recv().await {
                match event {
                    GenerationEvent::Token(text) => {
                        completion_tokens += 1;
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
                                .data(serde_json::to_string(&delta).unwrap_or_default()),
                        );
                    }
                    GenerationEvent::Done { finish_reason, .. } => {
                        // content_block_stop
                        let block_stop = AnthropicStreamContentBlockStop {
                            event_type: "content_block_stop".into(),
                            index: 0,
                        };
                        yield Ok(
                            axum::response::sse::Event::default()
                                .event("content_block_stop")
                                .data(serde_json::to_string(&block_stop).unwrap_or_default()),
                        );
                        // message_delta (carries stop_reason + usage)
                        let anthropic_stop = if finish_reason == "length" {
                            "max_tokens"
                        } else {
                            "end_turn"
                        };
                        let msg_delta = AnthropicStreamMessageDelta {
                            event_type: "message_delta".into(),
                            delta: AnthropicMessageDelta {
                                stop_reason: anthropic_stop.into(),
                            },
                            usage: AnthropicStreamUsage {
                                output_tokens: completion_tokens,
                            },
                        };
                        yield Ok(
                            axum::response::sse::Event::default()
                                .event("message_delta")
                                .data(serde_json::to_string(&msg_delta).unwrap_or_default()),
                        );
                        // message_stop
                        let stop = AnthropicStreamStop {
                            event_type: "message_stop".into(),
                        };
                        yield Ok(
                            axum::response::sse::Event::default()
                                .event("message_stop")
                                .data(serde_json::to_string(&stop).unwrap_or_default()),
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
        let mut finish_reason = "stop";
        let mut stream = rx;

        while let Some(event) = stream.recv().await {
            match event {
                GenerationEvent::Token(text) => content.push_str(&text),
                GenerationEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason: reason,
                } => {
                    prompt_tok_count = prompt_tokens;
                    completion_tok_count = completion_tokens;
                    finish_reason = reason;
                }
                GenerationEvent::Error(msg) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"type": "error", "error": {"type": "api_error", "message": msg}})),
                    )
                        .into_response();
                }
            }
        }

        let anthropic_stop = if finish_reason == "length" {
            "max_tokens"
        } else {
            "end_turn"
        };
        let response = AnthropicResponse {
            id: msg_id,
            msg_type: "message".into(),
            role: "assistant".into(),
            content: vec![AnthropicContent {
                content_type: "text".into(),
                text: content,
            }],
            model,
            stop_reason: Some(anthropic_stop.into()),
            usage: AnthropicUsage {
                input_tokens: prompt_tok_count,
                output_tokens: completion_tok_count,
            },
        };

        Json(response).into_response()
    }
}
