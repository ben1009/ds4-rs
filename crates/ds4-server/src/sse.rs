use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_core::Stream;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::types::GenerationEvent;

pub fn stream_from_receiver(
    rx: mpsc::Receiver<GenerationEvent>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(rx).map(|event| {
        let data = match event {
            GenerationEvent::Token(text) => serde_json::json!({"token": text}).to_string(),
            GenerationEvent::Done {
                prompt_tokens,
                completion_tokens,
            } => serde_json::json!({
                "done": true,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            })
            .to_string(),
            GenerationEvent::Error(msg) => serde_json::json!({"error": msg}).to_string(),
        };
        Ok(Event::default().data(data))
    });

    Sse::new(stream)
}
