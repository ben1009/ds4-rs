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

#[cfg(test)]
mod tests {
    use super::*;

    fn format_event(event: &GenerationEvent) -> String {
        match event {
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
        }
    }

    #[test]
    fn token_event_format() {
        let data = format_event(&GenerationEvent::Token("Hello".into()));
        assert_eq!(data, r#"{"token":"Hello"}"#);
    }

    #[test]
    fn done_event_format() {
        let data = format_event(&GenerationEvent::Done {
            prompt_tokens: 10,
            completion_tokens: 5,
        });
        let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["done"], true);
        assert_eq!(parsed["prompt_tokens"], 10);
        assert_eq!(parsed["completion_tokens"], 5);
    }

    #[test]
    fn error_event_format() {
        let data = format_event(&GenerationEvent::Error("something failed".into()));
        let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["error"], "something failed");
    }

    #[test]
    fn token_empty_string() {
        let data = format_event(&GenerationEvent::Token("".into()));
        let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["token"], "");
    }

    #[test]
    fn token_unicode() {
        let data = format_event(&GenerationEvent::Token("你好".into()));
        let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["token"], "你好");
    }

    #[test]
    fn done_zero_counts() {
        let data = format_event(&GenerationEvent::Done {
            prompt_tokens: 0,
            completion_tokens: 0,
        });
        let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["done"], true);
        assert_eq!(parsed["prompt_tokens"], 0);
        assert_eq!(parsed["completion_tokens"], 0);
    }

    #[test]
    fn error_empty_message() {
        let data = format_event(&GenerationEvent::Error("".into()));
        let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["error"], "");
    }
}
