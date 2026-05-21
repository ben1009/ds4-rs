/// Render messages in DeepSeek V4 chat template format.
///
/// Format:
/// ```text
/// <｜begin▁of▁sentence｜>system prompt<｜User｜>user msg<｜Assistant｜>
/// ```
pub fn render(messages: &[(&str, &str)]) -> String {
    let mut out = String::from("<｜begin▁of▁sentence｜>");

    for (role, content) in messages {
        match *role {
            "system" => {
                out.push_str(content);
            }
            "user" => {
                out.push_str("<｜User｜>");
                out.push_str(content);
            }
            "assistant" => {
                out.push_str("<｜Assistant｜>");
                out.push_str(content);
            }
            _ => {}
        }
    }

    // End with Assistant turn marker so the model generates a response
    if messages.last().is_some_and(|(r, _)| *r != "assistant") {
        out.push_str("<｜Assistant｜>");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_user_message() {
        let result = render(&[("user", "Hello")]);
        assert_eq!(
            result,
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>"
        );
    }

    #[test]
    fn system_and_user() {
        let result = render(&[("system", "You are helpful."), ("user", "Hi")]);
        assert_eq!(
            result,
            "<｜begin▁of▁sentence｜>You are helpful.<｜User｜>Hi<｜Assistant｜>"
        );
    }

    #[test]
    fn multi_turn() {
        let result = render(&[("user", "Hi"), ("assistant", "Hello!"), ("user", "Bye")]);
        assert_eq!(
            result,
            "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜>Hello!<｜User｜>Bye<｜Assistant｜>"
        );
    }

    #[test]
    fn ends_with_assistant_no_trailing_marker() {
        let result = render(&[("user", "Hi"), ("assistant", "Hello!")]);
        assert_eq!(
            result,
            "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜>Hello!"
        );
    }

    #[test]
    fn empty_messages() {
        // No messages — no trailing Assistant marker since there's nothing to respond to
        let result = render(&[]);
        assert_eq!(result, "<｜begin▁of▁sentence｜>");
    }

    #[test]
    fn unknown_role_ignored() {
        let result = render(&[("tool", "output"), ("user", "Hi")]);
        assert_eq!(result, "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜>");
    }

    #[test]
    fn system_only() {
        let result = render(&[("system", "Be helpful.")]);
        assert_eq!(result, "<｜begin▁of▁sentence｜>Be helpful.<｜Assistant｜>");
    }

    #[test]
    fn multiple_system_messages() {
        let result = render(&[("system", "Rule 1."), ("system", "Rule 2."), ("user", "Hi")]);
        assert_eq!(
            result,
            "<｜begin▁of▁sentence｜>Rule 1.Rule 2.<｜User｜>Hi<｜Assistant｜>"
        );
    }

    #[test]
    fn unicode_content() {
        let result = render(&[("user", "你好世界")]);
        assert_eq!(
            result,
            "<｜begin▁of▁sentence｜><｜User｜>你好世界<｜Assistant｜>"
        );
    }

    #[test]
    fn empty_content() {
        let result = render(&[("user", "")]);
        assert_eq!(result, "<｜begin▁of▁sentence｜><｜User｜><｜Assistant｜>");
    }

    #[test]
    fn three_turn_conversation() {
        let result = render(&[
            ("user", "A"),
            ("assistant", "B"),
            ("user", "C"),
            ("assistant", "D"),
            ("user", "E"),
        ]);
        assert_eq!(
            result,
            "<｜begin▁of▁sentence｜><｜User｜>A<｜Assistant｜>B<｜User｜>C<｜Assistant｜>D<｜User｜>E<｜Assistant｜>"
        );
    }
}
