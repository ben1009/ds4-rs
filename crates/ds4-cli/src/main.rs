use std::{
    io::{BufRead, Write},
    path::PathBuf,
};

use anyhow::Result;
use clap::Parser;
use ds4_core::{engine::Engine, session::Session};

/// Split `bytes` into (valid UTF-8 prefix length, invalid-sequence length).
/// If invalid_len > 0, those bytes should be replaced with U+FFFD and drained
/// together with the valid prefix. If invalid_len == 0, any bytes beyond the
/// valid prefix are a trailing partial multi-byte sequence and should be held.
fn utf8_split(bytes: &[u8]) -> (usize, usize) {
    match std::str::from_utf8(bytes) {
        Ok(_) => (bytes.len(), 0),
        Err(e) => match e.error_len() {
            Some(n) => (e.valid_up_to(), n),
            None => (e.valid_up_to(), 0),
        },
    }
}

/// Drain valid UTF-8 chunks from `pending`, replacing any invalid sequences
/// with U+FFFD. Trailing partial multi-byte sequences are left in `pending`
/// unless `flush_partial` is true (end-of-stream).
fn write_utf8<W: Write>(
    w: &mut W,
    pending: &mut Vec<u8>,
    flush_partial: bool,
) -> std::io::Result<()> {
    loop {
        let (valid, invalid) = utf8_split(pending);
        if valid > 0 {
            w.write_all(&pending[..valid])?;
        }
        if invalid > 0 {
            w.write_all("\u{FFFD}".as_bytes())?;
            pending.drain(..valid + invalid);
        } else {
            pending.drain(..valid);
            break;
        }
    }
    if flush_partial && !pending.is_empty() {
        w.write_all("\u{FFFD}".as_bytes())?;
        pending.clear();
    }
    Ok(())
}

/// ds4-rs: DeepSeek V4 Flash inference engine for Linux
#[derive(Parser)]
#[command(name = "ds4", version)]
struct Args {
    /// Path to the GGUF model file
    #[arg(long, default_value = "./ds4flash.gguf")]
    model: PathBuf,

    /// Prompt for one-shot generation
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value = "256")]
    max_tokens: u32,

    /// Context size
    #[arg(long, default_value = "2048")]
    ctx: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    let engine = Engine::open(&args.model)?;

    match args.prompt {
        Some(prompt) => one_shot(&engine, &prompt, args.max_tokens, args.ctx)?,
        None => repl(&engine, args.max_tokens, args.ctx)?,
    }

    Ok(())
}

fn one_shot(
    engine: &std::sync::Arc<Engine>,
    prompt: &str,
    max_tokens: u32,
    ctx: u32,
) -> Result<()> {
    let mut session = Session::new(engine.clone(), ctx)?;
    let tokens = engine.tokenizer.encode(prompt, true);
    tracing::info!("Prompt: {} tokens", tokens.len());

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    generate_turn(engine, &mut session, &tokens, max_tokens, &mut handle)
}

/// One parsed line of REPL input.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// Generate a response for the user-supplied prompt text.
    Prompt(String),
    /// Discard the entire session (`/reset` or `/clear`).
    Reset,
    /// Rewind to position `n` (`/rewind <n>`).
    Rewind(u32),
    /// Show context size and current position (`/ctx`).
    ShowCtx,
    /// Print the help text (`/help`).
    Help,
    /// Leave the REPL (`/exit` or `/quit`).
    Exit,
    /// Empty input — skip silently.
    Empty,
    /// Unknown slash command (`/foo bar` → message echoed back to user).
    Unknown(String),
}

/// Parse one line of REPL input. Slash commands are recognised case-
/// insensitively; everything else falls through as a prompt.
///
/// Whitespace handling:
/// * The trailing newline (`\n` / `\r`) is stripped from any line.
/// * For *commands* we additionally trim leading whitespace so `  /help` still works.
/// * For *prompts* we keep all interior and surrounding whitespace (apart from the line terminator)
///   so indented code, alignment, or trailing spaces survive — matching what `-p` does in one-shot
///   mode, and what the user actually typed.
///
/// No-argument commands reject any trailing junk and route to
/// [`Command::Unknown`] so a typo like `/reset later` cannot silently
/// destroy session state.
fn parse_command(line: &str) -> Command {
    let stripped = line.trim_end_matches(['\n', '\r']);
    if stripped.trim().is_empty() {
        return Command::Empty;
    }
    let cmd_view = stripped.trim_start();
    if let Some(rest) = cmd_view.strip_prefix('/') {
        let mut parts = rest.splitn(2, char::is_whitespace);
        let head = parts.next().unwrap_or("").to_ascii_lowercase();
        let arg = parts.next().unwrap_or("").trim();
        let no_args = |c: Command| -> Command {
            if arg.is_empty() {
                c
            } else {
                Command::Unknown(format!("/{head} does not take arguments"))
            }
        };
        return match head.as_str() {
            "reset" | "clear" => no_args(Command::Reset),
            "rewind" => match arg.parse::<u32>() {
                Ok(n) => Command::Rewind(n),
                Err(_) => {
                    Command::Unknown(format!("/rewind needs a non-negative integer, got {arg:?}"))
                }
            },
            "ctx" => no_args(Command::ShowCtx),
            "help" | "?" => no_args(Command::Help),
            "exit" | "quit" => no_args(Command::Exit),
            other => Command::Unknown(format!("unknown command /{other}")),
        };
    }
    Command::Prompt(stripped.to_string())
}

const REPL_HELP: &str = "\
Commands:
  /reset            discard the entire session (alias: /clear)
  /rewind <n>       rewind to position n (drops tokens past it)
  /ctx              show ctx_size and current position
  /help             this help (alias: /?)
  /exit             leave the REPL (alias: /quit)
Anything else is treated as a prompt.";

fn repl(engine: &std::sync::Arc<Engine>, max_tokens: u32, ctx: u32) -> Result<()> {
    let mut session = Session::new(engine.clone(), ctx)?;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut input = stdin.lock();
    let mut line = String::new();

    writeln!(out, "ds4 REPL — /help for commands, /exit to leave.")?;
    out.flush()?;

    loop {
        write!(out, "> ")?;
        out.flush()?;
        line.clear();
        let n = input.read_line(&mut line)?;
        if n == 0 {
            // EOF (Ctrl-D).
            writeln!(out)?;
            return Ok(());
        }

        match parse_command(&line) {
            Command::Empty => continue,
            Command::Exit => return Ok(()),
            Command::Help => {
                writeln!(out, "{REPL_HELP}")?;
            }
            Command::ShowCtx => {
                writeln!(
                    out,
                    "ctx_size={}  pos={}  tokens={}",
                    session.ctx_size(),
                    session.pos(),
                    session.tokens().len()
                )?;
            }
            Command::Reset => {
                session.invalidate();
                writeln!(out, "session reset.")?;
            }
            Command::Rewind(n) => {
                let before = session.pos();
                if let Err(err) = session.rewind(n) {
                    writeln!(out, "error rewinding: {err}")?;
                } else {
                    writeln!(out, "rewound from pos {} to pos {}", before, session.pos())?;
                }
            }
            Command::Unknown(msg) => {
                writeln!(out, "error: {msg} — try /help")?;
            }
            Command::Prompt(text) => {
                // Only prepend BOS for the very first prefill of the session;
                // multi-turn input is appended without another BOS.
                let add_bos = session.tokens().is_empty();
                let tokens = engine.tokenizer.encode(&text, add_bos);
                if let Err(err) = generate_turn(engine, &mut session, &tokens, max_tokens, &mut out)
                {
                    writeln!(out, "error: {err}")?;
                }
            }
        }
        out.flush()?;
    }
}

/// Run one turn against an already-encoded prompt: prefill, then generate up
/// to `max_tokens` tokens, streaming UTF-8 output. Empty input is a no-op.
///
/// Errors leave session state intact: `Session::prefill` and `eval_token`
/// already roll their own state back on forward failures, so the REPL can
/// continue with the previous turn's tokens / KV intact. No extra snapshot
/// is needed at this layer.
fn generate_turn<W: Write>(
    engine: &Engine,
    session: &mut Session,
    tokens: &[u32],
    max_tokens: u32,
    out: &mut W,
) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }

    let logits = session.prefill(tokens)?;
    let mut token =
        Session::argmax(logits).ok_or_else(|| anyhow::anyhow!("prefill returned empty logits"))?;
    let eos = engine.tokenizer.eos_token();
    let mut pending: Vec<u8> = Vec::new();

    for _ in 0..max_tokens {
        if token == eos {
            break;
        }
        engine.tokenizer.append_token_bytes(token, &mut pending);
        write_utf8(out, &mut pending, false)?;
        out.flush()?;
        let logits = session.eval_token(token)?;
        token = Session::argmax(logits)
            .ok_or_else(|| anyhow::anyhow!("eval_token returned empty logits"))?;
    }
    write_utf8(out, &mut pending, true)?;
    writeln!(out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, utf8_split, write_utf8};

    #[test]
    fn holds_partial_multibyte() {
        assert_eq!(utf8_split(&[0xC3]), (0, 0));
        assert_eq!(utf8_split(&[0xC3, 0xA9]), (2, 0));
    }

    #[test]
    fn replaces_invalid_with_fffd() {
        let mut pending = vec![b'A', 0xFF, b'B'];
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, "A\u{FFFD}B".as_bytes());
        assert!(pending.is_empty());
    }

    #[test]
    fn end_of_stream_replaces_dangling_partial() {
        let mut pending = vec![b'A', 0xC3];
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, b"A");
        assert_eq!(pending, vec![0xC3]);
        write_utf8(&mut out, &mut pending, true).unwrap();
        assert_eq!(out, "A\u{FFFD}".as_bytes());
        assert!(pending.is_empty());
    }

    #[test]
    fn utf8_split_all_valid() {
        assert_eq!(utf8_split(b"hello"), (5, 0));
        assert_eq!(utf8_split("héllo".as_bytes()), ("héllo".len(), 0));
    }

    #[test]
    fn utf8_split_invalid_in_middle() {
        let bytes = [b'a', 0xFF, b'b'];
        let (valid, invalid) = utf8_split(&bytes);
        assert_eq!(valid, 1);
        assert_eq!(invalid, 1);
    }

    #[test]
    fn utf8_split_empty() {
        assert_eq!(utf8_split(&[]), (0, 0));
    }

    #[test]
    fn write_utf8_pure_ascii_drains_all() {
        let mut pending = b"hello world".to_vec();
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, b"hello world");
        assert!(pending.is_empty());
    }

    #[test]
    fn write_utf8_multiple_invalid_sequences() {
        let mut pending = vec![b'a', 0xFF, b'b', 0xFE, b'c'];
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, "a\u{FFFD}b\u{FFFD}c".as_bytes());
        assert!(pending.is_empty());
    }

    #[test]
    fn write_utf8_holds_trailing_partial_without_flush() {
        let mut pending = vec![b'a', 0xE2, 0x82];
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, b"a");
        assert_eq!(pending, vec![0xE2, 0x82]);
    }

    #[test]
    fn write_utf8_completes_partial_on_more_bytes() {
        let mut pending = vec![0xE2, 0x82];
        let mut out = Vec::new();
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert!(out.is_empty());
        pending.push(0xAC);
        write_utf8(&mut out, &mut pending, false).unwrap();
        assert_eq!(out, "\u{20AC}".as_bytes());
        assert!(pending.is_empty());
    }

    #[test]
    fn args_defaults() {
        let args = Args::parse_from(["ds4"]);
        assert_eq!(args.model.to_str().unwrap(), "./ds4flash.gguf");
        assert!(args.prompt.is_none());
        assert_eq!(args.max_tokens, 256);
        assert_eq!(args.ctx, 2048);
    }

    #[test]
    fn args_with_prompt_short_flag() {
        let args = Args::parse_from(["ds4", "-p", "hello"]);
        assert_eq!(args.prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn args_with_prompt_long_flag() {
        let args = Args::parse_from(["ds4", "--prompt", "world"]);
        assert_eq!(args.prompt.as_deref(), Some("world"));
    }

    #[test]
    fn args_with_max_tokens() {
        let args = Args::parse_from(["ds4", "-n", "10"]);
        assert_eq!(args.max_tokens, 10);
        let args = Args::parse_from(["ds4", "--max-tokens", "42"]);
        assert_eq!(args.max_tokens, 42);
    }

    #[test]
    fn args_with_ctx() {
        let args = Args::parse_from(["ds4", "--ctx", "1024"]);
        assert_eq!(args.ctx, 1024);
    }

    #[test]
    fn args_with_custom_model_path() {
        let args = Args::parse_from(["ds4", "--model", "/tmp/custom.gguf"]);
        assert_eq!(args.model.to_str().unwrap(), "/tmp/custom.gguf");
    }

    #[test]
    fn args_combined_flags() {
        let args = Args::parse_from([
            "ds4", "--model", "/m.gguf", "-p", "hi", "-n", "5", "--ctx", "64",
        ]);
        assert_eq!(args.model.to_str().unwrap(), "/m.gguf");
        assert_eq!(args.prompt.as_deref(), Some("hi"));
        assert_eq!(args.max_tokens, 5);
        assert_eq!(args.ctx, 64);
    }

    #[test]
    fn args_rejects_unknown_flag() {
        assert!(Args::try_parse_from(["ds4", "--no-such-flag"]).is_err());
    }

    #[test]
    fn args_rejects_invalid_max_tokens() {
        assert!(Args::try_parse_from(["ds4", "-n", "not-a-number"]).is_err());
    }

    // -----------------------------------------------------------------------
    // REPL command parser
    //
    // The slash-command surface stays small, but it is the only place in
    // the CLI that interprets user input — easy to break without tests.
    // -----------------------------------------------------------------------

    use super::{Command, parse_command};

    #[test]
    fn parse_empty_line_is_empty() {
        assert_eq!(parse_command(""), Command::Empty);
        assert_eq!(parse_command("   \t\n"), Command::Empty);
    }

    #[test]
    fn parse_plain_text_is_prompt() {
        assert_eq!(
            parse_command("hello world"),
            Command::Prompt("hello world".to_string())
        );
    }

    #[test]
    fn parse_strips_only_line_terminator_from_prompt() {
        // Preserve interior + surrounding whitespace so prompts like
        // indented code or alignment-sensitive text reach the model
        // unchanged. Only the trailing CR/LF is dropped.
        assert_eq!(
            parse_command("   hi\n"),
            Command::Prompt("   hi".to_string())
        );
        assert_eq!(
            parse_command("foo bar  \r\n"),
            Command::Prompt("foo bar  ".to_string())
        );
        assert_eq!(
            parse_command("\t\tindented"),
            Command::Prompt("\t\tindented".to_string())
        );
    }

    #[test]
    fn parse_reset_aliases() {
        assert_eq!(parse_command("/reset"), Command::Reset);
        assert_eq!(parse_command("/clear"), Command::Reset);
        assert_eq!(parse_command("/RESET"), Command::Reset);
    }

    #[test]
    fn parse_no_arg_command_with_extra_text_is_unknown() {
        // Codex P2 fix: a typo like `/reset later` must NOT silently
        // wipe session state. Anything other than the bare command
        // routes to Unknown so the REPL can complain instead of acting.
        for input in [
            "/reset later",
            "/clear now",
            "/ctx 5",
            "/help me",
            "/? thing",
            "/exit now",
            "/quit please",
        ] {
            match parse_command(input) {
                Command::Unknown(msg) => assert!(
                    msg.contains("does not take arguments"),
                    "input {input:?} got msg {msg:?}",
                ),
                other => panic!("input {input:?}: expected Unknown, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_leading_whitespace_before_command_still_parses() {
        // Trim leading whitespace for *commands* only (not prompts).
        assert_eq!(parse_command("   /help"), Command::Help);
        assert_eq!(parse_command("\t/reset\n"), Command::Reset);
    }

    #[test]
    fn parse_rewind_with_arg() {
        assert_eq!(parse_command("/rewind 5"), Command::Rewind(5));
        assert_eq!(parse_command("/rewind  42  "), Command::Rewind(42));
    }

    #[test]
    fn parse_rewind_without_arg_is_unknown() {
        // splitn(2) gives "" as the arg → parse fails → Unknown.
        match parse_command("/rewind") {
            Command::Unknown(_) => (),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn parse_rewind_negative_is_unknown() {
        match parse_command("/rewind -1") {
            Command::Unknown(_) => (),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn parse_ctx_help_exit() {
        assert_eq!(parse_command("/ctx"), Command::ShowCtx);
        assert_eq!(parse_command("/help"), Command::Help);
        assert_eq!(parse_command("/?"), Command::Help);
        assert_eq!(parse_command("/exit"), Command::Exit);
        assert_eq!(parse_command("/quit"), Command::Exit);
    }

    #[test]
    fn parse_unknown_slash_command() {
        match parse_command("/nope") {
            Command::Unknown(msg) => assert!(msg.contains("nope")),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn parse_slash_inside_word_is_prompt() {
        // Only a leading '/' triggers command parsing.
        assert_eq!(
            parse_command("path/to/file"),
            Command::Prompt("path/to/file".to_string())
        );
    }
}
