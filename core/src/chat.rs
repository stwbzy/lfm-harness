//! Chat formatting for LFM3.
//!
//! llama.cpp's built-in template engine cannot run the Jinja template shipped in
//! the LFM3 GGUF, so the subset the harness needs is reproduced here:
//! `<|startoftext|>` + ChatML turns, tool declarations appended to the system
//! prompt inside `<tools>…</tools>`, assistant turns with a `<think>…</think>`
//! block, and tool results wrapped in `<tool_response>`.
//!
//! Thinking rules (as in the template): assistant turns *before* the last user
//! message are rendered as `<think></think>` + answer (reasoning dropped);
//! turns *after* it (the current tool loop) are rendered verbatim when they
//! already contain `</think>`, so they match the generated tokens exactly.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// What the generation prompt ends with; generated text continues from it.
pub fn assistant_prefix(enable_thinking: bool) -> &'static str {
    if enable_thinking { "<think>" } else { "<think></think>" }
}

/// Renders the conversation and opens a new assistant turn.
///
/// `bos` is the model's BOS token text (`<|startoftext|>` for LFM3).
/// `tools` is the body of the `<tools>` block (one JSON object per line).
pub fn render_prompt(bos: &str, messages: &[Message], tools: Option<&str>, enable_thinking: bool) -> String {
    let mut out = String::from(bos);
    let (mut system, rest) = match messages.split_first() {
        Some((first, rest)) if first.role == Role::System => (first.content.clone(), rest),
        _ => (String::new(), messages),
    };
    if let Some(t) = tools.filter(|t| !t.is_empty()) {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str("<tools>\n");
        system.push_str(t);
        system.push_str("\n</tools>");
    }
    if !system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(&system);
        out.push_str("<|im_end|>\n");
    }

    let last_user = rest.iter().rposition(|m| m.role == Role::User);
    for (i, m) in rest.iter().enumerate() {
        out.push_str("<|im_start|>");
        out.push_str(m.role.as_str());
        out.push('\n');
        match m.role {
            Role::Assistant => {
                let current_loop = last_user.is_none_or(|u| i > u);
                if current_loop && m.content.contains("</think>") {
                    out.push_str(&m.content);
                } else if current_loop {
                    out.push_str("<think></think>");
                    out.push_str(&m.content);
                } else {
                    out.push_str("<think></think>");
                    out.push_str(strip_thinking(&m.content));
                }
            }
            Role::Tool if !m.content.contains("<tool_response>") => {
                out.push_str("<tool_response>\n");
                out.push_str(&m.content);
                out.push_str("\n</tool_response>");
            }
            _ => out.push_str(&m.content),
        }
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out.push_str(assistant_prefix(enable_thinking));
    out
}

/// Returns the answer part of a reply that may contain a `…</think>` block.
pub fn strip_thinking(content: &str) -> &str {
    match content.rfind("</think>") {
        Some(i) => content[i + "</think>".len()..].trim(),
        None => content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOS: &str = "<|startoftext|>";

    #[test]
    fn single_turn_without_thinking() {
        let msgs = [Message::new(Role::System, "Sé breve."), Message::new(Role::User, "Hola")];
        assert_eq!(
            render_prompt(BOS, &msgs, None, false),
            "<|startoftext|><|im_start|>system\nSé breve.<|im_end|>\n\
             <|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
    }

    #[test]
    fn history_drops_thinking() {
        let msgs = [
            Message::new(Role::User, "a"),
            Message::new(Role::Assistant, "pensando</think>respuesta"),
            Message::new(Role::User, "b"),
        ];
        assert_eq!(
            render_prompt(BOS, &msgs, None, true),
            "<|startoftext|><|im_start|>user\na<|im_end|>\n\
             <|im_start|>assistant\n<think></think>respuesta<|im_end|>\n\
             <|im_start|>user\nb<|im_end|>\n<|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn non_thinking_history_is_prefix_stable() {
        // Without thinking, the generated turn re-renders identically, so the
        // previous prompt + reply is a prefix of the next prompt (cache reuse).
        let mut msgs = vec![Message::new(Role::User, "a")];
        let p1 = render_prompt(BOS, &msgs, None, false);
        msgs.push(Message::new(Role::Assistant, "hola"));
        msgs.push(Message::new(Role::User, "b"));
        let p2 = render_prompt(BOS, &msgs, None, false);
        assert!(p2.starts_with(&format!("{p1}hola<|im_end|>\n")));
    }

    #[test]
    fn tools_block_matches_template() {
        let msgs = [Message::new(Role::System, "Sys"), Message::new(Role::User, "hola")];
        let p = render_prompt(BOS, &msgs, Some("{\"name\": \"x\"}"), false);
        assert!(p.starts_with(
            "<|startoftext|><|im_start|>system\nSys\n\n<tools>\n{\"name\": \"x\"}\n</tools><|im_end|>\n"
        ));
        let p = render_prompt(BOS, &msgs[1..], Some("T"), false);
        assert!(p.starts_with("<|startoftext|><|im_start|>system\n<tools>\nT\n</tools><|im_end|>\n"));
    }

    #[test]
    fn tool_loop_is_prefix_stable() {
        // The raw in-loop assistant turn = generation prefix + generated text.
        let mut msgs = vec![Message::new(Role::User, "1234*5678?")];
        let p1 = render_prompt(BOS, &msgs, Some("T"), false);
        let generated = "<tool_call>\n<function=calculator>\n</function>\n</tool_call>";
        let raw = format!("{}{generated}", assistant_prefix(false));
        msgs.push(Message::new(Role::Assistant, raw));
        msgs.push(Message::new(Role::Tool, "7006652"));
        let p2 = render_prompt(BOS, &msgs, Some("T"), false);
        assert!(p2.starts_with(&format!("{p1}{generated}<|im_end|>\n")), "{p2}");
        assert!(p2.ends_with(
            "<|im_start|>tool\n<tool_response>\n7006652\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think></think>"
        ));
    }
}
