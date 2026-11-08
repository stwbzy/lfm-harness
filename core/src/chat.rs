//! Chat formatting for LFM3.
//!
//! llama.cpp's built-in template engine cannot run the Jinja template shipped in
//! the LFM3 GGUF, so the subset the chat needs is reproduced here:
//! `<|startoftext|>` + ChatML turns, with assistant turns prefixed by a
//! `<think>…</think>` block. Assistant turns before the last user message get an
//! empty `<think></think>` (thinking is dropped from history), exactly as the
//! model's template does. Tool calls are not rendered yet.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
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

/// Renders the conversation and opens a new assistant turn.
///
/// `bos` is the model's BOS token text (`<|startoftext|>` for LFM3).
/// With `enable_thinking` the prompt ends in `<think>` and the model reasons
/// first; otherwise it ends in `<think></think>` and answers directly.
pub fn render_prompt(bos: &str, messages: &[Message], enable_thinking: bool) -> String {
    let mut out = String::from(bos);
    let mut rest = messages;
    if let Some(first) = messages.first()
        && first.role == Role::System
    {
        if !first.content.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(&first.content);
            out.push_str("<|im_end|>\n");
        }
        rest = &messages[1..];
    }
    for m in rest {
        out.push_str("<|im_start|>");
        out.push_str(m.role.as_str());
        out.push('\n');
        if m.role == Role::Assistant {
            // History turns: thinking stripped, empty think block kept.
            let content = strip_thinking(&m.content);
            out.push_str("<think></think>");
            out.push_str(content);
        } else {
            out.push_str(&m.content);
        }
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out.push_str(if enable_thinking { "<think>" } else { "<think></think>" });
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
            render_prompt(BOS, &msgs, false),
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
            render_prompt(BOS, &msgs, true),
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
        let p1 = render_prompt(BOS, &msgs, false);
        msgs.push(Message::new(Role::Assistant, "hola"));
        msgs.push(Message::new(Role::User, "b"));
        let p2 = render_prompt(BOS, &msgs, false);
        assert!(p2.starts_with(&format!("{p1}hola<|im_end|>\n")));
    }
}
