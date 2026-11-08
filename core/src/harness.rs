//! The tool loop: generate → (tool call → validate → confirm → execute → feed
//! result back) → … → final answer, bounded by [`Limits`].

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use crate::chat::{Message, Role, assistant_prefix};
use crate::engine::{Engine, Event, GenParams, GenStats};
use crate::store::Store;
use crate::tools::{self, Limits, Policy, ToolContext};

pub struct TurnRequest<'a> {
    pub messages: &'a [Message],
    pub params: &'a GenParams,
    /// Names of the tools the user enabled; empty = plain chat.
    pub enabled_tools: &'a [String],
    pub now: &'a str,
    pub limits: &'a Limits,
}

#[derive(Debug, Clone, Copy)]
pub enum TurnEvent<'a> {
    Text(&'a str),
    ThinkEnd,
    /// The model is writing a tool call.
    ToolCallStarted,
    /// A parsed call (before validation). `args` is a JSON object.
    ToolCall { name: &'a str, args: &'a str },
    ToolResult { name: &'a str, ok: bool, content: &'a str },
    /// A new assistant message begins (after a tool result).
    NewMessage,
}

/// A message produced during the turn, ready to be stored and re-sent.
#[derive(Debug, Clone, Serialize)]
pub struct TurnMessage {
    pub role: &'static str,
    /// What goes back into the prompt on later turns.
    pub content: String,
    pub reasoning: String,
    /// JSON for tool rounds (`{"tool_call": …}` / `{"tool": …}`).
    pub meta: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub prompt_tps: f64,
    pub gen_tokens: usize,
    pub gen_tps: f64,
    pub tool_calls: usize,
    /// Calls that were rejected, invalid or failed: nothing was done for them,
    /// whatever the model's final answer claims.
    pub failed_calls: usize,
    pub cancelled: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TurnOutput {
    pub messages: Vec<TurnMessage>,
    pub stats: TurnStats,
}

/// Runs one user turn. `confirm(name, args_json)` is asked before any tool
/// whose policy is [`Policy::Confirm`]; returning false denies the call.
pub fn run_turn(
    engine: &mut Engine,
    store: Option<&std::sync::Mutex<Store>>,
    req: &TurnRequest,
    cancel: &AtomicBool,
    mut on_event: impl FnMut(TurnEvent<'_>),
    mut confirm: impl FnMut(&str, &str) -> bool,
) -> Result<TurnOutput> {
    let block = tools::tools_block(req.enabled_tools);
    let tools_opt = (!block.is_empty()).then_some(block.as_str());
    let mut convo = req.messages.to_vec();
    let mut out = Vec::new();
    let mut stats = TurnStats::default();
    let (mut gen_ms, mut gen_tokens) = (0.0, 0usize);
    // A final round is always allowed after the last permitted call.
    let max_rounds = req.limits.max_calls_per_turn + 2;

    for round in 0..max_rounds {
        if round > 0 {
            on_event(TurnEvent::NewMessage);
        }
        let round_out = engine.generate(&convo, tools_opt, req.params, cancel, |ev| match ev {
            Event::Text(t) => on_event(TurnEvent::Text(t)),
            Event::ThinkEnd => on_event(TurnEvent::ThinkEnd),
            Event::ToolCallStarted => on_event(TurnEvent::ToolCallStarted),
        })?;
        record(&mut stats, &round_out.stats);
        gen_ms += round_out.stats.gen_ms;
        gen_tokens += round_out.stats.gen_tokens;

        let Some(raw_call) = round_out.tool_call else {
            let (reasoning, content) = split_reasoning(&round_out.text, req.params.enable_thinking);
            if !content.is_empty() || !reasoning.is_empty() {
                out.push(TurnMessage { role: "assistant", content, reasoning, meta: None });
            }
            stats.cancelled = round_out.stats.cancelled;
            break;
        };

        // The raw turn is kept verbatim so the next round's prompt extends the
        // cached tokens exactly.
        let assistant_raw = format!("{}{}", assistant_prefix(req.params.enable_thinking), round_out.text);
        let (result, ok, call_meta) = if stats.tool_calls >= req.limits.max_calls_per_turn {
            on_event(TurnEvent::ToolCall { name: "?", args: "{}" });
            (
                format!(
                    "ACCIÓN NO REALIZADA: límite de {} llamadas por mensaje alcanzado. \
                     Responde al usuario con lo que ya sabes.",
                    req.limits.max_calls_per_turn
                ),
                false,
                json!({"name": null, "error": "limit"}),
            )
        } else {
            stats.tool_calls += 1;
            handle_call(&raw_call, store, req, &mut on_event, &mut confirm, cancel)
        };
        if !ok {
            stats.failed_calls += 1;
        }
        let tool_name = call_meta["name"].as_str().unwrap_or("?").to_string();
        let result = tools::sanitize_result(&result, req.limits.max_result_chars);
        on_event(TurnEvent::ToolResult { name: &tool_name, ok, content: &result });

        let assistant = TurnMessage {
            role: "assistant",
            content: assistant_raw,
            reasoning: String::new(),
            meta: Some(json!({ "tool_call": call_meta }).to_string()),
        };
        let tool = TurnMessage {
            role: "tool",
            content: result,
            reasoning: String::new(),
            meta: Some(json!({ "tool": tool_name, "ok": ok }).to_string()),
        };
        convo.push(Message::new(Role::Assistant, assistant.content.clone()));
        convo.push(Message::new(Role::Tool, tool.content.clone()));
        out.push(assistant);
        out.push(tool);

        // A rejected action ends the turn with a fixed message: small models
        // tend to claim the action happened anyway.
        if call_meta["denied"] == json!(true) {
            let text = format!("No se realizó «{tool_name}» porque lo rechazaste. No se ha modificado nada.");
            on_event(TurnEvent::NewMessage);
            on_event(TurnEvent::Text(&text));
            out.push(TurnMessage { role: "assistant", content: text, reasoning: String::new(), meta: None });
            break;
        }
        if cancel.load(Ordering::Relaxed) {
            stats.cancelled = true;
            break;
        }
        if round + 1 == max_rounds {
            out.push(TurnMessage {
                role: "assistant",
                content: "No pude terminar: el modelo siguió pidiendo herramientas.".into(),
                reasoning: String::new(),
                meta: None,
            });
        }
    }

    stats.gen_tokens = gen_tokens;
    stats.gen_tps = if gen_ms > 0.0 { gen_tokens as f64 * 1000.0 / gen_ms } else { 0.0 };
    Ok(TurnOutput { messages: out, stats })
}

/// Last round's prompt figures are the meaningful ones.
fn record(stats: &mut TurnStats, g: &GenStats) {
    stats.prompt_tokens = g.prompt_tokens;
    stats.reused_tokens = g.reused_tokens;
    stats.prompt_tps = g.prompt_tps();
}

fn handle_call(
    raw: &str,
    store: Option<&std::sync::Mutex<Store>>,
    req: &TurnRequest,
    on_event: &mut impl FnMut(TurnEvent<'_>),
    confirm: &mut impl FnMut(&str, &str) -> bool,
    cancel: &AtomicBool,
) -> (String, bool, serde_json::Value) {
    let call = match tools::parse_tool_call(raw) {
        Ok(c) => c,
        Err(e) => {
            on_event(TurnEvent::ToolCall { name: "?", args: "{}" });
            return (failure(&format!("llamada mal formada: {e}")), false, json!({"name": null, "error": "parse"}));
        }
    };
    let args = call.args_json();
    let meta = json!({"name": call.name, "args": serde_json::from_str::<serde_json::Value>(&args).unwrap_or_default()});
    on_event(TurnEvent::ToolCall { name: &call.name, args: &args });

    let def = match tools::validate(&call, req.enabled_tools, req.limits) {
        Ok(d) => d,
        Err(e) => return (failure(&e.to_string()), false, meta),
    };
    if def.policy == Policy::Confirm && !confirm(&call.name, &args) {
        let msg = format!("ACCIÓN NO REALIZADA: el usuario rechazó `{}`. No se ha modificado nada.", call.name);
        let mut meta = meta;
        meta["denied"] = json!(true);
        return (msg, false, meta);
    }
    if cancel.load(Ordering::Relaxed) {
        return (failure("cancelado por el usuario"), false, meta);
    }
    let guard = store.and_then(|m| m.lock().ok());
    let ctx = ToolContext { store: guard.as_deref(), now: req.now, limits: req.limits };
    match tools::execute(&call, &ctx) {
        Ok(r) => (r, true, meta),
        Err(e) => (failure(&format!("{e:#}")), false, meta),
    }
}

/// Result text for a call that did nothing, worded so the model reports it.
fn failure(reason: &str) -> String {
    format!("ACCIÓN NO REALIZADA (error: {reason}). No se ha hecho nada; díselo al usuario.")
}

/// Splits generated text into (reasoning, answer) at the last `</think>`.
/// With thinking enabled the whole text is reasoning until `</think>`.
pub fn split_reasoning(text: &str, enable_thinking: bool) -> (String, String) {
    match text.rfind("</think>") {
        Some(i) => (text[..i].trim().to_string(), text[i + "</think>".len()..].trim().to_string()),
        None if enable_thinking => (text.trim().to_string(), String::new()),
        None => (String::new(), text.trim().to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::split_reasoning;

    #[test]
    fn splits_reasoning() {
        assert_eq!(split_reasoning("hola", false), ("".into(), "hola".into()));
        assert_eq!(split_reasoning("pienso</think> hola", false), ("pienso".into(), "hola".into()));
        assert_eq!(split_reasoning("pienso…", true), ("pienso…".into(), "".into()));
    }
}
