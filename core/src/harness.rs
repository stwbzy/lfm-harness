//! The tool loop: generate → (tool call → validate → confirm → execute → feed
//! result back) → … → final answer, bounded by [`Limits`].

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use crate::chat::{Message, Role, assistant_prefix};
use crate::codecheck::{self, CodeIntent};
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
    /// Detect code requests and answer them in code mode.
    pub code_mode: bool,
    /// 0 for a normal turn; N > 0 when this turn is the N-th attempt to fix
    /// a previous answer (samples hotter, see [`repair_profile`]).
    pub repair_attempt: u32,
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
    /// Code mode only: `rust_ok`, `rust_repaired`, `rust_syntax_error` or
    /// `unchecked` (another language). Syntax only — nothing is compiled.
    pub code_check: Option<String>,
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
    let intent = req
        .code_mode
        .then(|| req.messages.iter().rev().find(|m| m.role == Role::User))
        .flatten()
        .and_then(|m| codecheck::detect_code_intent(&m.content));
    let code_params = intent.as_ref().map(|i| code_profile(req.params, i)).map(|p| {
        if req.repair_attempt > 0 { repair_profile(&p, req.repair_attempt) } else { p }
    });
    let params = code_params.as_ref().unwrap_or(req.params);
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
        let round_out = engine.generate(&convo, tools_opt, params, cancel, |ev| match ev {
            Event::Text(t) => on_event(TurnEvent::Text(t)),
            Event::ThinkEnd => on_event(TurnEvent::ThinkEnd),
            Event::ToolCallStarted => on_event(TurnEvent::ToolCallStarted),
        })?;
        record(&mut stats, &round_out.stats);
        gen_ms += round_out.stats.gen_ms;
        gen_tokens += round_out.stats.gen_tokens;

        let Some(raw_call) = round_out.tool_call else {
            let (mut reasoning, mut content) = split_reasoning(&round_out.text, params.enable_thinking);
            if let Some(intent) = &intent
                && !round_out.stats.cancelled
            {
                let raw = format!("{}{}", assistant_prefix(params.enable_thinking), round_out.text);
                let check = verify_code(engine, &convo, &raw, &content, intent, tools_opt, params, cancel, &mut on_event)?;
                stats.code_check = Some(check.status.into());
                if let Some((r, c, g)) = check.replacement {
                    (reasoning, content) = (r, c);
                    gen_ms += g.gen_ms;
                    gen_tokens += g.gen_tokens;
                }
            }
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

/// Sampling for code: deterministic-ish, no repetition penalty (code repeats
/// `}`, `;`, identifiers legitimately), answer forced into a fenced block that
/// ends generation when closed, a short reasoning budget and no tool calls.
pub fn code_profile(base: &GenParams, intent: &CodeIntent) -> GenParams {
    GenParams {
        temperature: 0.15,
        top_k: 40,
        top_p: 1.0,
        min_p: 0.1,
        repeat_penalty: 1.0,
        max_tokens: base.max_tokens.max(1536),
        answer_prefill: intent.prefill(),
        stop_after: vec!["```".into()],
        max_think_tokens: if base.enable_thinking { 384 } else { 0 },
        allow_tool_calls: false,
        ..base.clone()
    }
}

/// Sampling for a repair attempt. At code-mode temperature the model tends to
/// return the very same answer even after seeing the error (9 of 26 repairs in
/// the benchmark), so repairs sample hotter and with a different seed.
pub fn repair_profile(params: &GenParams, attempt: u32) -> GenParams {
    GenParams {
        temperature: params.temperature.max(0.6),
        seed: params.seed.wrapping_add(attempt.wrapping_mul(7919)),
        ..params.clone()
    }
}

struct CodeCheck {
    status: &'static str,
    /// (reasoning, content, stats) of the repaired answer, if one was made.
    replacement: Option<(String, String, GenStats)>,
}

/// Rust answers are syntax-checked with `syn`; on error the model gets one
/// repair round with a single, precise error message.
#[allow(clippy::too_many_arguments)]
fn verify_code(
    engine: &mut Engine,
    convo: &[Message],
    raw_answer: &str,
    content: &str,
    intent: &CodeIntent,
    tools: Option<&str>,
    params: &GenParams,
    cancel: &AtomicBool,
    on_event: &mut impl FnMut(TurnEvent<'_>),
) -> Result<CodeCheck> {
    let is_rust = intent.lang == Some("rust") || content.starts_with("```rust") || content.starts_with("```rs");
    let unchecked = CodeCheck { status: "unchecked", replacement: None };
    if !is_rust {
        return Ok(unchecked);
    }
    let Some(code) = codecheck::first_code_block(content) else { return Ok(unchecked) };
    let err = match codecheck::check_rust_syntax(code) {
        Ok(_) => return Ok(CodeCheck { status: "rust_ok", replacement: None }),
        Err(e) => e,
    };
    let feedback = err.feedback();
    on_event(TurnEvent::ToolCall { name: "rust_syntax_check", args: "{}" });
    on_event(TurnEvent::ToolResult { name: "rust_syntax_check", ok: false, content: &feedback });
    on_event(TurnEvent::NewMessage);

    let mut repair = convo.to_vec();
    repair.push(Message::new(Role::Assistant, raw_answer));
    repair.push(Message::new(Role::User, feedback));
    let repair_params = repair_profile(params, 1);
    let fixed = engine.generate(&repair, tools, &repair_params, cancel, |ev| match ev {
        Event::Text(t) => on_event(TurnEvent::Text(t)),
        Event::ThinkEnd => on_event(TurnEvent::ThinkEnd),
        Event::ToolCallStarted => {}
    })?;
    let (reasoning, content) = split_reasoning(&fixed.text, params.enable_thinking);
    let ok = codecheck::first_code_block(&content).is_some_and(|c| codecheck::check_rust_syntax(c).is_ok());
    Ok(CodeCheck {
        status: if ok { "rust_repaired" } else { "rust_syntax_error" },
        replacement: Some((reasoning, content, fixed.stats)),
    })
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
