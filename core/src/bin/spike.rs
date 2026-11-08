//! Terminal chat to validate the engine on host and on the phone (via adb shell).
//!
//! Usage: spike <model.gguf> [--think] [--tools] [--db FILE] [--ctx N] [--threads N] [--batch-threads N]
//!        spike <model.gguf> --bench "prompt"   (one answer, then stats)
//!
//! With `--tools` every built-in tool is enabled; confirmations are read from stdin (y/n).

use std::io::{BufRead, Write};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use harness_core::harness::{TurnEvent, TurnRequest, run_turn};
use harness_core::store::Store;
use harness_core::tools::{Limits, TOOLS};
use harness_core::{Engine, EngineConfig, GenParams, Message, Role};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("usage: spike <model.gguf> [--think] [--tools] [--db FILE] [--ctx N] [--bench PROMPT]")?;
    let mut cfg = EngineConfig::default();
    let mut params = GenParams::default();
    let mut bench = None;
    let mut tools = Vec::new();
    let mut db = None;
    let mut code_mode = true;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--think" => params.enable_thinking = true,
            "--tools" => tools = TOOLS.iter().map(|t| t.name.to_string()).collect(),
            "--no-code-mode" => code_mode = false,
            "--db" => db = Some(args.next().context("--db FILE")?),
            "--ctx" => cfg.n_ctx = args.next().context("--ctx N")?.parse()?,
            "--threads" => cfg.n_threads = args.next().context("--threads N")?.parse()?,
            "--batch-threads" => cfg.n_threads_batch = args.next().context("--batch-threads N")?.parse()?,
            "--bench" => bench = Some(args.next().context("--bench PROMPT")?),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let store = Mutex::new(match db {
        Some(p) => Store::open(p)?,
        None => Store::open_in_memory()?,
    });

    let t = std::time::Instant::now();
    let mut engine = Engine::load(&model, cfg)?;
    eprintln!("[loaded in {:.1}s, n_ctx={}]", t.elapsed().as_secs_f32(), engine.n_ctx());

    let mut history = vec![Message::new(
        Role::System,
        "Eres un asistente útil. Responde en el idioma del usuario.",
    )];
    let cancel = AtomicBool::new(false);
    let limits = Limits::default();
    let stdin = std::io::stdin();
    let lines = std::cell::RefCell::new(stdin.lock().lines());

    loop {
        let input = match &bench {
            Some(p) => p.clone(),
            None => {
                eprint!("\n> ");
                match lines.borrow_mut().next() {
                    Some(l) => l?,
                    None => break,
                }
            }
        };
        if input.trim().is_empty() {
            continue;
        }
        if input.trim() == "/new" {
            // New chat, engine cache kept (exercises the system checkpoint).
            history.truncate(1);
            continue;
        }
        if input.trim() == "/reset" {
            history.truncate(1);
            engine.reset();
            continue;
        }
        history.push(Message::new(Role::User, input));
        let req = TurnRequest {
            messages: &history,
            params: &params,
            enabled_tools: &tools,
            now: "sábado, 7 de noviembre de 2026, 21:30 (Europe/Madrid, UTC+01:00)",
            limits: &limits,
            code_mode,
            repair_attempt: 0,
        };
        let out = run_turn(
            &mut engine,
            Some(&store),
            &req,
            &cancel,
            |ev| {
                match ev {
                    TurnEvent::Text(t) => print!("{t}"),
                    TurnEvent::ThinkEnd => print!("\n--- (fin del razonamiento) ---\n"),
                    TurnEvent::ToolCallStarted => print!("\n[llamando herramienta…]"),
                    TurnEvent::ToolCall { name, args } => print!("\n[tool] {name} {args}"),
                    TurnEvent::ToolResult { name, ok, content } => {
                        print!("\n[{}] {name} → {content}\n", if ok { "ok" } else { "error" })
                    }
                    TurnEvent::NewMessage => {}
                }
                let _ = std::io::stdout().flush();
            },
            |name, args| {
                eprint!("\n¿Permitir {name} {args}? [y/N] ");
                matches!(lines.borrow_mut().next(), Some(Ok(l)) if l.trim().eq_ignore_ascii_case("y"))
            },
        )?;
        println!();
        let s = &out.stats;
        eprintln!(
            "[prompt {} tok ({} cached) {:.1} tok/s | gen {} tok {:.1} tok/s | {} tool calls]",
            s.prompt_tokens, s.reused_tokens, s.prompt_tps, s.gen_tokens, s.gen_tps, s.tool_calls
        );
        for m in out.messages {
            history.push(Message::new(Role::parse(m.role).unwrap(), m.content));
        }
        if bench.is_some() {
            break;
        }
    }
    Ok(())
}
