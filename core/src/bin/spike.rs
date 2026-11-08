//! Terminal chat to validate the engine on host and on the phone (via adb shell).
//!
//! Usage: spike <model.gguf> [--think] [--ctx N] [--threads N]
//!        spike <model.gguf> --bench "prompt"   (one answer, then stats)

use std::io::{BufRead, Write};
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use harness_core::{Engine, EngineConfig, Event, GenParams, Message, Role};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("usage: spike <model.gguf> [--think] [--ctx N] [--threads N] [--bench PROMPT]")?;
    let mut cfg = EngineConfig::default();
    let mut params = GenParams::default();
    let mut bench = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--think" => params.enable_thinking = true,
            "--ctx" => cfg.n_ctx = args.next().context("--ctx N")?.parse()?,
            "--threads" => cfg.n_threads = args.next().context("--threads N")?.parse()?,
            "--batch-threads" => cfg.n_threads_batch = args.next().context("--batch-threads N")?.parse()?,
            "--bench" => bench = Some(args.next().context("--bench PROMPT")?),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    let t = std::time::Instant::now();
    let mut engine = Engine::load(&model, cfg)?;
    eprintln!("[loaded in {:.1}s, n_ctx={}]", t.elapsed().as_secs_f32(), engine.n_ctx());

    let mut history = vec![Message::new(
        Role::System,
        "Eres un asistente útil. Responde en el idioma del usuario.",
    )];
    let cancel = AtomicBool::new(false);
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        let input = match &bench {
            Some(p) => p.clone(),
            None => {
                eprint!("\n> ");
                match lines.next() {
                    Some(l) => l?,
                    None => break,
                }
            }
        };
        if input.trim().is_empty() {
            continue;
        }
        if input.trim() == "/reset" {
            history.truncate(1);
            engine.reset();
            continue;
        }
        history.push(Message::new(Role::User, input));
        let mut reply = String::new();
        let stats = engine.generate(&history, &params, &cancel, |ev| {
            match ev {
                Event::Text(t) => {
                    print!("{t}");
                    reply.push_str(t);
                }
                Event::ThinkEnd => {
                    print!("\n--- (fin del razonamiento) ---\n");
                    reply.push_str("</think>");
                }
            }
            let _ = std::io::stdout().flush();
        })?;
        println!();
        eprintln!(
            "[prompt {} tok ({} cached) {:.1} tok/s | gen {} tok {:.1} tok/s]",
            stats.prompt_tokens,
            stats.reused_tokens,
            stats.prompt_tps(),
            stats.gen_tokens,
            stats.gen_tps()
        );
        history.push(Message::new(Role::Assistant, reply));
        if bench.is_some() {
            break;
        }
    }
    Ok(())
}
