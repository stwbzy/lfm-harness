//! Rust coding benchmark for the local model.
//!
//! For each task in `bench/rust_tasks.txt`: ask the model, extract its ```rust
//! block, append hidden tests, compile with `rustc --test` and run. Optionally
//! feed compiler/test errors back for repair rounds.
//!
//! cargo run --release --example codebench -- <model.gguf> [--think] [--repair N]
//!     [--only id,id] [--out results.jsonl] [--temp T] [--tasks FILE]
//! cargo run --release --example codebench -- --refs refs.txt   (validate tests)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use harness_core::codecheck;
use harness_core::harness::{TurnRequest, run_turn, split_reasoning};
use harness_core::tools::Limits;
use harness_core::{Engine, EngineConfig, Event, GenParams, Message, Role};
use serde::Serialize;

const SYSTEM: &str = "Eres un programador experto en Rust. Responde con el código completo en un único bloque ```rust, sin función main salvo que se pida.";

struct Task {
    id: String,
    prompt: String,
    tests: String,
    forbid: Vec<String>,
}

fn load_tasks(path: &Path) -> Result<Vec<Task>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut tasks = Vec::new();
    for chunk in text.split("\n=== ").skip(1) {
        let (id, body) = chunk.split_once('\n').context("task without body")?;
        let mut sections: BTreeMap<&str, String> = BTreeMap::new();
        let mut current = "";
        for line in body.lines() {
            if let Some(name) = line.strip_prefix("--- ") {
                current = name.trim();
                sections.entry(current).or_default();
            } else if !current.is_empty() {
                let s = sections.get_mut(current).unwrap();
                s.push_str(line);
                s.push('\n');
            }
        }
        tasks.push(Task {
            id: id.trim().to_string(),
            prompt: sections.get("prompt").cloned().unwrap_or_default().trim().to_string(),
            tests: sections.get("tests").cloned().unwrap_or_default(),
            forbid: sections
                .get("forbid")
                .map(|s| s.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect())
                .unwrap_or_default(),
        });
    }
    Ok(tasks)
}

/// Picks the fenced code block most likely to be the solution.
fn extract_code(reply: &str) -> Option<String> {
    let mut blocks = Vec::new();
    let mut rest = reply;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let (lang, body) = after.split_once('\n')?;
        let Some(end) = body.find("```") else {
            // Unterminated block (truncated output): take what is there.
            blocks.push((lang.trim().to_string(), body.to_string()));
            break;
        };
        blocks.push((lang.trim().to_string(), body[..end].to_string()));
        rest = &body[end + 3..];
    }
    let rust: Vec<_> = blocks.iter().filter(|(l, _)| l.is_empty() || l == "rust" || l == "rs").collect();
    rust.iter().max_by_key(|(_, b)| b.len()).map(|(_, b)| b.clone())
}

#[derive(Serialize, Clone)]
struct Attempt {
    outcome: String,
    tests_passed: usize,
    tests_failed: usize,
    error_codes: Vec<String>,
    /// Errors located in the hidden tests (signature/interface mismatch).
    interface_errors: usize,
    gen_tokens: usize,
    gen_seconds: f64,
    diagnostics: String,
    /// First error only (what `--harness` feeds back).
    short_feedback: String,
    /// Harness syntax check result (`--harness` only).
    code_check: Option<String>,
}

#[derive(Serialize)]
struct TaskResult {
    id: String,
    think: bool,
    attempts: Vec<Attempt>,
    reply: Vec<String>,
}

fn check(code: &str, task: &Task, workdir: &Path) -> Attempt {
    let mut a = Attempt {
        outcome: String::new(),
        tests_passed: 0,
        tests_failed: 0,
        error_codes: vec![],
        interface_errors: 0,
        gen_tokens: 0,
        gen_seconds: 0.0,
        diagnostics: String::new(),
        short_feedback: String::new(),
        code_check: None,
    };
    if let Some(f) = task.forbid.iter().find(|f| code.contains(f.as_str())) {
        a.outcome = "forbidden".into();
        a.diagnostics = format!("uses forbidden `{f}`");
        return a;
    }
    let src = workdir.join(format!("{}.rs", task.id));
    let bin = workdir.join(format!("{}.bin", task.id));
    let code_lines = code.lines().count() + 2;
    let full = format!(
        "#![allow(dead_code, unused)]\n{code}\n#[cfg(test)]\nmod hidden_tests {{\n    use super::*;\n{}\n}}\n",
        task.tests
    );
    std::fs::write(&src, full).unwrap();
    let out = Command::new("rustc")
        .args(["--edition", "2021", "--test", "-C", "debuginfo=0", "-o"])
        .arg(&bin)
        .arg(&src)
        .output()
        .expect("rustc");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).to_string();
        a.outcome = "compile_error".into();
        let mut codes: Vec<String> = err
            .lines()
            .filter_map(|l| l.strip_prefix("error[").and_then(|r| r.split(']').next()).map(String::from))
            .collect();
        codes.sort();
        codes.dedup();
        a.error_codes = codes;
        // "--> file.rs:LINE:COL" after the solution's lines = inside the tests.
        a.interface_errors = err
            .lines()
            .filter_map(|l| l.trim().strip_prefix("--> "))
            .filter_map(|loc| loc.rsplitn(3, ':').nth(1)?.parse::<usize>().ok())
            .filter(|&line| line > code_lines + 1)
            .count();
        a.diagnostics = trim_diag(&err);
        a.short_feedback = codecheck::first_rustc_error(&err);
        return a;
    }
    let mut child = Command::new(&bin).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let start = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if start.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            a.outcome = "timeout".into();
            return a;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    if let Some(line) = stdout.lines().find(|l| l.starts_with("test result:")) {
        let num = |key: &str| {
            line.split(';')
                .find(|p| p.contains(key))
                .and_then(|p| p.split_whitespace().rev().nth(1)?.parse().ok())
                .unwrap_or(0)
        };
        a.tests_passed = num("passed");
        a.tests_failed = num("failed");
    }
    a.outcome = if out.status.success() { "pass".into() } else { "test_fail".into() };
    if !out.status.success() {
        a.diagnostics = trim_diag(&stdout);
        a.short_feedback = codecheck::first_test_failure(&stdout);
    }
    a
}

fn trim_diag(s: &str) -> String {
    let keep: Vec<&str> = s
        .lines()
        .filter(|l| !l.starts_with("warning") && !l.contains("= note: `#[warn"))
        .collect();
    let joined = keep.join("\n");
    joined.chars().take(2500).collect()
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut think = false;
    let mut repair = 0usize;
    let mut only: Option<Vec<String>> = None;
    let mut out_path = PathBuf::from("codebench-results.jsonl");
    let mut tasks_path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/bench/rust_tasks.txt"));
    let mut refs = None;
    let mut temp = None;
    let mut harness_mode = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--think" => think = true,
            "--harness" => harness_mode = true,
            "--repair" => repair = args.next().context("--repair N")?.parse()?,
            "--only" => only = Some(args.next().context("--only ids")?.split(',').map(String::from).collect()),
            "--out" => out_path = args.next().context("--out FILE")?.into(),
            "--tasks" => tasks_path = args.next().context("--tasks FILE")?.into(),
            "--refs" => refs = Some(PathBuf::from(args.next().context("--refs FILE")?)),
            "--temp" => temp = Some(args.next().context("--temp T")?.parse::<f32>()?),
            other if model.is_none() && !other.starts_with("--") => model = Some(other.to_string()),
            other => bail!("unknown argument {other}"),
        }
    }
    let mut tasks = load_tasks(&tasks_path)?;
    if let Some(only) = &only {
        tasks.retain(|t| only.contains(&t.id));
    }
    let workdir = std::env::temp_dir().join(format!("codebench-{}", std::process::id()));
    std::fs::create_dir_all(&workdir)?;

    // Validate the hidden tests against reference solutions.
    if let Some(refs) = refs {
        let text = std::fs::read_to_string(refs)?;
        let mut ok = true;
        for chunk in text.split("\n=== ").skip(1) {
            let (id, code) = chunk.split_once('\n').unwrap();
            let task = tasks.iter().find(|t| t.id == id.trim()).context("unknown ref id")?;
            let a = check(code, task, &workdir);
            println!("{:<14} {:<14} {}/{}", task.id, a.outcome, a.tests_passed, a.tests_passed + a.tests_failed);
            if a.outcome != "pass" {
                ok = false;
                println!("{}", a.diagnostics);
            }
        }
        std::fs::remove_dir_all(&workdir).ok();
        if !ok {
            bail!("reference check failed");
        }
        return Ok(());
    }

    let model = model.context("usage: codebench <model.gguf> [--think] [--repair N]")?;
    let mut engine = Engine::load(&model, EngineConfig { n_ctx: 8192, ..Default::default() })?;
    let mut params = GenParams { enable_thinking: think, max_tokens: if think { 3072 } else { 1536 }, ..Default::default() };
    if let Some(t) = temp {
        params.temperature = t;
    }
    let cancel = AtomicBool::new(false);
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path)?;
    let mut summary = Vec::new();

    for task in &tasks {
        let mut convo = vec![Message::new(Role::System, SYSTEM), Message::new(Role::User, task.prompt.clone())];
        let mut result = TaskResult { id: task.id.clone(), think, attempts: vec![], reply: vec![] };
        for round in 0..=repair {
            let t0 = Instant::now();
            let (answer, gen_tokens, code_check) = if harness_mode {
                // Same path as the app: code mode, syn check + one repair.
                let limits = Limits::default();
                let req = TurnRequest {
                    messages: &convo,
                    params: &params,
                    enabled_tools: &[],
                    now: "",
                    limits: &limits,
                    code_mode: true,
                    repair_attempt: round as u32,
                };
                let out = run_turn(&mut engine, None, &req, &cancel, |_| {}, |_, _| false)?;
                let answer = out.messages.last().map(|m| m.content.clone()).unwrap_or_default();
                (answer, out.stats.gen_tokens, out.stats.code_check)
            } else {
                let reply = engine.generate(&convo, None, &params, &cancel, |_ev: Event| {})?;
                (split_reasoning(&reply.text, think).1, reply.stats.gen_tokens, None)
            };
            let mut attempt = match extract_code(&answer) {
                Some(code) => check(&code, task, &workdir),
                None => Attempt {
                    outcome: "no_code".into(),
                    tests_passed: 0,
                    tests_failed: 0,
                    error_codes: vec![],
                    interface_errors: 0,
                    gen_tokens: 0,
                    gen_seconds: 0.0,
                    diagnostics: String::new(),
                    short_feedback: String::new(),
                    code_check: None,
                },
            };
            attempt.gen_tokens = gen_tokens;
            attempt.code_check = code_check;
            attempt.gen_seconds = t0.elapsed().as_secs_f64();
            eprintln!(
                "{:<14} round {round}: {:<14} tests {}/{} codes {:?} ({} tok, {:.0}s){}",
                task.id,
                attempt.outcome,
                attempt.tests_passed,
                attempt.tests_passed + attempt.tests_failed,
                attempt.error_codes,
                attempt.gen_tokens,
                attempt.gen_seconds,
                attempt.code_check.as_deref().map(|c| format!(" [{c}]")).unwrap_or_default()
            );
            let passed = attempt.outcome == "pass";
            let feedback = match attempt.outcome.as_str() {
                "no_code" => "No has incluido un bloque ```rust con el código. Devuelve el código completo.".to_string(),
                "forbidden" => format!("No cumple los requisitos: {}. Corrígelo y devuelve el código completo.", attempt.diagnostics),
                "timeout" => "Los tests no terminan (bucle infinito o bloqueo). Corrígelo y devuelve el código completo.".into(),
                _ if harness_mode => format!(
                    "El código falla con este error (corrige solo eso):\n```\n{}\n```\nDevuelve el código completo en Rust.",
                    attempt.short_feedback
                ),
                _ => format!(
                    "El código falla. Salida del compilador/tests:\n```\n{}\n```\nCorrígelo y devuelve el código completo en un único bloque ```rust.",
                    attempt.diagnostics
                ),
            };
            result.attempts.push(attempt);
            result.reply.push(answer.clone());
            if passed || round == repair {
                break;
            }
            convo.push(Message::new(Role::Assistant, answer));
            convo.push(Message::new(Role::User, feedback));
        }
        summary.push((task.id.clone(), result.attempts.iter().map(|a| a.outcome.clone()).collect::<Vec<_>>()));
        use std::io::Write;
        writeln!(out, "{}", serde_json::to_string(&result)?)?;
        engine.reset();
    }

    let first = summary.iter().filter(|(_, o)| o[0] == "pass").count();
    let any = summary.iter().filter(|(_, o)| o.last().is_some_and(|x| x == "pass")).count();
    println!("\n== {} tasks, think={think}, repair={repair}", summary.len());
    for (id, o) in &summary {
        println!("{id:<14} {}", o.join(" -> "));
    }
    println!("pass@1: {first}/{}   after repair: {any}/{}", summary.len(), summary.len());
    std::fs::remove_dir_all(&workdir).ok();
    Ok(())
}
