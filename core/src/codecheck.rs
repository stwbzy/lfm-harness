//! Code-mode helpers: detect code requests, pull code out of replies, check
//! Rust syntax on-device (`syn`, no compiler needed) and condense compiler
//! output for repair prompts.
//!
//! On the phone there is no `rustc`, so only *syntax* and item presence can be
//! verified; type and borrow errors stay undetected and the UI must say so.

/// A request that should be answered with code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeIntent {
    /// Fence language (`rust`, `python`…) or None when not stated.
    pub lang: Option<&'static str>,
}

impl CodeIntent {
    /// What the answer is forced to start with.
    pub fn prefill(&self) -> String {
        match self.lang {
            Some(l) => format!("```{l}\n"),
            // Let the model name the language itself.
            None => "```".into(),
        }
    }
}

const CODE_WORDS: &[&str] = &[
    "código", "codigo", "programa ", "programa.", "programa,", "función", "funcion", "implementa", "implementar",
    "escribe en ", "script", "compila", "algoritmo", "refactoriza", "snippet", "```", "fn ", "struct ", "impl ",
    "write a function", "write code", "code ", "function ",
];

/// Phrases of a conceptual question ("qué es Rust", "explica…"): a language name
/// alone in such a message is not a request for code.
const QUESTION_WORDS: &[&str] = &[
    "qué es", "que es", "qué son", "que son", "explica", "explícame", "explicame", "por qué", "por que",
    "diferencia", "ventajas", "cuál", "cual ", "cómo funciona", "como funciona", "what is", "what are", "explain",
    "why ", "difference", "who ", "quién", "quien ", "historia", "cuéntame", "cuentame",
];

/// Ordered so that longer names win (`javascript` before `java`).
const LANGS: &[(&str, &[&str])] = &[
    ("rust", &["rust", ".rs "]),
    ("typescript", &["typescript"]),
    ("javascript", &["javascript", " js ", "node.js"]),
    ("python", &["python", " py "]),
    ("kotlin", &["kotlin"]),
    ("java", &["java "]),
    ("cpp", &["c++", "cpp"]),
    ("csharp", &["c#", "csharp"]),
    ("go", &["golang", " en go "]),
    ("bash", &["bash", "shell", " sh "]),
    ("sql", &[" sql", "sql "]),
    ("html", &["html"]),
    ("css", &[" css"]),
    ("c", &[" en c ", "lenguaje c "]),
];

/// Heuristic: does this user message ask for code (and in which language)?
pub fn detect_code_intent(text: &str) -> Option<CodeIntent> {
    let t = format!(" {} ", text.to_lowercase());
    let lang = LANGS.iter().find(|(_, keys)| keys.iter().any(|k| t.contains(k))).map(|(l, _)| *l);
    let asks_code = CODE_WORDS.iter().any(|w| t.contains(w));
    // A language name alone ("hazlo en rust") also counts, unless the message is
    // a conceptual question about the language ("qué es Rust").
    let asks_concept = QUESTION_WORDS.iter().any(|w| t.contains(w));
    let lang_only = !asks_concept && lang.is_some_and(|l| l != "c" && l != "go");
    (asks_code || lang_only).then_some(CodeIntent { lang })
}

/// Contents of the first fenced block (accepting an unterminated one).
pub fn first_code_block(text: &str) -> Option<&str> {
    let start = text.find("```")?;
    let after = &text[start + 3..];
    let body = &after[after.find('\n')? + 1..];
    Some(match body.find("```") {
        Some(end) => &body[..end],
        None => body,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxError {
    pub line: usize,
    pub column: usize,
    pub message: String,
    /// The offending source line.
    pub source_line: String,
}

impl SyntaxError {
    /// Short feedback for a repair round (one error at a time).
    pub fn feedback(&self) -> String {
        format!(
            "Error de sintaxis en la línea {}: {}.\nLínea: `{}`\nCorrige solo ese error y devuelve el código completo.",
            self.line,
            self.message,
            self.source_line.trim()
        )
    }
}

/// Parses Rust source with `syn`. On success returns the top-level items
/// (`fn name`, `struct Name`, `impl Trait for Type`…).
pub fn check_rust_syntax(code: &str) -> Result<Vec<String>, SyntaxError> {
    match syn::parse_file(code) {
        Ok(file) => Ok(file.items.iter().filter_map(item_name).collect()),
        Err(e) => {
            let start = e.span().start();
            Err(SyntaxError {
                line: start.line,
                column: start.column + 1,
                message: e.to_string(),
                source_line: code.lines().nth(start.line.saturating_sub(1)).unwrap_or("").to_string(),
            })
        }
    }
}

fn item_name(item: &syn::Item) -> Option<String> {
    Some(match item {
        syn::Item::Fn(f) => format!("fn {}", f.sig.ident),
        syn::Item::Struct(s) => format!("struct {}", s.ident),
        syn::Item::Enum(e) => format!("enum {}", e.ident),
        syn::Item::Trait(t) => format!("trait {}", t.ident),
        syn::Item::Impl(i) => {
            let ty = type_name(&i.self_ty);
            match &i.trait_ {
                Some((path, _)) => format!(
                    "impl {} for {ty}",
                    path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
                ),
                None => format!("impl {ty}"),
            }
        }
        _ => return None,
    })
}

fn type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default(),
        _ => "?".into(),
    }
}

/// Condenses `rustc` stderr to its first error: header, location, the code
/// line and any `help:` suggestion. Small models fix one error at a time far
/// better than they digest a wall of diagnostics.
pub fn first_rustc_error(stderr: &str) -> String {
    let mut lines = stderr.lines().skip_while(|l| !l.starts_with("error"));
    let mut out = Vec::new();
    if let Some(head) = lines.next() {
        out.push(head.to_string());
    }
    for l in lines {
        if l.starts_with("error") || l.starts_with("warning") {
            break;
        }
        let t = l.trim_start();
        // Keep location, source lines (`NN | code`), carets and help/note text.
        if t.starts_with("-->") || t.starts_with("help:") || t.starts_with("= help") || t.starts_with("= note")
            || t.split_once('|').is_some_and(|(n, _)| n.trim().chars().all(|c| c.is_ascii_digit()) && !n.trim().is_empty())
            || (t.starts_with('|') && t.contains('^'))
        {
            out.push(l.to_string());
        }
    }
    let s = out.join("\n");
    s.chars().take(900).collect()
}

/// Same idea for a failing `cargo test`/`--test` binary: the first panic.
pub fn first_test_failure(stdout: &str) -> String {
    let mut out = Vec::new();
    let mut in_block = false;
    for l in stdout.lines() {
        if l.starts_with("---- ") {
            if in_block {
                break;
            }
            in_block = true;
        }
        if in_block && !l.trim().is_empty() && !l.starts_with("note: run with") {
            out.push(l);
        }
    }
    out.join("\n").chars().take(900).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent() {
        assert_eq!(detect_code_intent("Escribe en Rust una función que sume"), Some(CodeIntent { lang: Some("rust") }));
        assert_eq!(detect_code_intent("dame un script de python"), Some(CodeIntent { lang: Some("python") }));
        assert_eq!(detect_code_intent("hazme código en javascript"), Some(CodeIntent { lang: Some("javascript") }));
        assert_eq!(detect_code_intent("implementa un algoritmo de ordenación"), Some(CodeIntent { lang: None }));
        assert_eq!(detect_code_intent("¿qué tiempo hace hoy?"), None);
        assert_eq!(detect_code_intent("cuéntame un chiste"), None);
        assert_eq!(detect_code_intent("Explica en dos frases que es Rust"), None);
        assert_eq!(detect_code_intent("what is python?"), None);
        assert_eq!(detect_code_intent("hazlo en rust"), Some(CodeIntent { lang: Some("rust") }));
        assert_eq!(CodeIntent { lang: Some("rust") }.prefill(), "```rust\n");
    }

    #[test]
    fn code_blocks() {
        assert_eq!(first_code_block("hola\n```rust\nfn a() {}\n```\nfin"), Some("fn a() {}\n"));
        assert_eq!(first_code_block("```\nx\n"), Some("x\n"), "unterminated");
        assert_eq!(first_code_block("sin código"), None);
    }

    #[test]
    fn syntax_ok_lists_items() {
        let items = check_rust_syntax("pub struct S; impl Default for S { fn default() -> Self { S } } fn f(x: i32) -> i32 { x }").unwrap();
        assert_eq!(items, vec!["struct S", "impl Default for S", "fn f"]);
    }

    #[test]
    fn syntax_error_has_location() {
        let code = "fn f() {\n    let x = ;\n}\n";
        let e = check_rust_syntax(code).unwrap_err();
        assert_eq!(e.line, 2);
        assert_eq!(e.source_line.trim(), "let x = ;");
        assert!(e.feedback().contains("línea 2"));
        // Prose mixed into code (a common failure) is a syntax error too.
        assert!(check_rust_syntax("fn f() {}\nPero para cumplir con esto, podemos hacer:\n").is_err());
    }

    #[test]
    fn condenses_rustc_output() {
        let stderr = "warning: unused\n\nerror[E0308]: mismatched types\n --> a.rs:9:13\n  |\n7 |  match token {\n  |        ----- this expression has type `i64`\n9 |  \"+\" => {\n  |  ^^^ expected `i64`, found `&str`\n\nerror[E0599]: no method\n --> a.rs:1:1\n";
        let s = first_rustc_error(stderr);
        assert!(s.starts_with("error[E0308]: mismatched types"));
        assert!(s.contains("--> a.rs:9:13") && s.contains("expected `i64`, found `&str`"));
        assert!(!s.contains("E0599"), "only the first error");
    }
}
