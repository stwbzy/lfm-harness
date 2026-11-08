//! Built-in tools the model may call.
//!
//! Everything stays inside the app: no network, no filesystem access, no
//! shell. Tools only touch the app's own SQLite store (chats, notes) or pure
//! computation. Every call is validated against an allow-list and hard limits,
//! and tools that modify data require the user's confirmation.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value, json};

use crate::store::Store;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Read-only or pure: runs without asking.
    Auto,
    /// Modifies data: the user must approve each call.
    Confirm,
}

pub struct ToolDef {
    pub name: &'static str,
    description: &'static str,
    /// (name, JSON type, description, required)
    params: &'static [(&'static str, &'static str, &'static str, bool)],
    pub policy: Policy,
}

pub const TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "calculator",
        description: "Evalúa con exactitud una expresión aritmética. Admite + - * / % ^, paréntesis, \
                      sqrt, abs, round, floor, ceil, ln, log10, sin, cos, tan, pi y e. Usa punto decimal.",
        params: &[("expression", "string", "Expresión, p. ej. (3+4)*2^3", true)],
        policy: Policy::Auto,
    },
    ToolDef {
        name: "get_datetime",
        description: "Devuelve la fecha, la hora y la zona horaria actuales del dispositivo.",
        params: &[],
        policy: Policy::Auto,
    },
    ToolDef {
        name: "search_chats",
        description: "Busca un texto en las conversaciones guardadas del usuario y devuelve fragmentos.",
        params: &[("query", "string", "Texto a buscar (2-100 caracteres)", true)],
        policy: Policy::Auto,
    },
    ToolDef {
        name: "list_notes",
        description: "Lista los títulos de las notas guardadas del usuario.",
        params: &[],
        policy: Policy::Auto,
    },
    ToolDef {
        name: "read_note",
        description: "Lee el contenido de una nota por su título.",
        params: &[("title", "string", "Título exacto de la nota", true)],
        policy: Policy::Auto,
    },
    ToolDef {
        name: "save_note",
        description: "Crea o reemplaza una nota del usuario. Requiere su aprobación.",
        params: &[
            ("title", "string", "Título corto (máx. 80 caracteres)", true),
            ("content", "string", "Contenido (máx. 4000 caracteres)", true),
        ],
        policy: Policy::Confirm,
    },
    ToolDef {
        name: "delete_note",
        description: "Borra una nota del usuario por su título. Requiere su aprobación.",
        params: &[("title", "string", "Título exacto de la nota", true)],
        policy: Policy::Confirm,
    },
];

pub fn find(name: &str) -> Option<&'static ToolDef> {
    TOOLS.iter().find(|t| t.name == name)
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub max_calls_per_turn: usize,
    pub max_args_bytes: usize,
    pub max_result_chars: usize,
    pub max_notes: usize,
    pub max_note_chars: usize,
    pub max_title_chars: usize,
    pub max_search_results: usize,
    pub max_expr_chars: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_calls_per_turn: 3,
            max_args_bytes: 2048,
            max_result_chars: 2000,
            max_notes: 100,
            max_note_chars: 4000,
            max_title_chars: 80,
            max_search_results: 5,
            max_expr_chars: 200,
        }
    }
}

pub struct ToolContext<'a> {
    pub store: Option<&'a Store>,
    /// Human-readable local date/time supplied by the app.
    pub now: &'a str,
    pub limits: &'a Limits,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Arguments in the order the model wrote them.
    pub args: Vec<(String, String)>,
}

impl ToolCall {
    pub fn arg(&self, key: &str) -> Option<&str> {
        self.args.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn args_json(&self) -> String {
        let map: Map<String, Value> = self.args.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
        Value::Object(map).to_string()
    }

    fn args_bytes(&self) -> usize {
        self.args.iter().map(|(k, v)| k.len() + v.len()).sum()
    }
}

/// Parses `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n…</function>\n</tool_call>`.
pub fn parse_tool_call(raw: &str) -> Result<ToolCall> {
    let body = raw.trim();
    let body = body.strip_prefix("<tool_call>").context("falta <tool_call>")?;
    let body = body.strip_suffix("</tool_call>").context("falta </tool_call>")?;
    let body = body.trim();
    let body = body.strip_prefix("<function=").context("falta <function=…>")?;
    let (name, mut rest) = body.split_once('>').context("etiqueta <function> sin cerrar")?;
    let name = name.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        bail!("nombre de función inválido");
    }
    let mut args = Vec::new();
    loop {
        rest = rest.trim_start();
        if let Some(r) = rest.strip_prefix("</function>") {
            if !r.trim().is_empty() {
                bail!("texto inesperado tras </function>");
            }
            break;
        }
        let r = rest.strip_prefix("<parameter=").context("se esperaba <parameter=…> o </function>")?;
        let (key, r) = r.split_once('>').context("etiqueta <parameter> sin cerrar")?;
        let (value, r) = r.split_once("</parameter>").context("falta </parameter>")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        args.push((key.trim().to_string(), value.to_string()));
        rest = r;
    }
    Ok(ToolCall { name: name.to_string(), args })
}

/// The `<tools>` block body: one JSON object per line, formatted like the
/// model's template (`", "` and `": "` separators, key order preserved).
pub fn tools_block(enabled: &[String]) -> String {
    TOOLS
        .iter()
        .filter(|t| enabled.iter().any(|e| e == t.name))
        .map(|t| {
            let mut props = Map::new();
            let mut required = Vec::new();
            for (name, ty, desc, req) in t.params {
                props.insert((*name).into(), json!({"type": ty, "description": desc}));
                if *req {
                    required.push(Value::String((*name).into()));
                }
            }
            let mut params = Map::new();
            params.insert("type".into(), "object".into());
            params.insert("properties".into(), Value::Object(props));
            if !required.is_empty() {
                params.insert("required".into(), Value::Array(required));
            }
            let mut f = Map::new();
            f.insert("name".into(), t.name.into());
            f.insert("description".into(), t.description.into());
            f.insert("parameters".into(), Value::Object(params));
            py_json(&Value::Object(f))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// JSON with Python's default separators, as the template's `tojson` emits.
fn py_json(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), py_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// Checks a parsed call against the allow-list and argument limits.
pub fn validate(call: &ToolCall, enabled: &[String], limits: &Limits) -> Result<&'static ToolDef> {
    let def = find(&call.name)
        .filter(|d| enabled.iter().any(|e| e == d.name))
        .ok_or_else(|| anyhow!("la herramienta `{}` no existe o está desactivada. Disponibles: {}", call.name, enabled.join(", ")))?;
    if call.args_bytes() > limits.max_args_bytes {
        bail!("argumentos demasiado largos (máx. {} bytes)", limits.max_args_bytes);
    }
    for (key, _) in &call.args {
        if !def.params.iter().any(|(p, ..)| p == key) {
            bail!("parámetro desconocido `{key}` para `{}`", def.name);
        }
    }
    for (p, _, _, required) in def.params {
        if *required && call.arg(p).is_none_or(|v| v.trim().is_empty()) {
            bail!("falta el parámetro obligatorio `{p}`");
        }
    }
    Ok(def)
}

/// Runs a validated call. Errors are meant to be shown to the model.
pub fn execute(call: &ToolCall, ctx: &ToolContext) -> Result<String> {
    let l = ctx.limits;
    let store = || ctx.store.ok_or_else(|| anyhow!("almacenamiento no disponible"));
    let arg = |k: &str| call.arg(k).map(str::trim).unwrap_or("");
    let out = match call.name.as_str() {
        "calculator" => {
            let expr = arg("expression");
            if expr.chars().count() > l.max_expr_chars {
                bail!("expresión demasiado larga (máx. {} caracteres)", l.max_expr_chars);
            }
            format!("{expr} = {}", format_number(calc::eval(expr)?))
        }
        "get_datetime" => ctx.now.to_string(),
        "search_chats" => {
            let q = arg("query");
            let n = q.chars().count();
            if !(2..=100).contains(&n) {
                bail!("la búsqueda debe tener entre 2 y 100 caracteres");
            }
            let hits = store()?.search_messages(q, l.max_search_results)?;
            if hits.is_empty() {
                format!("No hay resultados para \"{q}\".")
            } else {
                hits.iter()
                    .map(|h| {
                        let who = if h.role == "user" { "usuario" } else { "asistente" };
                        format!("- [{}] {who}: {}", h.conversation_title, h.snippet.replace('\n', " "))
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        "list_notes" => {
            let notes = store()?.list_notes()?;
            if notes.is_empty() {
                "No hay notas guardadas.".into()
            } else {
                notes.iter().map(|n| format!("- {}", n.title)).collect::<Vec<_>>().join("\n")
            }
        }
        "read_note" => match store()?.get_note(arg("title"))? {
            Some(n) => format!("{}\n\n{}", n.title, n.content),
            None => bail!("no existe una nota titulada \"{}\"", arg("title")),
        },
        "save_note" => {
            let (title, content) = (arg("title"), call.arg("content").unwrap_or("").trim());
            if title.chars().count() > l.max_title_chars {
                bail!("título demasiado largo (máx. {} caracteres)", l.max_title_chars);
            }
            if content.chars().count() > l.max_note_chars {
                bail!("contenido demasiado largo (máx. {} caracteres)", l.max_note_chars);
            }
            let s = store()?;
            if s.get_note(title)?.is_none() && s.note_count()? >= l.max_notes {
                bail!("límite de {} notas alcanzado; borra alguna primero", l.max_notes);
            }
            if s.save_note(title, content)? {
                format!("Nota \"{title}\" creada.")
            } else {
                format!("Nota \"{title}\" actualizada.")
            }
        }
        "delete_note" => {
            if store()?.delete_note(arg("title"))? {
                format!("Nota \"{}\" borrada.", arg("title"))
            } else {
                bail!("no existe una nota titulada \"{}\"", arg("title"))
            }
        }
        other => bail!("herramienta `{other}` no implementada"),
    };
    Ok(out)
}

/// Truncates a result and neutralizes chat-control markup so tool output
/// (which may contain user-written text) cannot fake turns or tool calls.
pub fn sanitize_result(text: &str, max_chars: usize) -> String {
    let mut s: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        s.push_str("\n[…resultado recortado]");
    }
    for tag in ["<|", "<tool", "</tool", "<think", "</think", "<function", "</function", "<parameter", "</parameter"] {
        if s.contains(tag) {
            s = s.replace(tag, &format!("<\u{200B}{}", &tag[1..]));
        }
    }
    s
}

fn format_number(x: f64) -> String {
    if x.is_nan() {
        return "indefinido".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "infinito".into() } else { "-infinito".into() };
    }
    if x.fract() == 0.0 && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    // Round away binary noise (0.1+0.2) while keeping 12 significant digits.
    let digits = 12 - (x.abs().log10().floor() as i32 + 1);
    let r = if (0..=15).contains(&digits) {
        let p = 10f64.powi(digits);
        (x * p).round() / p
    } else {
        x
    };
    format!("{r}")
}

/// Small recursive-descent evaluator (no allocation of user-controlled size
/// beyond the input, bounded recursion depth).
mod calc {
    use anyhow::{Result, bail};

    const MAX_DEPTH: usize = 64;

    pub fn eval(input: &str) -> Result<f64> {
        if input.contains(',') {
            bail!("usa punto decimal, no coma");
        }
        let src: Vec<char> = input
            .chars()
            .map(|c| match c {
                '×' => '*',
                '÷' => '/',
                '−' => '-',
                c => c,
            })
            .filter(|c| !c.is_whitespace())
            .collect();
        if src.is_empty() {
            bail!("expresión vacía");
        }
        let mut p = Parser { s: src, i: 0, depth: 0 };
        let v = p.expr()?;
        if p.i != p.s.len() {
            bail!("carácter inesperado `{}`", p.s[p.i]);
        }
        Ok(v)
    }

    struct Parser {
        s: Vec<char>,
        i: usize,
        depth: usize,
    }

    impl Parser {
        fn peek(&self) -> Option<char> {
            self.s.get(self.i).copied()
        }

        fn enter(&mut self) -> Result<()> {
            self.depth += 1;
            if self.depth > MAX_DEPTH {
                bail!("expresión demasiado anidada");
            }
            Ok(())
        }

        // expr := term (('+'|'-') term)*
        fn expr(&mut self) -> Result<f64> {
            self.enter()?;
            let mut v = self.term()?;
            while let Some(c @ ('+' | '-')) = self.peek() {
                self.i += 1;
                let r = self.term()?;
                v = if c == '+' { v + r } else { v - r };
            }
            self.depth -= 1;
            Ok(v)
        }

        // term := unary (('*'|'/'|'%') unary)*
        fn term(&mut self) -> Result<f64> {
            let mut v = self.unary()?;
            while let Some(c @ ('*' | '/' | '%')) = self.peek() {
                self.i += 1;
                let r = self.unary()?;
                v = match c {
                    '*' => v * r,
                    '/' if r == 0.0 => bail!("división por cero"),
                    '/' => v / r,
                    _ if r == 0.0 => bail!("módulo por cero"),
                    _ => v % r,
                };
            }
            Ok(v)
        }

        // unary := ('-'|'+') unary | power
        fn unary(&mut self) -> Result<f64> {
            match self.peek() {
                Some('-') => {
                    self.i += 1;
                    self.enter()?;
                    let v = -self.unary()?;
                    self.depth -= 1;
                    Ok(v)
                }
                Some('+') => {
                    self.i += 1;
                    self.unary()
                }
                _ => self.power(),
            }
        }

        // power := atom ('^' unary)?   (right-associative)
        fn power(&mut self) -> Result<f64> {
            let base = self.atom()?;
            if self.peek() == Some('^') {
                self.i += 1;
                self.enter()?;
                let e = self.unary()?;
                self.depth -= 1;
                return Ok(base.powf(e));
            }
            Ok(base)
        }

        fn atom(&mut self) -> Result<f64> {
            match self.peek() {
                Some('(') => {
                    self.i += 1;
                    let v = self.expr()?;
                    if self.peek() != Some(')') {
                        bail!("falta `)`");
                    }
                    self.i += 1;
                    Ok(v)
                }
                Some(c) if c.is_ascii_digit() || c == '.' => self.number(),
                Some(c) if c.is_ascii_alphabetic() => {
                    let start = self.i;
                    while self.peek().is_some_and(|c| c.is_ascii_alphanumeric()) {
                        self.i += 1;
                    }
                    let name: String = self.s[start..self.i].iter().collect::<String>().to_lowercase();
                    match name.as_str() {
                        "pi" => return Ok(std::f64::consts::PI),
                        "e" => return Ok(std::f64::consts::E),
                        _ => {}
                    }
                    if self.peek() != Some('(') {
                        bail!("`{name}` desconocido");
                    }
                    let x = self.atom()?;
                    Ok(match name.as_str() {
                        "sqrt" if x < 0.0 => bail!("raíz de número negativo"),
                        "sqrt" => x.sqrt(),
                        "abs" => x.abs(),
                        "round" => x.round(),
                        "floor" => x.floor(),
                        "ceil" => x.ceil(),
                        "ln" if x <= 0.0 => bail!("logaritmo de número no positivo"),
                        "ln" => x.ln(),
                        "log10" | "log" if x <= 0.0 => bail!("logaritmo de número no positivo"),
                        "log10" | "log" => x.log10(),
                        "sin" => x.sin(),
                        "cos" => x.cos(),
                        "tan" => x.tan(),
                        _ => bail!("función `{name}` desconocida"),
                    })
                }
                Some(c) => bail!("carácter inesperado `{c}`"),
                None => bail!("expresión incompleta"),
            }
        }

        fn number(&mut self) -> Result<f64> {
            let start = self.i;
            while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '.') {
                self.i += 1;
            }
            // Optional exponent: 1e6, 2.5E-3
            if self.peek().is_some_and(|c| c == 'e' || c == 'E')
                && self.s.get(self.i + 1).is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+')
            {
                self.i += 2;
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            let text: String = self.s[start..self.i].iter().collect();
            text.parse::<f64>().map_err(|_| anyhow::anyhow!("número inválido `{text}`"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<String> {
        TOOLS.iter().map(|t| t.name.to_string()).collect()
    }

    #[test]
    fn parses_model_output() {
        let raw = "<tool_call>\n<function=calculator>\n<parameter=expression>\n1234*5678\n</parameter>\n</function>\n</tool_call>";
        let c = parse_tool_call(raw).unwrap();
        assert_eq!(c.name, "calculator");
        assert_eq!(c.arg("expression"), Some("1234*5678"));
        let c = parse_tool_call("<tool_call>\n<function=get_datetime>\n</function>\n</tool_call>").unwrap();
        assert!(c.args.is_empty());
        let multi = "<tool_call>\n<function=save_note>\n<parameter=title>\nA\n</parameter>\n\
                     <parameter=content>\nlínea 1\nlínea 2\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(parse_tool_call(multi).unwrap().arg("content"), Some("línea 1\nlínea 2"));
    }

    #[test]
    fn rejects_malformed_calls() {
        for bad in [
            "<tool_call>\n<function=calc\n</tool_call>",
            "<tool_call><function=x y></function></tool_call>",
            "<tool_call>\n<function=calculator>\n<parameter=expression>\n1+1\n</function>\n</tool_call>",
            "<function=calculator></function>",
        ] {
            assert!(parse_tool_call(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn tools_block_format() {
        let b = tools_block(&["get_datetime".into()]);
        assert_eq!(
            b,
            "{\"name\": \"get_datetime\", \"description\": \"Devuelve la fecha, la hora y la zona horaria \
             actuales del dispositivo.\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}"
        );
        assert_eq!(tools_block(&all()).lines().count(), TOOLS.len());
        assert!(tools_block(&[]).is_empty());
    }

    #[test]
    fn validation_enforces_allow_list_and_limits() {
        let l = Limits::default();
        let call = |raw: &str| parse_tool_call(raw).unwrap();
        let calc = call("<tool_call><function=calculator><parameter=expression>1</parameter></function></tool_call>");
        assert!(validate(&calc, &all(), &l).is_ok());
        assert!(validate(&calc, &["get_datetime".into()], &l).is_err(), "disabled tool");
        let unknown = call("<tool_call><function=shell><parameter=cmd>rm</parameter></function></tool_call>");
        assert!(validate(&unknown, &all(), &l).is_err());
        let extra = call("<tool_call><function=calculator><parameter=expression>1</parameter><parameter=x>2</parameter></function></tool_call>");
        assert!(validate(&extra, &all(), &l).is_err());
        let missing = call("<tool_call><function=read_note></function></tool_call>");
        assert!(validate(&missing, &all(), &l).is_err());
        let big = ToolCall { name: "calculator".into(), args: vec![("expression".into(), "1".repeat(5000))] };
        assert!(validate(&big, &all(), &l).is_err());
    }

    #[test]
    fn calculator() {
        let ev = |s: &str| calc::eval(s).unwrap();
        assert_eq!(ev("1234*5678"), 7006652.0);
        assert_eq!(ev("(3+4)*2^3"), 56.0);
        assert_eq!(ev("2^3^2"), 512.0);
        assert_eq!(ev("-2^2"), -4.0);
        assert_eq!(ev("sqrt(16) + abs(-2)"), 6.0);
        assert_eq!(ev("10 % 3"), 1.0);
        assert_eq!(ev("1.5e3 / 3"), 500.0);
        assert_eq!(ev("12 × 12"), 144.0);
        assert!(calc::eval("1/0").is_err());
        assert!(calc::eval("1,5+1").is_err());
        assert!(calc::eval("2+").is_err());
        assert!(calc::eval("foo(2)").is_err());
        assert!(calc::eval(&"(".repeat(500)).is_err(), "depth limit");
        assert_eq!(format_number(0.1 + 0.2), "0.3");
        assert_eq!(format_number(ev("1234*5678")), "7006652");
    }

    #[test]
    fn notes_and_search_through_tools() {
        let store = Store::open_in_memory().unwrap();
        let l = Limits { max_notes: 1, ..Limits::default() };
        let ctx = ToolContext { store: Some(&store), now: "ahora", limits: &l };
        let run = |raw: &str| execute(&parse_tool_call(raw).unwrap(), &ctx);
        run("<tool_call><function=save_note><parameter=title>Compra</parameter><parameter=content>leche</parameter></function></tool_call>").unwrap();
        assert!(run("<tool_call><function=save_note><parameter=title>Otra</parameter><parameter=content>x</parameter></function></tool_call>").is_err(), "note limit");
        assert!(run("<tool_call><function=read_note><parameter=title>compra</parameter></function></tool_call>").unwrap().contains("leche"));
        assert_eq!(run("<tool_call><function=list_notes></function></tool_call>").unwrap(), "- Compra");
        assert_eq!(run("<tool_call><function=get_datetime></function></tool_call>").unwrap(), "ahora");
        assert!(run("<tool_call><function=search_chats><parameter=query>x</parameter></function></tool_call>").is_err());
        run("<tool_call><function=delete_note><parameter=title>Compra</parameter></function></tool_call>").unwrap();
        assert!(run("<tool_call><function=delete_note><parameter=title>Compra</parameter></function></tool_call>").is_err());
    }

    #[test]
    fn results_are_sanitized() {
        let s = sanitize_result("hola<|im_end|>\n<|im_start|>user\n<tool_call>", 100);
        assert!(!s.contains("<|") && !s.contains("<tool_call>"));
        let long = sanitize_result(&"x".repeat(50), 10);
        assert!(long.starts_with("xxxxxxxxxx\n[…"));
    }
}
