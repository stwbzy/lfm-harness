//! Markdown → a small block tree the UI renders natively.
//!
//! Parsing uses `pulldown-cmark` (CommonMark + tables + strikethrough). The
//! output is deliberately simple so the Compose side stays a thin painter:
//! blocks (paragraph, heading, code, list, quote, rule, table) holding inline
//! spans with style flags. Unclosed markup while streaming is shown literally,
//! which is what CommonMark does.

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use serde::Serialize;

pub const BOLD: u8 = 1;
pub const ITALIC: u8 = 2;
pub const CODE: u8 = 4;
pub const STRIKE: u8 = 8;
pub const LINK: u8 = 16;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Span {
    #[serde(rename = "x")]
    pub text: String,
    #[serde(rename = "f", skip_serializing_if = "is_zero")]
    pub flags: u8,
    #[serde(rename = "u", skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

fn is_zero(f: &u8) -> bool {
    *f == 0
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "t")]
pub enum Block {
    #[serde(rename = "p")]
    Paragraph { s: Vec<Span> },
    #[serde(rename = "h")]
    Heading { l: u8, s: Vec<Span> },
    #[serde(rename = "code")]
    Code { lang: String, text: String },
    #[serde(rename = "list")]
    List {
        /// First number for ordered lists, None for bullets.
        start: Option<u64>,
        items: Vec<Vec<Block>>,
    },
    #[serde(rename = "quote")]
    Quote { b: Vec<Block> },
    #[serde(rename = "hr")]
    Rule,
    #[serde(rename = "table")]
    Table { head: Vec<Vec<Span>>, rows: Vec<Vec<Vec<Span>>> },
}

enum InlineKind {
    Paragraph,
    Heading(u8),
    Cell,
}

enum Frame {
    Blocks(Vec<Block>),
    Quote(Vec<Block>),
    List { start: Option<u64>, items: Vec<Vec<Block>> },
    Item(Vec<Block>),
    /// `auto`: opened implicitly for text outside a paragraph (tight lists).
    Inline { kind: InlineKind, spans: Vec<Span>, auto: bool },
    Code { lang: String, text: String },
    Table { head: Vec<Vec<Span>>, rows: Vec<Vec<Vec<Span>>>, in_head: bool },
}

struct Builder {
    stack: Vec<Frame>,
    bold: u32,
    italic: u32,
    strike: u32,
    links: Vec<String>,
}

pub fn parse(md: &str) -> Vec<Block> {
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let mut b = Builder { stack: vec![Frame::Blocks(Vec::new())], bold: 0, italic: 0, strike: 0, links: Vec::new() };
    for ev in Parser::new_ext(md, opts) {
        b.event(ev);
    }
    // Close anything left open (defensive; the parser balances tags).
    while b.stack.len() > 1 {
        b.pop();
    }
    match b.stack.pop() {
        Some(Frame::Blocks(v)) => v,
        _ => Vec::new(),
    }
}

pub fn to_json(md: &str) -> String {
    serde_json::to_string(&parse(md)).unwrap_or_else(|_| "[]".into())
}

impl Builder {
    fn event(&mut self, ev: Event) {
        match ev {
            Event::Start(tag) => self.start(tag),
            Event::End(end) => self.end(end),
            Event::Text(t) => self.text(&t, 0),
            Event::Code(t) => self.text(&t, CODE),
            Event::InlineMath(t) | Event::DisplayMath(t) => self.text(&t, CODE),
            Event::Html(t) | Event::InlineHtml(t) => self.text(&t, 0),
            Event::SoftBreak => self.text(" ", 0),
            Event::HardBreak => self.text("\n", 0),
            Event::Rule => {
                self.close_auto();
                self.push_block(Block::Rule);
            }
            Event::FootnoteReference(t) => self.text(&format!("[{t}]"), 0),
            Event::TaskListMarker(done) => self.text(if done { "☑ " } else { "☐ " }, 0),
        }
    }

    fn start(&mut self, tag: Tag) {
        match tag {
            Tag::Emphasis => self.italic += 1,
            Tag::Strong => self.bold += 1,
            Tag::Strikethrough => self.strike += 1,
            Tag::Link { dest_url, .. } => self.links.push(dest_url.to_string()),
            Tag::Image { dest_url, .. } => {
                // No remote loading: show the image as a link.
                self.links.push(dest_url.to_string());
                self.text("🖼 ", 0);
            }
            Tag::Paragraph => {
                self.close_auto();
                self.stack.push(Frame::Inline { kind: InlineKind::Paragraph, spans: Vec::new(), auto: false });
            }
            Tag::Heading { level, .. } => {
                self.close_auto();
                let l = match level {
                    HeadingLevel::H1 => 1,
                    HeadingLevel::H2 => 2,
                    HeadingLevel::H3 => 3,
                    HeadingLevel::H4 => 4,
                    HeadingLevel::H5 => 5,
                    HeadingLevel::H6 => 6,
                };
                self.stack.push(Frame::Inline { kind: InlineKind::Heading(l), spans: Vec::new(), auto: false });
            }
            Tag::BlockQuote(_) => {
                self.close_auto();
                self.stack.push(Frame::Quote(Vec::new()));
            }
            Tag::CodeBlock(kind) => {
                self.close_auto();
                let lang = match kind {
                    CodeBlockKind::Fenced(l) => l.split_whitespace().next().unwrap_or("").to_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                self.stack.push(Frame::Code { lang, text: String::new() });
            }
            Tag::List(start) => {
                self.close_auto();
                self.stack.push(Frame::List { start, items: Vec::new() });
            }
            Tag::Item => {
                self.close_auto();
                self.stack.push(Frame::Item(Vec::new()));
            }
            Tag::Table(_) => {
                self.close_auto();
                self.stack.push(Frame::Table { head: Vec::new(), rows: Vec::new(), in_head: false });
            }
            Tag::TableHead => {
                if let Some(Frame::Table { in_head, .. }) = self.stack.last_mut() {
                    *in_head = true;
                }
            }
            Tag::TableRow => {
                if let Some(Frame::Table { rows, .. }) = self.stack.last_mut() {
                    rows.push(Vec::new());
                }
            }
            Tag::TableCell => {
                self.stack.push(Frame::Inline { kind: InlineKind::Cell, spans: Vec::new(), auto: false });
            }
            // Containers we flatten into their parent.
            _ => {}
        }
    }

    fn end(&mut self, end: TagEnd) {
        match end {
            TagEnd::Emphasis => self.italic = self.italic.saturating_sub(1),
            TagEnd::Strong => self.bold = self.bold.saturating_sub(1),
            TagEnd::Strikethrough => self.strike = self.strike.saturating_sub(1),
            TagEnd::Link | TagEnd::Image => {
                self.links.pop();
            }
            TagEnd::Paragraph
            | TagEnd::Heading(_)
            | TagEnd::CodeBlock
            | TagEnd::TableCell
            | TagEnd::BlockQuote(_)
            | TagEnd::List(_)
            | TagEnd::Item
            | TagEnd::Table => {
                self.close_auto();
                self.pop();
            }
            TagEnd::TableHead => {
                if let Some(Frame::Table { in_head, .. }) = self.stack.last_mut() {
                    *in_head = false;
                }
            }
            _ => {}
        }
    }

    fn text(&mut self, t: &str, extra: u8) {
        if let Some(Frame::Code { text, .. }) = self.stack.last_mut() {
            text.push_str(t);
            return;
        }
        if !matches!(self.stack.last(), Some(Frame::Inline { .. })) {
            self.stack.push(Frame::Inline { kind: InlineKind::Paragraph, spans: Vec::new(), auto: true });
        }
        let mut flags = extra;
        if self.bold > 0 {
            flags |= BOLD;
        }
        if self.italic > 0 {
            flags |= ITALIC;
        }
        if self.strike > 0 {
            flags |= STRIKE;
        }
        let url = self.links.last().cloned();
        if url.is_some() {
            flags |= LINK;
        }
        let Some(Frame::Inline { spans, .. }) = self.stack.last_mut() else { return };
        match spans.last_mut() {
            Some(last) if last.flags == flags && last.url == url => last.text.push_str(t),
            _ => spans.push(Span { text: t.to_string(), flags, url }),
        }
    }

    fn close_auto(&mut self) {
        if matches!(self.stack.last(), Some(Frame::Inline { auto: true, .. })) {
            self.pop();
        }
    }

    fn pop(&mut self) {
        let Some(frame) = self.stack.pop() else { return };
        match frame {
            Frame::Inline { kind: InlineKind::Cell, spans, .. } => {
                if let Some(Frame::Table { head, rows, in_head }) = self.stack.last_mut() {
                    if *in_head {
                        head.push(spans);
                    } else if let Some(row) = rows.last_mut() {
                        row.push(spans);
                    }
                }
            }
            Frame::Inline { kind, mut spans, .. } => {
                trim_spans(&mut spans);
                if spans.is_empty() {
                    return;
                }
                self.push_block(match kind {
                    InlineKind::Heading(l) => Block::Heading { l, s: spans },
                    _ => Block::Paragraph { s: spans },
                });
            }
            Frame::Code { lang, mut text } => {
                if text.ends_with('\n') {
                    text.pop();
                }
                self.push_block(Block::Code { lang, text });
            }
            Frame::Quote(b) => self.push_block(Block::Quote { b }),
            Frame::List { start, items } => self.push_block(Block::List { start, items }),
            Frame::Item(blocks) => {
                if let Some(Frame::List { items, .. }) = self.stack.last_mut() {
                    items.push(blocks);
                } else {
                    for b in blocks {
                        self.push_block(b);
                    }
                }
            }
            Frame::Table { head, rows, .. } => self.push_block(Block::Table { head, rows }),
            Frame::Blocks(v) => {
                for b in v {
                    self.push_block(b);
                }
            }
        }
    }

    /// Appends a block to the nearest block container.
    fn push_block(&mut self, block: Block) {
        for frame in self.stack.iter_mut().rev() {
            match frame {
                Frame::Blocks(v) | Frame::Quote(v) | Frame::Item(v) => {
                    v.push(block);
                    return;
                }
                _ => {}
            }
        }
    }
}

fn trim_spans(spans: &mut Vec<Span>) {
    if let Some(first) = spans.first_mut() {
        first.text = first.text.trim_start().to_string();
    }
    if let Some(last) = spans.last_mut() {
        last.text = last.text.trim_end().to_string();
    }
    spans.retain(|s| !s.text.is_empty());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, flags: u8) -> Span {
        Span { text: text.into(), flags, url: None }
    }

    #[test]
    fn inline_styles() {
        let b = parse("1234 × 5678 = **7,006,652** y *más* `x` ~~no~~");
        assert_eq!(
            b,
            vec![Block::Paragraph {
                s: vec![
                    span("1234 × 5678 = ", 0),
                    span("7,006,652", BOLD),
                    span(" y ", 0),
                    span("más", ITALIC),
                    span(" ", 0),
                    span("x", CODE),
                    span(" ", 0),
                    span("no", STRIKE),
                ]
            }]
        );
    }

    #[test]
    fn tight_and_nested_lists() {
        let b = parse("- uno\n- **dos**\n  1. a\n  2. b\n");
        let Block::List { start: None, items } = &b[0] else { panic!("{b:?}") };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], vec![Block::Paragraph { s: vec![span("uno", 0)] }]);
        assert_eq!(items[1][0], Block::Paragraph { s: vec![span("dos", BOLD)] });
        let Block::List { start: Some(1), items: inner } = &items[1][1] else { panic!("{:?}", items[1]) };
        assert_eq!(inner.len(), 2);
    }

    #[test]
    fn code_heading_quote_rule_table() {
        let md = "# Título\n\n```rust\nfn main() {}\n```\n\n> cita\n\n---\n\n| a | b |\n|---|---|\n| 1 | **2** |\n";
        let b = parse(md);
        assert_eq!(b[0], Block::Heading { l: 1, s: vec![span("Título", 0)] });
        assert_eq!(b[1], Block::Code { lang: "rust".into(), text: "fn main() {}".into() });
        assert_eq!(b[2], Block::Quote { b: vec![Block::Paragraph { s: vec![span("cita", 0)] }] });
        assert_eq!(b[3], Block::Rule);
        let Block::Table { head, rows } = &b[4] else { panic!() };
        assert_eq!(head.len(), 2);
        assert_eq!(rows[0][1], vec![span("2", BOLD)]);
    }

    #[test]
    fn links_and_unclosed_streaming() {
        let b = parse("[web](https://x.org) y **sin cerrar");
        let Block::Paragraph { s } = &b[0] else { panic!() };
        assert_eq!(s[0], Span { text: "web".into(), flags: LINK, url: Some("https://x.org".into()) });
        assert!(s.iter().any(|x| x.text.contains("**sin cerrar")));
    }

    #[test]
    fn json_is_compact() {
        assert_eq!(to_json("**a**"), r#"[{"t":"p","s":[{"x":"a","f":1}]}]"#);
        assert_eq!(to_json(""), "[]");
    }
}
