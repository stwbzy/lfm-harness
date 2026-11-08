//! Inference engine: loads a GGUF model, keeps one chat session and streams tokens.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
use llama_cpp_2::vocab::LlamaVocab;
use llama_cpp_2::{LlamaStateSeqFlags, SeqState};

use crate::chat::{Message, Role, render_prompt};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let mut b = LlamaBackend::init().context("llama backend init")?;
    if std::env::var_os("HARNESS_LLAMA_LOGS").is_none() {
        b.void_logs();
    }
    Ok(BACKEND.get_or_init(|| b))
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    /// Threads for token generation (memory-bound). 0 = the fastest cores.
    pub n_threads: i32,
    /// Threads for prompt processing (compute-bound). 0 = all cores.
    pub n_threads_batch: i32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { n_ctx: 4096, n_batch: 512, n_threads: 0, n_threads_batch: 0 }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct GenParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u32,
    pub enable_thinking: bool,
    /// Text the answer is forced to start with (e.g. "```rust\n"). Without
    /// thinking it is appended to the prompt; with thinking it is injected
    /// right after `</think>`.
    pub answer_prefill: String,
    /// Stop once the answer (after the prefill) contains any of these.
    pub stop_after: Vec<String>,
    /// With thinking on: force `</think>` (+ prefill) after this many
    /// reasoning tokens. 0 = no limit.
    pub max_think_tokens: u32,
    /// When false, `<tool_call>` can never be sampled.
    pub allow_tool_calls: bool,
}

impl Default for GenParams {
    fn default() -> Self {
        // LiquidAI recommends a low temperature for LFM models.
        Self {
            max_tokens: 1024,
            temperature: 0.3,
            top_k: 50,
            top_p: 1.0,
            min_p: 0.15,
            repeat_penalty: 1.05,
            repeat_last_n: 64,
            seed: 0xC0FFEE,
            enable_thinking: false,
            answer_prefill: String::new(),
            stop_after: Vec::new(),
            max_think_tokens: 0,
            allow_tool_calls: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event<'a> {
    /// A chunk of generated text (always whole UTF-8 characters).
    Text(&'a str),
    /// The model closed its reasoning block (`</think>`).
    ThinkEnd,
    /// The model started writing a tool call (its markup is not streamed).
    ToolCallStarted,
}

#[derive(Debug, Clone, Default)]
pub struct GenOutput {
    pub stats: GenStats,
    /// Everything generated, special tokens included (`</think>`, tool markup).
    pub text: String,
    /// The complete `<tool_call>…</tool_call>` block, if generation stopped on one.
    pub tool_call: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// Prompt tokens served from the cache of the previous turn.
    pub reused_tokens: usize,
    pub prompt_ms: f64,
    pub gen_tokens: usize,
    pub gen_ms: f64,
    pub cancelled: bool,
}

impl GenStats {
    pub fn prompt_tps(&self) -> f64 {
        rate(self.prompt_tokens - self.reused_tokens, self.prompt_ms)
    }
    pub fn gen_tps(&self) -> f64 {
        rate(self.gen_tokens, self.gen_ms)
    }
}

fn rate(n: usize, ms: f64) -> f64 {
    if ms > 0.0 { n as f64 * 1000.0 / ms } else { 0.0 }
}

pub struct Engine {
    // Field order matters: `ctx` borrows `*model` and must drop first.
    ctx: LlamaContext<'static>,
    /// Tokens currently held in the context memory (sequence 0).
    cached: Vec<LlamaToken>,
    /// Recurrent/SWA state saved at the end of the last prompt, so the next turn
    /// can rewind there instead of re-evaluating everything (hybrid models
    /// cannot drop the tail of their conv state otherwise).
    checkpoint: Option<(usize, SeqState)>,
    /// Same, taken at the end of the system turn.
    sys_checkpoint: Option<(usize, SeqState)>,
    bos: String,
    im_end: Option<LlamaToken>,
    newline: Option<LlamaToken>,
    think_end: Option<LlamaToken>,
    tool_start: Option<LlamaToken>,
    tool_end: Option<LlamaToken>,
    model: Box<LlamaModel>,
    config: EngineConfig,
}

// The context is only touched through `&mut Engine`.
unsafe impl Send for Engine {}

impl Engine {
    pub fn load(path: impl AsRef<Path>, config: EngineConfig) -> Result<Self> {
        let backend = backend()?;
        let path = path.as_ref();
        if !path.is_file() {
            bail!("model not found: {}", path.display());
        }
        let model = Box::new(
            LlamaModel::load_from_file(backend, path, &LlamaModelParams::default())
                .with_context(|| format!("loading {}", path.display()))?,
        );

        let (auto_gen, auto_batch) = default_threads();
        let threads = if config.n_threads > 0 { config.n_threads } else { auto_gen };
        let threads_batch = if config.n_threads_batch > 0 { config.n_threads_batch } else { auto_batch };
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(config.n_ctx))
            .with_n_batch(config.n_batch)
            .with_n_threads(threads)
            .with_n_threads_batch(threads_batch);
        let ctx = model.new_context(backend, params).context("creating context")?;
        // SAFETY: `model` is boxed (stable address), never moved out or replaced,
        // and `ctx` is declared before it so it is dropped first.
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

        let vocab = model.vocab();
        let bos = String::from_utf8_lossy(&vocab.token_to_piece(vocab.bos(), true, None))
            .into_owned();
        let special = |text: &str| match vocab.tokenize(text.as_bytes(), false, true).as_slice() {
            [t] => Some(*t),
            _ => None,
        };
        let think_end = special("</think>");
        let tool_start = special("<tool_call>");
        let tool_end = special("</tool_call>");

        let im_end = special("<|im_end|>");
        let newline = special("\n");
        Ok(Self {
            ctx,
            cached: Vec::new(),
            checkpoint: None,
            sys_checkpoint: None,
            bos,
            im_end,
            newline,
            think_end,
            tool_start,
            tool_end,
            model,
            config,
        })
    }

    pub fn n_ctx(&self) -> u32 {
        self.ctx.n_ctx()
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.model.vocab().tokenize(text.as_bytes(), false, true).len()
    }

    /// Forgets the cached conversation (e.g. when switching chats).
    pub fn reset(&mut self) {
        self.ctx.clear_kv_cache();
        self.cached.clear();
        self.checkpoint = None;
        self.sys_checkpoint = None;
    }

    /// Generates the assistant reply for `messages`, streaming [`Event`]s.
    /// Stops on end-of-generation, `max_tokens`, a completed tool call, or when
    /// `cancel` becomes true. `tools` is the `<tools>` block body, if any.
    ///
    /// Everything streamed before an [`Event::ThinkEnd`] is reasoning. LFM3
    /// fine-tunes sometimes reason even with thinking disabled, so callers
    /// should handle `ThinkEnd` in both modes. Tool-call markup is never
    /// streamed as text; it is returned in [`GenOutput::tool_call`].
    pub fn generate(
        &mut self,
        messages: &[Message],
        tools: Option<&str>,
        params: &GenParams,
        cancel: &AtomicBool,
        mut on_event: impl FnMut(Event<'_>),
    ) -> Result<GenOutput> {
        let n_ctx = self.ctx.n_ctx() as usize;
        let tokens = self.fit_prompt(messages, tools, params, n_ctx)?;

        let mut stats = GenStats { prompt_tokens: tokens.len(), ..Default::default() };
        let t0 = Instant::now();
        let start = self.reuse_prefix(&tokens);
        stats.reused_tokens = start;

        // Prompt evaluation in n_batch chunks; logits only for the last token.
        // Chunks also break at the end of the system turn to checkpoint it.
        let sys_end = self.system_end(&tokens).filter(|&e| e > start && e < tokens.len());
        let n_batch = self.config.n_batch as usize;
        let mut batch = LlamaBatch::new(n_batch.max(1), 1);
        let mut pos = start;
        while pos < tokens.len() {
            if cancel.load(Ordering::Relaxed) {
                stats.cancelled = true;
                return Ok(GenOutput { stats, text: String::new(), tool_call: None });
            }
            batch.clear();
            let mut end = (pos + n_batch).min(tokens.len());
            if let Some(e) = sys_end.filter(|&e| e > pos && e < end) {
                end = e;
            }
            for (i, &tok) in tokens[pos..end].iter().enumerate() {
                let p = pos + i;
                batch.add(tok, p as i32, &[0], p == tokens.len() - 1)?;
            }
            self.ctx.decode(&mut batch).context("prompt decode")?;
            self.cached.extend_from_slice(&tokens[pos..end]);
            pos = end;
            if Some(end) == sys_end {
                self.sys_checkpoint = self.snapshot();
            }
        }
        stats.prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // The rendered prompt is always a prefix of the next turn's prompt.
        self.checkpoint = self.snapshot();

        let mut sampler = self.sampler(params);
        sampler.accept_many(&tokens);
        let vocab = self.model.vocab();
        let mut utf8 = Utf8Stream::default();
        let mut logits_idx = batch.n_tokens() - 1;
        let max_new = (params.max_tokens as usize).min(n_ctx - tokens.len() - 1);
        let t1 = Instant::now();
        // Every generated piece, special tokens included (for the raw history).
        let mut raw: Vec<u8> = Vec::new();
        // Some(bytes) while inside <tool_call>…</tool_call>.
        let mut tool_buf: Option<Vec<u8>> = None;
        let mut tool_call = None;
        // Answer phase: immediately without thinking (prefill is in the prompt).
        let mut in_answer = !params.enable_thinking;
        let mut think_tokens = 0u32;
        // Answer text after the prefill, for `stop_after`.
        let mut answer: Vec<u8> = Vec::new();
        if in_answer && !params.answer_prefill.is_empty() {
            raw.extend_from_slice(params.answer_prefill.as_bytes());
            on_event(Event::Text(&params.answer_prefill));
        }

        while stats.gen_tokens < max_new {
            if cancel.load(Ordering::Relaxed) {
                stats.cancelled = true;
                break;
            }
            let token = sampler.sample(&self.ctx, logits_idx);
            sampler.accept(token);
            if vocab.is_eog(token) {
                break;
            }
            stats.gen_tokens += 1;
            let piece = vocab.token_to_piece(token, true, None);
            raw.extend_from_slice(&piece);

            if Some(token) == self.tool_start && tool_buf.is_none() {
                if let Some(text) = utf8.flush() {
                    on_event(Event::Text(&text));
                }
                on_event(Event::ToolCallStarted);
                tool_buf = Some(piece.clone());
            } else if let Some(buf) = tool_buf.as_mut() {
                buf.extend_from_slice(&piece);
                if Some(token) == self.tool_end {
                    tool_call = Some(String::from_utf8_lossy(buf).into_owned());
                }
            } else if Some(token) == self.think_end {
                if let Some(text) = utf8.flush() {
                    on_event(Event::Text(&text));
                }
                on_event(Event::ThinkEnd);
            } else if let Some(text) = utf8.push(&piece) {
                on_event(Event::Text(&text));
            }

            batch.clear();
            batch.add(token, self.cached.len() as i32, &[0], true)?;
            self.ctx.decode(&mut batch).context("token decode")?;
            self.cached.push(token);
            logits_idx = 0;
            if tool_call.is_some() {
                // One call per round: the harness runs it and continues.
                break;
            }

            if !in_answer {
                think_tokens += 1;
                let natural_end = Some(token) == self.think_end;
                let over_budget = params.max_think_tokens > 0 && think_tokens >= params.max_think_tokens;
                if natural_end || over_budget {
                    in_answer = true;
                    let mut forced = String::new();
                    if !natural_end {
                        // Budget exhausted: close the reasoning ourselves.
                        if let Some(text) = utf8.flush() {
                            on_event(Event::Text(&text));
                        }
                        on_event(Event::ThinkEnd);
                        forced.push_str("</think>");
                    }
                    if !params.answer_prefill.is_empty() {
                        on_event(Event::Text(&params.answer_prefill));
                        forced.push_str(&params.answer_prefill);
                    }
                    if !forced.is_empty() {
                        raw.extend_from_slice(forced.as_bytes());
                        logits_idx = inject(&mut self.ctx, &mut self.cached, &vocab, &forced, &mut batch, &mut sampler)?;
                    }
                }
            } else if !params.stop_after.is_empty() {
                answer.extend_from_slice(&piece);
                let tail = &answer[answer.len().saturating_sub(24)..];
                if params.stop_after.iter().any(|s| contains_bytes(tail, s.as_bytes())) {
                    break;
                }
            }
        }
        if let Some(text) = utf8.flush() {
            on_event(Event::Text(&text));
        }
        stats.gen_ms = t1.elapsed().as_secs_f64() * 1000.0;
        Ok(GenOutput { stats, text: String::from_utf8_lossy(&raw).into_owned(), tool_call })
    }

    /// Tokenizes the conversation, dropping the oldest messages (keeping the
    /// system prompt) until it fits with room for a reply (a quarter of the
    /// context or `max_tokens`). The window always starts at a user turn.
    fn fit_prompt(
        &self,
        messages: &[Message],
        tools: Option<&str>,
        params: &GenParams,
        n_ctx: usize,
    ) -> Result<Vec<LlamaToken>> {
        let reserve = (params.max_tokens as usize).min(n_ctx / 4).max(1);
        let (system, rest) = match messages.split_first() {
            Some((first, rest)) if first.role == Role::System => (Some(first), rest),
            _ => (None, messages),
        };
        let vocab = self.model.vocab();
        let mut buf: Vec<Message> = Vec::with_capacity(messages.len());
        for skip in 0..rest.len().max(1) {
            buf.clear();
            buf.extend(system.cloned());
            // Start on a user turn so assistant/tool turns keep their context.
            let window = &rest[skip..];
            let window = match window.iter().position(|m| m.role == Role::User) {
                Some(i) => &window[i..],
                None => continue,
            };
            buf.extend_from_slice(window);
            let mut prompt = render_prompt(&self.bos, &buf, tools, params.enable_thinking);
            if !params.enable_thinking {
                prompt.push_str(&params.answer_prefill);
            }
            let tokens = vocab.tokenize(prompt.as_bytes(), false, true);
            if tokens.len() + reserve < n_ctx {
                return Ok(tokens);
            }
        }
        bail!("the last message does not fit in the context ({n_ctx} tokens)")
    }

    /// Keeps the longest reusable cached prefix of `tokens` and returns its
    /// length. At least one token is always left to evaluate so logits exist.
    fn reuse_prefix(&mut self, tokens: &[LlamaToken]) -> usize {
        let common = self.cached.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        let keep = common.min(tokens.len().saturating_sub(1));
        if keep == self.cached.len() {
            return keep;
        }
        // Pure-attention models can truncate directly. On hybrid ones (LFM3) the
        // recurrent part refuses and nothing is modified.
        if keep > 0 && self.ctx.kv_cache_seq_rm(0, Some(keep as u32), None).is_ok() {
            self.truncate_cached(keep);
            return keep;
        }
        // Rewind to the furthest checkpoint inside the shared prefix: the end of
        // the previous prompt, or else the end of the system turn (shared by
        // every chat with the same system prompt and tools).
        let best = [&self.checkpoint, &self.sys_checkpoint]
            .into_iter()
            .flatten()
            .filter(|(n, _)| *n > 0 && *n <= keep)
            .max_by_key(|(n, _)| *n)
            .map(|(n, s)| (*n, s.clone()));
        if let Some((n, state)) = best
            && self.ctx.state_seq_set(&state, 0).is_ok()
            && self.ctx.kv_cache_seq_rm(0, Some(n as u32), None).is_ok()
        {
            self.truncate_cached(n);
            return n;
        }
        self.reset();
        0
    }

    /// Shortens the cached tokens, dropping checkpoints that lie beyond them.
    fn truncate_cached(&mut self, n: usize) {
        self.cached.truncate(n);
        for ck in [&mut self.checkpoint, &mut self.sys_checkpoint] {
            if ck.as_ref().is_some_and(|(m, _)| *m > n) {
                *ck = None;
            }
        }
    }

    fn snapshot(&self) -> Option<(usize, SeqState)> {
        self.ctx
            .state_seq_get(0, LlamaStateSeqFlags::PARTIAL_ONLY)
            .ok()
            .map(|s| (self.cached.len(), s))
    }

    /// Token index just after the system turn (`…<|im_end|>\n`), if any.
    fn system_end(&self, tokens: &[LlamaToken]) -> Option<usize> {
        let (im_end, nl) = (self.im_end?, self.newline?);
        // tokens[0] is BOS, then `<|im_start|>system`.
        let i = tokens.iter().position(|&t| t == im_end)?;
        (tokens.get(i + 1) == Some(&nl)).then_some(i + 2)
    }

    fn sampler(&self, p: &GenParams) -> LlamaSampler {
        let mut chain = Vec::new();
        if !p.allow_tool_calls
            && let Some(t) = self.tool_start
        {
            chain.push(LlamaSampler::logit_bias(self.model.n_vocab(), &[LlamaLogitBias::new(t, f32::NEG_INFINITY)]));
        }
        chain.extend([
            LlamaSampler::penalties(self.model.n_vocab(), p.repeat_last_n, p.repeat_penalty, 0.0, 0.0),
            LlamaSampler::top_k(p.top_k),
            LlamaSampler::top_p(p.top_p, 1),
            LlamaSampler::min_p(p.min_p, 1),
        ]);
        if p.temperature <= 0.0 {
            chain.push(LlamaSampler::greedy());
        } else {
            chain.push(LlamaSampler::temp(p.temperature));
            chain.push(LlamaSampler::dist(p.seed));
        }
        LlamaSampler::chain_simple(chain)
    }
}

/// Feeds forced text (closing `</think>`, answer prefill) into the context as
/// if the model had generated it. Returns the batch index holding the logits.
fn inject(
    ctx: &mut LlamaContext<'static>,
    cached: &mut Vec<LlamaToken>,
    vocab: &LlamaVocab<'_>,
    text: &str,
    batch: &mut LlamaBatch,
    sampler: &mut LlamaSampler,
) -> Result<i32> {
    let toks = vocab.tokenize(text.as_bytes(), false, true);
    batch.clear();
    for (i, &t) in toks.iter().enumerate() {
        batch.add(t, (cached.len() + i) as i32, &[0], i + 1 == toks.len())?;
    }
    ctx.decode(batch).context("inject decode")?;
    cached.extend_from_slice(&toks);
    sampler.accept_many(&toks);
    Ok(batch.n_tokens() - 1)
}

fn contains_bytes(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

/// (generation threads, prompt threads). On big.LITTLE phones generation is
/// fastest on the big cores only (little cores stall the memory-bound
/// threads), while prompt processing benefits from every core.
fn default_threads() -> (i32, i32) {
    let n = std::thread::available_parallelism().map_or(4, |n| n.get()) as i32;
    if !cfg!(target_os = "android") {
        return (n.min(8), n.min(8));
    }
    let freqs: Vec<u64> = (0..n)
        .filter_map(|i| {
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq"))
                .ok()?
                .trim()
                .parse()
                .ok()
        })
        .collect();
    // Cores faster than the slowest cluster count as "big" (max clocks alone can
    // be close: Helio G99 has A55 @ 2.0 GHz and A76 @ 2.2 GHz).
    let big = match (freqs.iter().min(), freqs.len() as i32 == n) {
        (Some(&min), true) => freqs.iter().filter(|&&f| f > min).count() as i32,
        _ => 0,
    };
    let gen_threads = if big >= 2 { big.min(4) } else { (n / 2).clamp(2, 4) };
    (gen_threads, n.min(8))
}

/// Re-assembles UTF-8 characters split across token boundaries.
#[derive(Default)]
struct Utf8Stream {
    buf: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> Option<String> {
        self.buf.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.buf) {
            Ok(_) => self.buf.len(),
            // An invalid sequence that is not just incomplete: emit lossily.
            Err(e) if e.error_len().is_some() => return self.flush(),
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return None;
        }
        let rest = self.buf.split_off(valid);
        let out = String::from_utf8(std::mem::replace(&mut self.buf, rest)).ok();
        out
    }

    fn flush(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            return None;
        }
        Some(String::from_utf8_lossy(&std::mem::take(&mut self.buf)).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8Stream;

    #[test]
    fn utf8_split_across_pieces() {
        let bytes = "¿qué?".as_bytes();
        let mut s = Utf8Stream::default();
        let mut out = String::new();
        for b in bytes {
            if let Some(t) = s.push(&[*b]) {
                out.push_str(&t);
            }
        }
        assert_eq!(out, "¿qué?");
        assert!(s.flush().is_none());
    }
}
