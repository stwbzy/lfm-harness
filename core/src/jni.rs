//! JNI bridge for `dev.stwby.lfmharness.engine.NativeEngine`.
//!
//! A handle (`jlong`) points to a boxed [`Session`]. `generate` blocks the
//! calling thread (Kotlin runs it on an IO dispatcher) and reports tokens
//! through a callback object; `cancel` may be called from any thread.
//! Messages, parameters and stats cross the boundary as JSON.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jint, jlong, jstring};
use serde::{Deserialize, Serialize};

use crate::{Engine, EngineConfig, Event, GenParams, Message, Role};

struct Session {
    engine: Mutex<Engine>,
    cancel: AtomicBool,
}

#[derive(Deserialize)]
struct JsonMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct JsonStats {
    prompt_tokens: usize,
    reused_tokens: usize,
    prompt_tps: f64,
    gen_tokens: usize,
    gen_tps: f64,
    cancelled: bool,
}

fn throw(env: &mut JNIEnv, err: &anyhow::Error) {
    if !env.exception_check().unwrap_or(false) {
        let _ = env.throw_new("java/lang/RuntimeException", format!("{err:#}"));
    }
}

/// SAFETY: `handle` must come from `load` and not have been freed.
unsafe fn session<'a>(handle: jlong) -> Result<&'a Session> {
    unsafe { (handle as *const Session).as_ref() }.ok_or_else(|| anyhow!("engine not loaded"))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_engine_NativeEngine_load<'l>(
    mut env: JNIEnv<'l>,
    _cls: JClass<'l>,
    path: JString<'l>,
    n_ctx: jint,
    n_threads: jint,
) -> jlong {
    let result = (|| -> Result<jlong> {
        let path: String = env.get_string(&path)?.into();
        let config = EngineConfig {
            n_ctx: n_ctx.max(512) as u32,
            n_threads,
            ..EngineConfig::default()
        };
        let engine = Engine::load(path, config)?;
        let session = Box::new(Session { engine: Mutex::new(engine), cancel: AtomicBool::new(false) });
        Ok(Box::into_raw(session) as jlong)
    })();
    result.unwrap_or_else(|e| {
        throw(&mut env, &e);
        0
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_engine_NativeEngine_free<'l>(
    _env: JNIEnv<'l>,
    _cls: JClass<'l>,
    handle: jlong,
) {
    if handle != 0 {
        // SAFETY: handle came from `load`; Kotlin guarantees a single free with
        // no generation in flight.
        drop(unsafe { Box::from_raw(handle as *mut Session) });
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_engine_NativeEngine_cancel<'l>(
    _env: JNIEnv<'l>,
    _cls: JClass<'l>,
    handle: jlong,
) {
    if let Ok(s) = unsafe { session(handle) } {
        s.cancel.store(true, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_engine_NativeEngine_reset<'l>(
    mut env: JNIEnv<'l>,
    _cls: JClass<'l>,
    handle: jlong,
) {
    let result = (|| -> Result<()> {
        let s = unsafe { session(handle)? };
        s.engine.lock().map_err(|_| anyhow!("engine poisoned"))?.reset();
        Ok(())
    })();
    if let Err(e) = result {
        throw(&mut env, &e);
    }
}

/// `callback` must implement `onText(String)` and `onThinkEnd()`.
/// Returns stats as JSON.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_engine_NativeEngine_generate<'l>(
    mut env: JNIEnv<'l>,
    _cls: JClass<'l>,
    handle: jlong,
    messages_json: JString<'l>,
    params_json: JString<'l>,
    callback: JObject<'l>,
) -> jstring {
    let result = (|| -> Result<jstring> {
        let s = unsafe { session(handle)? };
        let messages: String = env.get_string(&messages_json)?.into();
        let params: String = env.get_string(&params_json)?.into();
        let messages = parse_messages(&messages)?;
        let params: GenParams = serde_json::from_str(&params).context("invalid params")?;

        s.cancel.store(false, Ordering::Relaxed);
        let mut engine = s.engine.lock().map_err(|_| anyhow!("engine poisoned"))?;
        let mut cb_error: Option<anyhow::Error> = None;
        let stats = engine.generate(&messages, &params, &s.cancel, |ev| {
            if cb_error.is_some() {
                return;
            }
            let r = env.with_local_frame(4, |env| -> Result<()> {
                match ev {
                    Event::Text(t) => {
                        let jt = env.new_string(t)?;
                        env.call_method(&callback, "onText", "(Ljava/lang/String;)V", &[JValue::Object(&jt)])?;
                    }
                    Event::ThinkEnd => {
                        env.call_method(&callback, "onThinkEnd", "()V", &[])?;
                    }
                }
                Ok(())
            });
            if let Err(e) = r {
                // A Java exception in the callback aborts generation.
                s.cancel.store(true, Ordering::Relaxed);
                cb_error = Some(e);
            }
        })?;
        if let Some(e) = cb_error {
            return Err(e);
        }

        let json = serde_json::to_string(&JsonStats {
            prompt_tokens: stats.prompt_tokens,
            reused_tokens: stats.reused_tokens,
            prompt_tps: stats.prompt_tps(),
            gen_tokens: stats.gen_tokens,
            gen_tps: stats.gen_tps(),
            cancelled: stats.cancelled,
        })?;
        Ok(env.new_string(json)?.into_raw())
    })();
    result.unwrap_or_else(|e| {
        throw(&mut env, &e);
        std::ptr::null_mut()
    })
}

fn parse_messages(json: &str) -> Result<Vec<Message>> {
    let raw: Vec<JsonMessage> = serde_json::from_str(json).context("invalid messages")?;
    raw.into_iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => return Err(anyhow!("unknown role {other}")),
            };
            Ok(Message { role, content: m.content })
        })
        .collect()
}
