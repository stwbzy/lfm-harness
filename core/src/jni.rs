//! JNI bridge for `dev.stwby.lfmharness.engine.NativeEngine`.
//!
//! A handle (`jlong`) points to a boxed [`Session`]. `generate` blocks the
//! calling thread (Kotlin runs it on an IO dispatcher) and reports tokens
//! through a callback object; `cancel` may be called from any thread.
//! Messages, parameters and stats cross the boundary as JSON.

use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jint, jlong, jstring};
use serde::Deserialize;

use crate::harness::{TurnEvent, TurnRequest, run_turn};
use crate::store::{NewMessage, Store, title_from};
use crate::tools::Limits;
use crate::{Engine, EngineConfig, GenParams, Message, Role};

struct Session {
    engine: Mutex<Engine>,
    cancel: AtomicBool,
}

#[derive(Deserialize)]
struct JsonMessage {
    role: String,
    content: String,
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

#[derive(Deserialize)]
struct TurnParams {
    #[serde(flatten)]
    generation: GenParams,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    now: String,
}

/// Runs one user turn (with tool loop). `store` is a `NativeStore` handle or 0.
///
/// `callback` implements `TurnCallback`: `onText(String)`, `onThinkEnd()`,
/// `onToolCallStarted()`, `onToolCall(String name, String argsJson)`,
/// `onToolResult(String name, boolean ok, String content)`, `onNewMessage()`
/// and `boolean confirmTool(String name, String argsJson)` (may block).
/// Returns the turn's messages and stats as JSON.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_engine_NativeEngine_generate<'l>(
    mut env: JNIEnv<'l>,
    _cls: JClass<'l>,
    handle: jlong,
    store: jlong,
    messages_json: JString<'l>,
    params_json: JString<'l>,
    callback: JObject<'l>,
) -> jstring {
    let result = (|| -> Result<jstring> {
        let s = unsafe { session(handle)? };
        let store = unsafe { (store as *const Mutex<Store>).as_ref() };
        let messages: String = env.get_string(&messages_json)?.into();
        let params: String = env.get_string(&params_json)?.into();
        let messages = parse_messages(&messages)?;
        let params: TurnParams = serde_json::from_str(&params).context("invalid params")?;
        let limits = Limits::default();
        let req = TurnRequest {
            messages: &messages,
            params: &params.generation,
            enabled_tools: &params.tools,
            now: &params.now,
            limits: &limits,
        };

        s.cancel.store(false, Ordering::Relaxed);
        let mut engine = s.engine.lock().map_err(|_| anyhow!("engine poisoned"))?;
        // Both callbacks need the env; they never run at the same time.
        let env_cell = RefCell::new(&mut env);
        let cb_error: RefCell<Option<anyhow::Error>> = RefCell::new(None);
        let fail = |e: anyhow::Error| {
            // A Java exception in a callback aborts generation.
            s.cancel.store(true, Ordering::Relaxed);
            cb_error.borrow_mut().get_or_insert(e);
        };

        let output = run_turn(
            &mut engine,
            store,
            &req,
            &s.cancel,
            |ev| {
                if cb_error.borrow().is_some() {
                    return;
                }
                let mut env = env_cell.borrow_mut();
                let r = env.with_local_frame(8, |env| -> Result<()> {
                    match ev {
                        TurnEvent::Text(t) => {
                            let jt = env.new_string(t)?;
                            env.call_method(&callback, "onText", "(Ljava/lang/String;)V", &[JValue::Object(&jt)])?;
                        }
                        TurnEvent::ThinkEnd => {
                            env.call_method(&callback, "onThinkEnd", "()V", &[])?;
                        }
                        TurnEvent::ToolCallStarted => {
                            env.call_method(&callback, "onToolCallStarted", "()V", &[])?;
                        }
                        TurnEvent::NewMessage => {
                            env.call_method(&callback, "onNewMessage", "()V", &[])?;
                        }
                        TurnEvent::ToolCall { name, args } => {
                            let (n, a) = (env.new_string(name)?, env.new_string(args)?);
                            env.call_method(
                                &callback,
                                "onToolCall",
                                "(Ljava/lang/String;Ljava/lang/String;)V",
                                &[JValue::Object(&n), JValue::Object(&a)],
                            )?;
                        }
                        TurnEvent::ToolResult { name, ok, content } => {
                            let (n, c) = (env.new_string(name)?, env.new_string(content)?);
                            env.call_method(
                                &callback,
                                "onToolResult",
                                "(Ljava/lang/String;ZLjava/lang/String;)V",
                                &[JValue::Object(&n), JValue::Bool(ok.into()), JValue::Object(&c)],
                            )?;
                        }
                    }
                    Ok(())
                });
                if let Err(e) = r {
                    fail(e);
                }
            },
            |name, args| {
                if cb_error.borrow().is_some() {
                    return false;
                }
                let mut env = env_cell.borrow_mut();
                let r = env.with_local_frame(4, |env| -> Result<bool> {
                    let (n, a) = (env.new_string(name)?, env.new_string(args)?);
                    Ok(env
                        .call_method(
                            &callback,
                            "confirmTool",
                            "(Ljava/lang/String;Ljava/lang/String;)Z",
                            &[JValue::Object(&n), JValue::Object(&a)],
                        )?
                        .z()?)
                });
                r.unwrap_or_else(|e| {
                    fail(e);
                    false
                })
            },
        );
        drop(env_cell);
        if let Some(e) = cb_error.into_inner() {
            return Err(e);
        }
        let json = serde_json::to_string(&output?)?;
        Ok(env.new_string(json)?.into_raw())
    })();
    result.unwrap_or_else(|e| {
        throw(&mut env, &e);
        std::ptr::null_mut()
    })
}

fn parse_role(s: &str) -> Result<Role> {
    Role::parse(s).ok_or_else(|| anyhow!("unknown role {s}"))
}

fn parse_messages(json: &str) -> Result<Vec<Message>> {
    let raw: Vec<JsonMessage> = serde_json::from_str(json).context("invalid messages")?;
    raw.into_iter()
        .map(|m| Ok(Message { role: parse_role(&m.role)?, content: m.content }))
        .collect()
}

// ---------------------------------------------------------------------------
// Conversation store: `dev.stwby.lfmharness.data.NativeStore`

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_data_NativeStore_open<'l>(
    mut env: JNIEnv<'l>,
    _cls: JClass<'l>,
    path: JString<'l>,
) -> jlong {
    let result = (|| -> Result<jlong> {
        let path: String = env.get_string(&path)?.into();
        let store = Store::open(path)?;
        Ok(Box::into_raw(Box::new(Mutex::new(store))) as jlong)
    })();
    result.unwrap_or_else(|e| {
        throw(&mut env, &e);
        0
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_data_NativeStore_close<'l>(
    _env: JNIEnv<'l>,
    _cls: JClass<'l>,
    handle: jlong,
) {
    if handle != 0 {
        drop(unsafe { Box::from_raw(handle as *mut Mutex<Store>) });
    }
}

/// Runs store operation `op` with JSON `args`; returns JSON.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_data_NativeStore_call<'l>(
    mut env: JNIEnv<'l>,
    _cls: JClass<'l>,
    handle: jlong,
    op: JString<'l>,
    args: JString<'l>,
) -> jstring {
    let result = (|| -> Result<jstring> {
        let store = unsafe { (handle as *const Mutex<Store>).as_ref() }.ok_or_else(|| anyhow!("store closed"))?;
        let op: String = env.get_string(&op)?.into();
        let args: String = env.get_string(&args)?.into();
        let args: serde_json::Value = serde_json::from_str(&args).context("invalid args")?;
        let store = store.lock().map_err(|_| anyhow!("store poisoned"))?;
        let out = store_call(&store, &op, &args)?;
        Ok(env.new_string(out.to_string())?.into_raw())
    })();
    result.unwrap_or_else(|e| {
        throw(&mut env, &e);
        std::ptr::null_mut()
    })
}

fn store_call(store: &Store, op: &str, a: &serde_json::Value) -> Result<serde_json::Value> {
    use serde_json::{Value, to_value};
    let id = |k: &str| a[k].as_i64().ok_or_else(|| anyhow!("missing integer `{k}`"));
    let text = |k: &str| a[k].as_str().ok_or_else(|| anyhow!("missing string `{k}`"));
    Ok(match op {
        "list" => to_value(store.list_conversations()?)?,
        "create" => to_value(store.create_conversation(&title_from(text("first_message")?))?)?,
        "rename" => {
            store.rename_conversation(id("id")?, text("title")?)?;
            Value::Null
        }
        "delete" => {
            store.delete_conversation(id("id")?)?;
            Value::Null
        }
        "messages" => to_value(store.messages(id("id")?)?)?,
        "add_message" => to_value(store.add_message(
            id("conversation_id")?,
            &NewMessage {
                role: Some(parse_role(text("role")?)?),
                content: text("content")?,
                reasoning: a["reasoning"].as_str().unwrap_or(""),
                stats: a["stats"].as_str(),
                meta: a["meta"].as_str(),
            },
        )?)?,
        "list_notes" => to_value(store.list_notes()?)?,
        "delete_note" => Value::Bool(store.delete_note(text("title")?)?),
        other => return Err(anyhow!("unknown store op `{other}`")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn store_ops_via_json() {
        let s = Store::open_in_memory().unwrap();
        let c = store_call(&s, "create", &json!({"first_message": "Hola zorro"})).unwrap();
        assert_eq!(c["title"], "Hola zorro");
        let id = c["id"].as_i64().unwrap();
        store_call(&s, "add_message", &json!({"conversation_id": id, "role": "user", "content": "hola"})).unwrap();
        let m = store_call(&s, "messages", &json!({"id": id})).unwrap();
        assert_eq!(m[0]["content"], "hola");
        assert!(m[0]["stats"].is_null());
        store_call(&s, "rename", &json!({"id": id, "title": "x"})).unwrap();
        assert_eq!(store_call(&s, "list", &json!({})).unwrap()[0]["title"], "x");
        assert!(store_call(&s, "nope", &json!({})).is_err());
        assert!(store_call(&s, "delete", &json!({})).is_err());
    }
}

// ---------------------------------------------------------------------------
// Markdown: `dev.stwby.lfmharness.markdown.NativeMarkdown`

/// Parses Markdown into the block JSON described in `markdown.rs`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_stwby_lfmharness_markdown_NativeMarkdown_parse<'l>(
    mut env: JNIEnv<'l>,
    _cls: JClass<'l>,
    text: JString<'l>,
) -> jstring {
    let result = (|| -> Result<jstring> {
        let text: String = env.get_string(&text)?.into();
        Ok(env.new_string(crate::markdown::to_json(&text))?.into_raw())
    })();
    result.unwrap_or_else(|e| {
        throw(&mut env, &e);
        std::ptr::null_mut()
    })
}
