//! LFM Harness core: local LLM inference and chat logic shared by all front-ends.

pub mod chat;
pub mod engine;
mod jni;

pub use chat::{Message, Role};
pub use engine::{Engine, EngineConfig, Event, GenParams, GenStats};
