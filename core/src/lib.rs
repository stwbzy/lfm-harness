//! LFM Harness core: local LLM inference and chat logic shared by all front-ends.

pub mod chat;
pub mod codecheck;
pub mod engine;
pub mod harness;
mod jni;
pub mod markdown;
pub mod store;
pub mod tools;

pub use chat::{Message, Role};
pub use engine::{Engine, EngineConfig, Event, GenOutput, GenParams, GenStats};
