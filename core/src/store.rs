//! Conversation persistence (SQLite).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

use crate::chat::Role;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Conversation {
    pub id: i64,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StoredMessage {
    pub id: i64,
    pub conversation_id: i64,
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub stats: Option<String>,
    pub created_at: i64,
}

pub struct NewMessage<'a> {
    pub role: Role,
    pub content: &'a str,
    pub reasoning: &'a str,
    pub stats: Option<&'a str>,
}

pub struct Store {
    db: Connection,
}

/// Each entry upgrades the schema by one version (`PRAGMA user_version`).
const MIGRATIONS: &[&str] = &["
    CREATE TABLE conversations (
        id          INTEGER PRIMARY KEY,
        title       TEXT NOT NULL,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL
    );
    CREATE TABLE messages (
        id              INTEGER PRIMARY KEY,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        reasoning       TEXT NOT NULL DEFAULT '',
        stats           TEXT,
        created_at      INTEGER NOT NULL
    );
    CREATE INDEX messages_by_conversation ON messages(conversation_id, id);
"];

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// A short title from the first user message.
pub fn title_from(text: &str) -> String {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("Nuevo chat");
    const MAX: usize = 48;
    if line.chars().count() <= MAX {
        line.to_string()
    } else {
        let mut t: String = line.chars().take(MAX).collect();
        t = t.trim_end().to_string();
        t.push('…');
        t
    }
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = Connection::open(path.as_ref())
            .with_context(|| format!("opening {}", path.as_ref().display()))?;
        Self::init(db)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(db: Connection) -> Result<Self> {
        db.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")?;
        let version: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version as usize > MIGRATIONS.len() {
            bail!("database schema v{version} is newer than this app");
        }
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(version as usize) {
            let tx = db.unchecked_transaction()?;
            tx.execute_batch(sql)?;
            tx.pragma_update(None, "user_version", (i + 1) as i64)?;
            tx.commit()?;
        }
        Ok(Self { db })
    }

    pub fn create_conversation(&self, title: &str) -> Result<Conversation> {
        let t = now_ms();
        self.db.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?1, ?2, ?2)",
            params![title, t],
        )?;
        Ok(Conversation { id: self.db.last_insert_rowid(), title: title.into(), created_at: t, updated_at: t })
    }

    /// Most recently updated first.
    pub fn list_conversations(&self) -> Result<Vec<Conversation>> {
        let mut st = self.db.prepare(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC, id DESC",
        )?;
        let rows = st.query_map([], |r| {
            Ok(Conversation { id: r.get(0)?, title: r.get(1)?, created_at: r.get(2)?, updated_at: r.get(3)? })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn conversation(&self, id: i64) -> Result<Option<Conversation>> {
        Ok(self
            .db
            .query_row(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?1",
                [id],
                |r| Ok(Conversation { id: r.get(0)?, title: r.get(1)?, created_at: r.get(2)?, updated_at: r.get(3)? }),
            )
            .optional()?)
    }

    pub fn rename_conversation(&self, id: i64, title: &str) -> Result<()> {
        self.db.execute("UPDATE conversations SET title = ?2 WHERE id = ?1", params![id, title])?;
        Ok(())
    }

    pub fn delete_conversation(&self, id: i64) -> Result<()> {
        self.db.execute("DELETE FROM conversations WHERE id = ?1", [id])?;
        Ok(())
    }

    pub fn messages(&self, conversation_id: i64) -> Result<Vec<StoredMessage>> {
        let mut st = self.db.prepare(
            "SELECT id, conversation_id, role, content, reasoning, stats, created_at
             FROM messages WHERE conversation_id = ?1 ORDER BY id",
        )?;
        let rows = st.query_map([conversation_id], |r| {
            Ok(StoredMessage {
                id: r.get(0)?,
                conversation_id: r.get(1)?,
                role: r.get(2)?,
                content: r.get(3)?,
                reasoning: r.get(4)?,
                stats: r.get(5)?,
                created_at: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Appends a message and bumps the conversation to the top of the list.
    pub fn add_message(&self, conversation_id: i64, m: &NewMessage) -> Result<StoredMessage> {
        let t = now_ms();
        let tx = self.db.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO messages (conversation_id, role, content, reasoning, stats, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![conversation_id, role_str(m.role), m.content, m.reasoning, m.stats, t],
        )?;
        let id = tx.last_insert_rowid();
        // Strictly newer than every other conversation, even within the same ms.
        let n = tx.execute(
            "UPDATE conversations
             SET updated_at = MAX(?2, (SELECT MAX(updated_at) + 1 FROM conversations))
             WHERE id = ?1",
            params![conversation_id, t],
        )?;
        if n == 0 {
            bail!("conversation {conversation_id} does not exist");
        }
        tx.commit()?;
        Ok(StoredMessage {
            id,
            conversation_id,
            role: role_str(m.role).into(),
            content: m.content.into(),
            reasoning: m.reasoning.into(),
            stats: m.stats.map(Into::into),
            created_at: t,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg<'a>(role: Role, content: &'a str) -> NewMessage<'a> {
        NewMessage { role, content, reasoning: "", stats: None }
    }

    #[test]
    fn roundtrip_and_ordering() {
        let s = Store::open_in_memory().unwrap();
        let a = s.create_conversation("a").unwrap();
        let b = s.create_conversation("b").unwrap();
        s.add_message(a.id, &msg(Role::User, "hola")).unwrap();
        s.add_message(
            a.id,
            &NewMessage { role: Role::Assistant, content: "hey", reasoning: "pienso", stats: Some("10 tok/s") },
        )
        .unwrap();

        let list = s.list_conversations().unwrap();
        assert_eq!(list[0].id, a.id, "conversation with the newest message comes first");
        assert_eq!(list[1].id, b.id);

        let m = s.messages(a.id).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].role, "assistant");
        assert_eq!(m[1].reasoning, "pienso");
        assert_eq!(m[1].stats.as_deref(), Some("10 tok/s"));
    }

    #[test]
    fn delete_cascades() {
        let s = Store::open_in_memory().unwrap();
        let a = s.create_conversation("a").unwrap();
        s.add_message(a.id, &msg(Role::User, "x")).unwrap();
        s.delete_conversation(a.id).unwrap();
        assert!(s.messages(a.id).unwrap().is_empty());
        assert!(s.conversation(a.id).unwrap().is_none());
        assert!(s.add_message(a.id, &msg(Role::User, "y")).is_err());
    }

    #[test]
    fn reopen_keeps_data() {
        let dir = std::env::temp_dir().join(format!("harness-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chats.db");
        {
            let s = Store::open(&path).unwrap();
            let c = s.create_conversation("persist").unwrap();
            s.add_message(c.id, &msg(Role::User, "hola")).unwrap();
        }
        let s = Store::open(&path).unwrap();
        assert_eq!(s.list_conversations().unwrap()[0].title, "persist");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn titles() {
        assert_eq!(title_from("\n  Hola mundo \nmás"), "Hola mundo");
        let long = "a".repeat(60);
        assert_eq!(title_from(&long).chars().count(), 49);
        assert_eq!(title_from("   "), "Nuevo chat");
    }
}
