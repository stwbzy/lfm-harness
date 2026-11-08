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
    /// JSON with tool-call details for tool rounds.
    pub meta: Option<String>,
    pub created_at: i64,
}

#[derive(Default)]
pub struct NewMessage<'a> {
    pub role: Option<Role>,
    pub content: &'a str,
    pub reasoning: &'a str,
    pub stats: Option<&'a str>,
    pub meta: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Note {
    pub title: String,
    pub content: String,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchHit {
    pub conversation_id: i64,
    pub conversation_title: String,
    pub role: String,
    pub snippet: String,
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
", "
    ALTER TABLE messages ADD COLUMN meta TEXT;
    CREATE TABLE notes (
        id          INTEGER PRIMARY KEY,
        title       TEXT NOT NULL UNIQUE COLLATE NOCASE,
        content     TEXT NOT NULL,
        updated_at  INTEGER NOT NULL
    );
"];

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
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
            "SELECT id, conversation_id, role, content, reasoning, stats, meta, created_at
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
                meta: r.get(6)?,
                created_at: r.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Appends a message and bumps the conversation to the top of the list.
    pub fn add_message(&self, conversation_id: i64, m: &NewMessage) -> Result<StoredMessage> {
        let role = m.role.context("message role missing")?.as_str();
        let t = now_ms();
        let tx = self.db.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO messages (conversation_id, role, content, reasoning, stats, meta, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![conversation_id, role, m.content, m.reasoning, m.stats, m.meta, t],
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
            role: role.into(),
            content: m.content.into(),
            reasoning: m.reasoning.into(),
            stats: m.stats.map(Into::into),
            meta: m.meta.map(Into::into),
            created_at: t,
        })
    }

    // -- notes (used by the note tools) -----------------------------------

    pub fn note_count(&self) -> Result<usize> {
        Ok(self.db.query_row("SELECT COUNT(*) FROM notes", [], |r| r.get::<_, i64>(0))? as usize)
    }

    /// Creates or replaces a note by title (case-insensitive). Returns true if new.
    pub fn save_note(&self, title: &str, content: &str) -> Result<bool> {
        let exists = self.get_note(title)?.is_some();
        self.db.execute(
            "INSERT INTO notes (title, content, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(title) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
            params![title, content, now_ms()],
        )?;
        Ok(!exists)
    }

    pub fn get_note(&self, title: &str) -> Result<Option<Note>> {
        Ok(self
            .db
            .query_row("SELECT title, content, updated_at FROM notes WHERE title = ?1", [title], |r| {
                Ok(Note { title: r.get(0)?, content: r.get(1)?, updated_at: r.get(2)? })
            })
            .optional()?)
    }

    /// Most recently updated first.
    pub fn list_notes(&self) -> Result<Vec<Note>> {
        let mut st = self.db.prepare("SELECT title, content, updated_at FROM notes ORDER BY updated_at DESC")?;
        let rows = st.query_map([], |r| Ok(Note { title: r.get(0)?, content: r.get(1)?, updated_at: r.get(2)? }))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn delete_note(&self, title: &str) -> Result<bool> {
        Ok(self.db.execute("DELETE FROM notes WHERE title = ?1", [title])? > 0)
    }

    // -- search ---------------------------------------------------------

    /// Case-insensitive substring search over user and final assistant messages.
    pub fn search_messages(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let pattern = format!("%{}%", query.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
        let mut st = self.db.prepare(
            "SELECT m.conversation_id, c.title, m.role, m.content
             FROM messages m JOIN conversations c ON c.id = m.conversation_id
             WHERE m.role IN ('user', 'assistant') AND m.meta IS NULL
               AND m.content LIKE ?1 ESCAPE '\\'
             ORDER BY m.id DESC LIMIT ?2",
        )?;
        let rows = st.query_map(params![pattern, limit as i64], |r| {
            let content: String = r.get(3)?;
            Ok(SearchHit {
                conversation_id: r.get(0)?,
                conversation_title: r.get(1)?,
                role: r.get(2)?,
                snippet: snippet(&content, query, 160),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}

/// Up to `max` chars of `text` around the first case-insensitive match of `query`.
fn snippet(text: &str, query: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let lower: Vec<char> = text.to_lowercase().chars().collect();
    let q: Vec<char> = query.to_lowercase().chars().collect();
    // Lowercasing can change length for a few scripts; fall back to the start.
    let at = if lower.len() == chars.len() && !q.is_empty() {
        lower.windows(q.len()).position(|w| w == q.as_slice()).unwrap_or(0)
    } else {
        0
    };
    let start = at.saturating_sub(max / 3);
    let end = (start + max).min(chars.len());
    let mut s: String = chars[start..end].iter().collect();
    if start > 0 {
        s.insert(0, '…');
    }
    if end < chars.len() {
        s.push('…');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg<'a>(role: Role, content: &'a str) -> NewMessage<'a> {
        NewMessage { role: Some(role), content, ..Default::default() }
    }

    #[test]
    fn roundtrip_and_ordering() {
        let s = Store::open_in_memory().unwrap();
        let a = s.create_conversation("a").unwrap();
        let b = s.create_conversation("b").unwrap();
        s.add_message(a.id, &msg(Role::User, "hola")).unwrap();
        s.add_message(
            a.id,
            &NewMessage {
                role: Some(Role::Assistant),
                content: "hey",
                reasoning: "pienso",
                stats: Some("10 tok/s"),
                meta: None,
            },
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
    fn notes_crud() {
        let s = Store::open_in_memory().unwrap();
        assert!(s.save_note("Compra", "leche").unwrap());
        assert!(!s.save_note("compra", "pan").unwrap(), "titles are case-insensitive");
        assert_eq!(s.note_count().unwrap(), 1);
        assert_eq!(s.get_note("COMPRA").unwrap().unwrap().content, "pan");
        assert!(s.delete_note("Compra").unwrap());
        assert!(!s.delete_note("Compra").unwrap());
        assert!(s.list_notes().unwrap().is_empty());
    }

    #[test]
    fn search_skips_tool_rounds_and_escapes() {
        let s = Store::open_in_memory().unwrap();
        let c = s.create_conversation("zorros").unwrap();
        s.add_message(c.id, &msg(Role::User, "Háblame de los zorros árticos")).unwrap();
        s.add_message(
            c.id,
            &NewMessage { role: Some(Role::Assistant), content: "zorros <tool_call>", meta: Some("{}"), ..Default::default() },
        )
        .unwrap();
        s.add_message(c.id, &msg(Role::User, "100% seguro")).unwrap();
        let hits = s.search_messages("ZORROS", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].conversation_title, "zorros");
        assert!(hits[0].snippet.contains("zorros árticos"));
        assert_eq!(s.search_messages("0%", 5).unwrap().len(), 1);
        assert!(s.search_messages("%", 5).unwrap().len() == 1, "% is literal");
    }

    #[test]
    fn migrates_v1_database() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(MIGRATIONS[0]).unwrap();
        db.pragma_update(None, "user_version", 1i64).unwrap();
        db.execute("INSERT INTO conversations (title, created_at, updated_at) VALUES ('old', 1, 1)", []).unwrap();
        let s = Store::init(db).unwrap();
        assert_eq!(s.list_conversations().unwrap()[0].title, "old");
        assert!(s.save_note("a", "b").unwrap());
    }

    #[test]
    fn titles() {
        assert_eq!(title_from("\n  Hola mundo \nmás"), "Hola mundo");
        let long = "a".repeat(60);
        assert_eq!(title_from(&long).chars().count(), 49);
        assert_eq!(title_from("   "), "Nuevo chat");
    }
}
