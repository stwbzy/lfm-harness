package dev.stwby.lfmharness.data

import dev.stwby.lfmharness.engine.Role
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/** JNI surface of the Rust SQLite store (`core/src/store.rs`, `core/src/jni.rs`). */
internal object NativeStore {
    init {
        System.loadLibrary("harness_core")
    }

    @JvmStatic external fun open(path: String): Long
    @JvmStatic external fun close(handle: Long)
    @JvmStatic external fun call(handle: Long, op: String, argsJson: String): String
}

data class Conversation(val id: Long, val title: String, val updatedAt: Long)

data class StoredMessage(
    val id: Long,
    val role: Role,
    val content: String,
    val reasoning: String,
    val stats: String?,
    /** Tool-round details (JSON) or null for normal messages. */
    val meta: String?,
)

data class Note(val title: String, val content: String, val updatedAt: Long)

/** Conversation history and notes persisted in `chats.db`. All calls run on the IO dispatcher. */
class ChatRepository(dbPath: String) {
    /** Also handed to the engine so tools can read chats and notes. */
    internal val handle = NativeStore.open(dbPath)

    private suspend fun call(op: String, args: JSONObject = JSONObject()): String =
        withContext(Dispatchers.IO) { NativeStore.call(handle, op, args.toString()) }

    suspend fun conversations(): List<Conversation> {
        val arr = JSONArray(call("list"))
        return List(arr.length()) { i -> arr.getJSONObject(i).toConversation() }
    }

    /** Creates a conversation titled after its first user message. */
    suspend fun create(firstMessage: String): Conversation =
        JSONObject(call("create", JSONObject().put("first_message", firstMessage))).toConversation()

    suspend fun rename(id: Long, title: String) {
        call("rename", JSONObject().put("id", id).put("title", title))
    }

    suspend fun delete(id: Long) {
        call("delete", JSONObject().put("id", id))
    }

    suspend fun messages(conversationId: Long): List<StoredMessage> {
        val arr = JSONArray(call("messages", JSONObject().put("id", conversationId)))
        return List(arr.length()) { i -> arr.getJSONObject(i).toMessage() }
    }

    suspend fun addMessage(
        conversationId: Long,
        role: Role,
        content: String,
        reasoning: String = "",
        stats: String? = null,
        meta: String? = null,
    ): StoredMessage {
        val args = JSONObject()
            .put("conversation_id", conversationId)
            .put("role", role.wire)
            .put("content", content)
            .put("reasoning", reasoning)
        if (stats != null) args.put("stats", stats)
        if (meta != null) args.put("meta", meta)
        return JSONObject(call("add_message", args)).toMessage()
    }

    suspend fun notes(): List<Note> {
        val arr = JSONArray(call("list_notes"))
        return List(arr.length()) { i ->
            arr.getJSONObject(i).run { Note(getString("title"), getString("content"), getLong("updated_at")) }
        }
    }

    suspend fun deleteNote(title: String) {
        call("delete_note", JSONObject().put("title", title))
    }

    private fun JSONObject.toConversation() =
        Conversation(getLong("id"), getString("title"), getLong("updated_at"))

    private fun JSONObject.toMessage() = StoredMessage(
        id = getLong("id"),
        role = Role.of(getString("role")),
        content = getString("content"),
        reasoning = optString("reasoning", ""),
        stats = if (isNull("stats")) null else getString("stats"),
        meta = if (!has("meta") || isNull("meta")) null else getString("meta"),
    )
}
