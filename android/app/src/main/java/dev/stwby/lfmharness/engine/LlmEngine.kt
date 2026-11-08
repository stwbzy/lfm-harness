package dev.stwby.lfmharness.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

enum class Role(val wire: String) {
    System("system"), User("user"), Assistant("assistant"), Tool("tool");

    companion object {
        fun of(wire: String) = entries.first { it.wire == wire }
    }
}

data class Turn(val role: Role, val content: String)

data class GenParams(
    val maxTokens: Int = 1024,
    val temperature: Float = 0.3f,
    val topK: Int = 50,
    val topP: Float = 1.0f,
    val minP: Float = 0.15f,
    val repeatPenalty: Float = 1.05f,
    val enableThinking: Boolean = false,
    /** Enabled tool names; empty = plain chat. */
    val tools: List<String> = emptyList(),
    /** Local date/time text for `get_datetime`. */
    val now: String = "",
    /** Detect code requests and answer them in code mode (Rust is syntax-checked). */
    val codeMode: Boolean = true,
) {
    fun toJson(): String = JSONObject()
        .put("max_tokens", maxTokens)
        .put("temperature", temperature.toDouble())
        .put("top_k", topK)
        .put("top_p", topP.toDouble())
        .put("min_p", minP.toDouble())
        .put("repeat_penalty", repeatPenalty.toDouble())
        .put("enable_thinking", enableThinking)
        .put("tools", JSONArray(tools))
        .put("now", now)
        .put("code_mode", codeMode)
        .toString()
}

/** A message produced by the turn, as it must be stored and re-sent. */
data class TurnMessage(val role: Role, val content: String, val reasoning: String, val meta: String?)

data class TurnStats(
    val promptTokens: Int,
    val reusedTokens: Int,
    val genTokens: Int,
    val genTps: Double,
    val toolCalls: Int,
    /** Rejected/invalid/failed calls: nothing was done, whatever the model says. */
    val failedCalls: Int,
    val cancelled: Boolean,
    /** `rust_ok`, `rust_repaired`, `rust_syntax_error`, `unchecked` or null. */
    val codeCheck: String?,
)

data class TurnResult(val messages: List<TurnMessage>, val stats: TurnStats) {
    companion object {
        fun parse(json: String): TurnResult {
            val o = JSONObject(json)
            val arr = o.getJSONArray("messages")
            val msgs = List(arr.length()) { i ->
                arr.getJSONObject(i).run {
                    TurnMessage(
                        Role.of(getString("role")), getString("content"), optString("reasoning", ""),
                        if (isNull("meta")) null else getString("meta"),
                    )
                }
            }
            val s = o.getJSONObject("stats")
            return TurnResult(
                msgs,
                TurnStats(
                    s.getInt("prompt_tokens"), s.getInt("reused_tokens"), s.getInt("gen_tokens"),
                    s.getDouble("gen_tps"), s.getInt("tool_calls"), s.getInt("failed_calls"), s.getBoolean("cancelled"),
                    if (s.isNull("code_check")) null else s.getString("code_check"),
                ),
            )
        }
    }
}

/** Process-wide engine: one model loaded at a time, one generation at a time. */
object LlmEngine {
    @Volatile private var handle = 0L
    private val mutex = Mutex()

    val isLoaded: Boolean get() = handle != 0L

    suspend fun load(path: String, nCtx: Int) = withContext(Dispatchers.IO) {
        mutex.withLock {
            unloadLocked()
            handle = NativeEngine.load(path, nCtx, 0)
        }
    }

    suspend fun unload() = withContext(Dispatchers.IO) { mutex.withLock { unloadLocked() } }

    private fun unloadLocked() {
        val h = handle
        handle = 0L
        if (h != 0L) NativeEngine.free(h)
    }

    /** Runs one user turn; [callback] is invoked on a background thread. */
    suspend fun generate(
        turns: List<Turn>,
        params: GenParams,
        storeHandle: Long,
        callback: TurnCallback,
    ): TurnResult = withContext(Dispatchers.IO) {
        mutex.withLock {
            check(handle != 0L) { "No hay modelo cargado" }
            val messages = JSONArray()
            turns.forEach { messages.put(JSONObject().put("role", it.role.wire).put("content", it.content)) }
            TurnResult.parse(NativeEngine.generate(handle, storeHandle, messages.toString(), params.toJson(), callback))
        }
    }

    /** Safe from any thread; the running generation stops at the next token. */
    fun cancel() {
        val h = handle
        if (h != 0L) NativeEngine.cancel(h)
    }
}
