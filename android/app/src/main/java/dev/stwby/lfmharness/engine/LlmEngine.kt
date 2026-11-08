package dev.stwby.lfmharness.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

enum class Role(val wire: String) { System("system"), User("user"), Assistant("assistant") }

data class Turn(val role: Role, val content: String)

data class GenParams(
    val maxTokens: Int = 1024,
    val temperature: Float = 0.3f,
    val topK: Int = 50,
    val topP: Float = 1.0f,
    val minP: Float = 0.15f,
    val repeatPenalty: Float = 1.05f,
    val enableThinking: Boolean = false,
) {
    fun toJson(): String = JSONObject()
        .put("max_tokens", maxTokens)
        .put("temperature", temperature.toDouble())
        .put("top_k", topK)
        .put("top_p", topP.toDouble())
        .put("min_p", minP.toDouble())
        .put("repeat_penalty", repeatPenalty.toDouble())
        .put("enable_thinking", enableThinking)
        .toString()
}

data class GenStats(
    val promptTokens: Int,
    val reusedTokens: Int,
    val promptTps: Double,
    val genTokens: Int,
    val genTps: Double,
    val cancelled: Boolean,
) {
    companion object {
        fun parse(json: String): GenStats = JSONObject(json).run {
            GenStats(
                getInt("prompt_tokens"), getInt("reused_tokens"), getDouble("prompt_tps"),
                getInt("gen_tokens"), getDouble("gen_tps"), getBoolean("cancelled"),
            )
        }
    }
}

sealed interface GenEvent {
    data class Text(val text: String) : GenEvent
    data object ThinkEnd : GenEvent
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

    /** Forget the cached conversation (switching chats). */
    suspend fun reset() = withContext(Dispatchers.IO) {
        mutex.withLock { if (handle != 0L) NativeEngine.reset(handle) }
    }

    /** Runs a generation; [onEvent] is invoked on a background thread. */
    suspend fun generate(
        turns: List<Turn>,
        params: GenParams,
        onEvent: (GenEvent) -> Unit,
    ): GenStats = withContext(Dispatchers.IO) {
        mutex.withLock {
            check(handle != 0L) { "No hay modelo cargado" }
            val messages = JSONArray()
            turns.forEach { messages.put(JSONObject().put("role", it.role.wire).put("content", it.content)) }
            val cb = object : TokenCallback {
                override fun onText(text: String) = onEvent(GenEvent.Text(text))
                override fun onThinkEnd() = onEvent(GenEvent.ThinkEnd)
            }
            GenStats.parse(NativeEngine.generate(handle, messages.toString(), params.toJson(), cb))
        }
    }

    /** Safe from any thread; the running generation stops at the next token. */
    fun cancel() {
        val h = handle
        if (h != 0L) NativeEngine.cancel(h)
    }
}
