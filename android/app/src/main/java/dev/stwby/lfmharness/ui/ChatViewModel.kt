package dev.stwby.lfmharness.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.stwby.lfmharness.HarnessApp
import dev.stwby.lfmharness.data.Conversation
import dev.stwby.lfmharness.data.Note
import dev.stwby.lfmharness.data.StoredMessage
import dev.stwby.lfmharness.engine.GenParams
import dev.stwby.lfmharness.engine.LlmEngine
import dev.stwby.lfmharness.engine.Role
import dev.stwby.lfmharness.engine.Turn
import dev.stwby.lfmharness.engine.TurnCallback
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

enum class Kind { Normal, ToolCall, ToolResult }

data class UiMessage(
    val id: Long,
    val role: Role,
    /** Text shown to the user. */
    val content: String = "",
    val reasoning: String = "",
    val streaming: Boolean = false,
    val stats: String? = null,
    val kind: Kind = Kind.Normal,
    /** Text sent back to the model on later turns (raw markup for tool rounds). */
    val history: String? = null,
    val toolName: String? = null,
    val toolArgs: String? = null,
    val toolOk: Boolean = true,
)

/** Built-in tools (names must match `core/src/tools.rs`) with UI labels. */
val TOOL_LABELS = linkedMapOf(
    "calculator" to "Calculadora",
    "get_datetime" to "Fecha y hora",
    "search_chats" to "Buscar en chats",
    "list_notes" to "Listar notas",
    "read_note" to "Leer notas",
    "save_note" to "Guardar notas (pide permiso)",
    "delete_note" to "Borrar notas (pide permiso)",
)

data class Settings(
    val systemPrompt: String = "Eres un asistente útil y conciso. Responde en el idioma del usuario.",
    val temperature: Float = 0.3f,
    val maxTokens: Int = 1024,
    val nCtx: Int = 4096,
    val enableThinking: Boolean = false,
    val toolsEnabled: Boolean = true,
    val codeMode: Boolean = true,
    val enabledTools: Set<String> = TOOL_LABELS.keys,
)

sealed interface ModelState {
    data object None : ModelState
    data class Importing(val progress: Float) : ModelState
    data class Loading(val name: String) : ModelState
    data class Ready(val name: String) : ModelState
}

/** A model file kept in the app's private storage. */
data class StoredModel(val name: String, val sizeBytes: Long)

/** A tool call waiting for the user's approval. */
data class PendingConfirm(val name: String, val args: Map<String, String>, val answer: CompletableDeferred<Boolean>)

data class UiState(
    val messages: List<UiMessage> = emptyList(),
    val conversations: List<Conversation> = emptyList(),
    /** Null until the first message of a new chat is sent. */
    val currentId: Long? = null,
    val model: ModelState = ModelState.None,
    /** Every imported model; the active one is named by [model]. */
    val models: List<StoredModel> = emptyList(),
    val generating: Boolean = false,
    val settings: Settings = Settings(),
    val confirm: PendingConfirm? = null,
    val notes: List<Note>? = null,
    val error: String? = null,
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("harness", Context.MODE_PRIVATE)
    private val modelsDir = File(app.filesDir, "models").apply { mkdirs() }
    private val repo = (app as HarnessApp).chats
    private val _state = MutableStateFlow(UiState(settings = loadSettings()))
    val state: StateFlow<UiState> = _state.asStateFlow()
    /** Negative ids for messages not yet loaded from the database. */
    private var tempId = -1L

    init {
        refreshModels()
        val path = prefs.getString("model_path", null)
        if (path != null && File(path).isFile) {
            if (LlmEngine.isLoaded) _state.update { it.copy(model = ModelState.Ready(File(path).name)) }
            else loadModel(File(path))
        }
        viewModelScope.launch {
            try {
                refreshConversations()
                val last = prefs.getLong("last_conversation", -1L)
                if (_state.value.conversations.any { it.id == last }) openConversation(last)
            } catch (e: Exception) {
                _state.update { it.copy(error = "Historial no disponible: ${e.message}") }
            }
        }
    }

    // -- conversations ------------------------------------------------------

    private suspend fun refreshConversations() {
        val list = repo.conversations()
        _state.update { it.copy(conversations = list) }
    }

    private suspend fun openConversation(id: Long) {
        val msgs = repo.messages(id).map(::toUi)
        _state.update { it.copy(currentId = id, messages = msgs, error = null) }
        prefs.edit().putLong("last_conversation", id).apply()
    }

    fun selectConversation(id: Long) {
        if (_state.value.generating || id == _state.value.currentId) return
        launchCatching { openConversation(id) }
    }

    fun deleteConversation(id: Long) {
        if (_state.value.generating) return
        launchCatching {
            repo.delete(id)
            if (_state.value.currentId == id) newChat()
            refreshConversations()
        }
    }

    fun renameConversation(id: Long, title: String) {
        if (title.isBlank()) return
        launchCatching {
            repo.rename(id, title.trim())
            refreshConversations()
        }
    }

    /** Clears the screen; the conversation row is created on the first message. */
    fun newChat() {
        if (_state.value.generating) return
        _state.update { it.copy(messages = emptyList(), currentId = null, error = null) }
        prefs.edit().remove("last_conversation").apply()
    }

    // -- notes ------------------------------------------------------------

    fun showNotes() = launchCatching { _state.update { it.copy(notes = emptyList()) }; reloadNotes() }

    fun hideNotes() = _state.update { it.copy(notes = null) }

    fun deleteNote(title: String) = launchCatching {
        repo.deleteNote(title)
        reloadNotes()
    }

    private suspend fun reloadNotes() {
        val notes = repo.notes()
        _state.update { it.copy(notes = notes) }
    }

    // -- model --------------------------------------------------------------

    fun importModel(uri: Uri) {
        if (_state.value.generating) return
        viewModelScope.launch {
            try {
                _state.update { it.copy(model = ModelState.Importing(0f), error = null) }
                val file = withContext(Dispatchers.IO) { copyToModels(uri) }
                refreshModels()
                loadModel(file)
            } catch (e: Exception) {
                _state.update { it.copy(model = ModelState.None, error = "Importación fallida: ${e.message}") }
            }
        }
    }

    private fun copyToModels(uri: Uri): File {
        val resolver = getApplication<Application>().contentResolver
        var name = "model.gguf"
        var size = -1L
        resolver.query(uri, null, null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                c.getColumnIndex(OpenableColumns.DISPLAY_NAME).takeIf { it >= 0 }?.let { name = c.getString(it) }
                c.getColumnIndex(OpenableColumns.SIZE).takeIf { it >= 0 }?.let { size = c.getLong(it) }
            }
        }
        require(name.endsWith(".gguf", ignoreCase = true)) { "el archivo no es .gguf" }
        require(!name.contains('/') && !name.endsWith(".part")) { "nombre de archivo no válido" }
        val out = File(modelsDir, name)
        val tmp = File(modelsDir, "$name.part")
        resolver.openInputStream(uri)!!.use { input ->
            tmp.outputStream().use { output ->
                val buf = ByteArray(1 shl 20)
                var copied = 0L
                var lastReport = 0f
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    output.write(buf, 0, n)
                    copied += n
                    if (size > 0) {
                        val p = copied.toFloat() / size
                        if (p - lastReport > 0.01f) {
                            lastReport = p
                            _state.update { it.copy(model = ModelState.Importing(p)) }
                        }
                    }
                }
            }
        }
        // Re-importing a file with the same name replaces it.
        out.delete()
        check(tmp.renameTo(out)) { "no se pudo guardar el modelo" }
        return out
    }

    private fun refreshModels() {
        val list = modelsDir.listFiles { f -> f.isFile && f.name.endsWith(".gguf", ignoreCase = true) }
            ?.map { StoredModel(it.name, it.length()) }?.sortedBy { it.name } ?: emptyList()
        _state.update { it.copy(models = list) }
    }

    /** Switches to another imported model (not while generating). */
    fun selectModel(name: String) {
        val s = _state.value
        if (s.generating || (s.model as? ModelState.Ready)?.name == name) return
        val file = File(modelsDir, name).takeIf { it.isFile } ?: return
        loadModel(file)
    }

    /** Deletes an imported model; the active one cannot be deleted. */
    fun deleteModel(name: String) {
        val s = _state.value
        val active = when (val m = s.model) {
            is ModelState.Ready -> m.name
            is ModelState.Loading -> m.name
            else -> null
        }
        if (s.generating || name == active) {
            _state.update { it.copy(error = "No se puede borrar el modelo en uso") }
            return
        }
        File(modelsDir, name).delete()
        refreshModels()
    }

    private fun loadModel(file: File) {
        viewModelScope.launch {
            _state.update { it.copy(model = ModelState.Loading(file.name), error = null) }
            try {
                LlmEngine.load(file.absolutePath, _state.value.settings.nCtx)
                prefs.edit().putString("model_path", file.absolutePath).apply()
                _state.update { it.copy(model = ModelState.Ready(file.name)) }
            } catch (e: Exception) {
                _state.update { it.copy(model = ModelState.None, error = "No se pudo cargar: ${e.message}") }
            }
        }
    }

    // -- generation ---------------------------------------------------------

    fun send(text: String) {
        val s = _state.value
        if (text.isBlank() || s.generating || s.model !is ModelState.Ready) return
        val user = UiMessage(tempId--, Role.User, text.trim())
        val history = s.messages + user
        val firstReply = UiMessage(tempId--, Role.Assistant, streaming = true)
        _state.update { it.copy(messages = history + firstReply, generating = true, error = null) }

        val turns = buildList {
            if (s.settings.systemPrompt.isNotBlank()) add(Turn(Role.System, s.settings.systemPrompt))
            history.forEach { add(Turn(it.role, it.history ?: it.content)) }
        }
        val params = GenParams(
            maxTokens = s.settings.maxTokens,
            temperature = s.settings.temperature,
            enableThinking = s.settings.enableThinking,
            tools = if (s.settings.toolsEnabled) TOOL_LABELS.keys.filter { it in s.settings.enabledTools } else emptyList(),
            now = nowText(),
            codeMode = s.settings.codeMode,
        )
        viewModelScope.launch {
            var convId: Long? = s.currentId
            try {
                // Persist the user turn first so it survives a crash mid-generation.
                if (convId == null) {
                    convId = repo.create(user.content).id
                    _state.update { it.copy(currentId = convId) }
                    prefs.edit().putLong("last_conversation", convId).apply()
                }
                repo.addMessage(convId, Role.User, user.content)
                refreshConversations()

                val result = LlmEngine.generate(turns, params, repo.handle, StreamCallback(firstReply.id, params.enableThinking))
                val st = result.stats
                val codeLabel = when (st.codeCheck) {
                    "rust_ok" -> " · ✓ sintaxis Rust válida (no compilado)"
                    "rust_repaired" -> " · ✓ sintaxis corregida automáticamente (no compilado)"
                    "rust_syntax_error" -> " · ⚠ el código Rust tiene errores de sintaxis"
                    "unchecked" -> " · código no verificado"
                    else -> ""
                }
                val info = "%.1f tok/s · %d tok · prompt %d (%d en caché)%s%s%s%s".format(
                    st.genTps, st.genTokens, st.promptTokens, st.reusedTokens, codeLabel,
                    if (st.toolCalls > 0) " · ${st.toolCalls} herramienta(s)" else "",
                    if (st.failedCalls > 0) " · ⚠ ${st.failedCalls} acción(es) no realizada(s)" else "",
                    if (st.cancelled) " · detenido" else "",
                )
                // Store the authoritative messages, then show them as stored.
                val lastFinal = result.messages.indexOfLast { it.meta == null }
                result.messages.forEachIndexed { i, m ->
                    repo.addMessage(convId, m.role, m.content, m.reasoning, if (i == lastFinal) info else null, m.meta)
                }
                openConversation(convId)
                refreshConversations()
            } catch (e: Exception) {
                // Keep whatever was streamed on screen; it is not saved.
                _state.update { st ->
                    st.copy(
                        messages = st.messages
                            .map { it.copy(streaming = false) }
                            .filterNot { it.role == Role.Assistant && it.kind == Kind.Normal && it.content.isEmpty() && it.reasoning.isEmpty() },
                        error = e.message,
                    )
                }
            } finally {
                _state.value.confirm?.answer?.complete(false)
                _state.update { it.copy(generating = false, confirm = null) }
            }
        }
    }

    fun stop() {
        _state.value.confirm?.answer?.complete(false)
        LlmEngine.cancel()
    }

    fun answerConfirm(allow: Boolean) {
        val c = _state.value.confirm ?: return
        _state.update { it.copy(confirm = null) }
        c.answer.complete(allow)
    }

    /** Streams one turn into the message list. Runs on the generation thread. */
    private inner class StreamCallback(firstId: Long, private val thinking: Boolean) : TurnCallback {
        private var currentId = firstId
        private var inReasoning = thinking

        override fun onText(text: String) {
            val r = inReasoning
            updateMessage(currentId) { m ->
                if (r) m.copy(reasoning = m.reasoning + text) else m.copy(content = m.content + text)
            }
        }

        override fun onThinkEnd() {
            val was = inReasoning
            inReasoning = false
            // With thinking off, text before </think> was reasoning too.
            updateMessage(currentId) { m ->
                m.copy(reasoning = if (was) m.reasoning else m.reasoning + m.content, content = "")
            }
        }

        override fun onToolCallStarted() {
            updateMessage(currentId) { it.copy(kind = Kind.ToolCall, toolName = "…") }
        }

        override fun onToolCall(name: String, argsJson: String) {
            updateMessage(currentId) { it.copy(kind = Kind.ToolCall, toolName = name, toolArgs = argsText(argsJson), streaming = false) }
        }

        override fun onToolResult(name: String, ok: Boolean, content: String) {
            val msg = UiMessage(tempId--, Role.Tool, content, kind = Kind.ToolResult, toolName = name, toolOk = ok)
            _state.update { it.copy(messages = it.messages + msg) }
        }

        override fun onNewMessage() {
            inReasoning = thinking
            val msg = UiMessage(tempId--, Role.Assistant, streaming = true)
            currentId = msg.id
            _state.update { it.copy(messages = it.messages + msg) }
        }

        override fun confirmTool(name: String, argsJson: String): Boolean {
            val answer = CompletableDeferred<Boolean>()
            _state.update { it.copy(confirm = PendingConfirm(name, argsMap(argsJson), answer)) }
            // Blocks the generation thread (never the UI thread) until answered.
            return runBlocking { answer.await() }
        }
    }

    // -- helpers ------------------------------------------------------------

    private fun toUi(m: StoredMessage): UiMessage {
        val meta = m.meta?.let { runCatching { JSONObject(it) }.getOrNull() }
        return when {
            m.role == Role.Tool -> UiMessage(
                m.id, m.role, m.content, kind = Kind.ToolResult,
                toolName = meta?.optString("tool"), toolOk = meta?.optBoolean("ok", true) ?: true,
            )
            meta?.has("tool_call") == true -> {
                val call = meta.getJSONObject("tool_call")
                UiMessage(
                    m.id, m.role, visibleBeforeToolCall(m.content), kind = Kind.ToolCall, history = m.content,
                    toolName = call.optString("name").ifEmpty { "?" },
                    toolArgs = call.optJSONObject("args")?.toString()?.let(::argsText),
                )
            }
            else -> UiMessage(m.id, m.role, m.content, m.reasoning, stats = m.stats)
        }
    }

    /** User-visible text of a raw tool-call turn: after any `</think>`, before `<tool_call>`. */
    private fun visibleBeforeToolCall(raw: String): String {
        val afterThink = raw.substringAfterLast("</think>", raw.removePrefix("<think>"))
        return afterThink.substringBefore("<tool_call>").trim()
    }

    private fun argsMap(json: String): Map<String, String> = runCatching {
        val o = JSONObject(json)
        o.keys().asSequence().associateWith { o.optString(it) }
    }.getOrDefault(emptyMap())

    private fun argsText(json: String) = argsMap(json).entries.joinToString(", ") { (k, v) ->
        val short = if (v.length > 60) v.take(60) + "…" else v
        "$k: ${short.replace('\n', ' ')}"
    }

    private fun nowText(): String {
        val now = ZonedDateTime.now()
        val es = Locale.forLanguageTag("es")
        val date = now.format(DateTimeFormatter.ofPattern("EEEE, d 'de' MMMM 'de' yyyy, HH:mm", es))
        return "$date (${now.zone.id}, UTC${now.offset.id.replace("Z", "+00:00")})"
    }

    private fun updateMessage(id: Long, f: (UiMessage) -> UiMessage) = _state.update { s ->
        s.copy(messages = s.messages.map { if (it.id == id) f(it) else it })
    }

    private fun launchCatching(block: suspend () -> Unit) = viewModelScope.launch {
        try {
            block()
        } catch (e: Exception) {
            _state.update { it.copy(error = e.message) }
        }
    }

    fun dismissError() = _state.update { it.copy(error = null) }

    fun updateSettings(new: Settings) {
        val old = _state.value.settings
        _state.update { it.copy(settings = new) }
        saveSettings(new)
        // Context size is fixed at load time.
        if (new.nCtx != old.nCtx && !_state.value.generating) {
            prefs.getString("model_path", null)?.let { File(it) }?.takeIf { it.isFile }?.let(::loadModel)
        }
    }

    private fun loadSettings() = Settings().let { d ->
        Settings(
            systemPrompt = prefs.getString("system_prompt", d.systemPrompt)!!,
            temperature = prefs.getFloat("temperature", d.temperature),
            maxTokens = prefs.getInt("max_tokens", d.maxTokens),
            nCtx = prefs.getInt("n_ctx", d.nCtx),
            enableThinking = prefs.getBoolean("thinking", d.enableThinking),
            toolsEnabled = prefs.getBoolean("tools_enabled", d.toolsEnabled),
            codeMode = prefs.getBoolean("code_mode", d.codeMode),
            enabledTools = prefs.getStringSet("enabled_tools", null)?.toSet() ?: d.enabledTools,
        )
    }

    private fun saveSettings(s: Settings) = prefs.edit()
        .putString("system_prompt", s.systemPrompt)
        .putFloat("temperature", s.temperature)
        .putInt("max_tokens", s.maxTokens)
        .putInt("n_ctx", s.nCtx)
        .putBoolean("thinking", s.enableThinking)
        .putBoolean("tools_enabled", s.toolsEnabled)
        .putBoolean("code_mode", s.codeMode)
        .putStringSet("enabled_tools", s.enabledTools)
        .apply()
}
