package dev.stwby.lfmharness.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.stwby.lfmharness.HarnessApp
import dev.stwby.lfmharness.data.Conversation
import dev.stwby.lfmharness.engine.GenEvent
import dev.stwby.lfmharness.engine.GenParams
import dev.stwby.lfmharness.engine.LlmEngine
import dev.stwby.lfmharness.engine.Role
import dev.stwby.lfmharness.engine.Turn
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class UiMessage(
    val id: Long,
    val role: Role,
    val content: String = "",
    val reasoning: String = "",
    val streaming: Boolean = false,
    val stats: String? = null,
)

data class Settings(
    val systemPrompt: String = "Eres un asistente útil y conciso. Responde en el idioma del usuario.",
    val temperature: Float = 0.3f,
    val maxTokens: Int = 1024,
    val nCtx: Int = 4096,
    val enableThinking: Boolean = false,
)

sealed interface ModelState {
    data object None : ModelState
    data class Importing(val progress: Float) : ModelState
    data class Loading(val name: String) : ModelState
    data class Ready(val name: String) : ModelState
}

data class UiState(
    val messages: List<UiMessage> = emptyList(),
    val conversations: List<Conversation> = emptyList(),
    /** Null until the first message of a new chat is sent. */
    val currentId: Long? = null,
    val model: ModelState = ModelState.None,
    val generating: Boolean = false,
    val settings: Settings = Settings(),
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
    private var genJob: Job? = null

    init {
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

    private suspend fun refreshConversations() {
        val list = repo.conversations()
        _state.update { it.copy(conversations = list) }
    }

    private suspend fun openConversation(id: Long) {
        val msgs = repo.messages(id).map {
            UiMessage(it.id, it.role, it.content, it.reasoning, stats = it.stats)
        }
        _state.update { it.copy(currentId = id, messages = msgs, error = null) }
        prefs.edit().putLong("last_conversation", id).apply()
    }

    fun selectConversation(id: Long) {
        if (_state.value.generating || id == _state.value.currentId) return
        viewModelScope.launch {
            try {
                openConversation(id)
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            }
        }
    }

    fun deleteConversation(id: Long) {
        if (_state.value.generating) return
        viewModelScope.launch {
            try {
                repo.delete(id)
                if (_state.value.currentId == id) newChat()
                refreshConversations()
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            }
        }
    }

    fun renameConversation(id: Long, title: String) {
        if (title.isBlank()) return
        viewModelScope.launch {
            try {
                repo.rename(id, title.trim())
                refreshConversations()
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            }
        }
    }

    fun importModel(uri: Uri) {
        if (_state.value.generating) return
        viewModelScope.launch {
            try {
                _state.update { it.copy(model = ModelState.Importing(0f), error = null) }
                val file = withContext(Dispatchers.IO) { copyToModels(uri) }
                loadModel(file)
            } catch (e: Exception) {
                _state.update { it.copy(model = ModelState.None, error = "Importación fallida: ${e.message}") }
            }
        }
    }

    private suspend fun copyToModels(uri: Uri): File {
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
        // Only one model is kept on disk.
        modelsDir.listFiles()?.forEach { it.delete() }
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
        check(tmp.renameTo(out)) { "no se pudo guardar el modelo" }
        return out
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

    fun send(text: String) {
        val s = _state.value
        if (text.isBlank() || s.generating || s.model !is ModelState.Ready) return
        val user = UiMessage(tempId--, Role.User, text.trim())
        val reply = UiMessage(tempId--, Role.Assistant, streaming = true)
        val history = s.messages + user
        _state.update { it.copy(messages = history + reply, generating = true, error = null) }

        val turns = buildList {
            if (s.settings.systemPrompt.isNotBlank()) add(Turn(Role.System, s.settings.systemPrompt))
            history.forEach { add(Turn(it.role, it.content)) }
        }
        val params = GenParams(
            maxTokens = s.settings.maxTokens,
            temperature = s.settings.temperature,
            enableThinking = s.settings.enableThinking,
        )
        genJob = viewModelScope.launch {
            var inReasoning = params.enableThinking
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

                val stats = LlmEngine.generate(turns, params) { ev ->
                    // `update` may retry its lambda, so keep side effects outside it.
                    when (ev) {
                        is GenEvent.Text -> {
                            val r = inReasoning
                            updateMessage(reply.id) { m ->
                                if (r) m.copy(reasoning = m.reasoning + ev.text)
                                else m.copy(content = m.content + ev.text)
                            }
                        }
                        GenEvent.ThinkEnd -> {
                            val was = inReasoning
                            inReasoning = false
                            // With thinking off, text before </think> was reasoning too.
                            updateMessage(reply.id) { m ->
                                m.copy(reasoning = if (was) m.reasoning else m.reasoning + m.content, content = "")
                            }
                        }
                    }
                }
                val info = "%.1f tok/s · %d tok · prompt %d (%d en caché)%s".format(
                    stats.genTps, stats.genTokens, stats.promptTokens, stats.reusedTokens,
                    if (stats.cancelled) " · detenido" else "",
                )
                updateMessage(reply.id) {
                    it.copy(content = it.content.trim(), reasoning = it.reasoning.trim(), streaming = false, stats = info)
                }
                val done = _state.value.messages.first { it.id == reply.id }
                if (done.content.isNotEmpty() || done.reasoning.isNotEmpty()) {
                    repo.addMessage(convId, Role.Assistant, done.content, done.reasoning, done.stats)
                } else {
                    removeMessage(reply.id)
                }
                refreshConversations()
            } catch (e: Exception) {
                // Keep whatever was streamed on screen; it is not saved.
                updateMessage(reply.id) { it.copy(streaming = false) }
                if (_state.value.messages.first { it.id == reply.id }.let { it.content.isEmpty() && it.reasoning.isEmpty() }) {
                    removeMessage(reply.id)
                }
                _state.update { it.copy(error = e.message) }
            } finally {
                _state.update { it.copy(generating = false) }
            }
        }
    }

    fun stop() = LlmEngine.cancel()

    /** Clears the screen; the conversation row is created on the first message. */
    fun newChat() {
        if (_state.value.generating) return
        _state.update { it.copy(messages = emptyList(), currentId = null, error = null) }
        prefs.edit().remove("last_conversation").apply()
    }

    private fun removeMessage(id: Long) = _state.update { s ->
        s.copy(messages = s.messages.filterNot { it.id == id })
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

    private fun updateMessage(id: Long, f: (UiMessage) -> UiMessage) = _state.update { s ->
        s.copy(messages = s.messages.map { if (it.id == id) f(it) else it })
    }

    private fun loadSettings() = Settings().let { d ->
        Settings(
            systemPrompt = prefs.getString("system_prompt", d.systemPrompt)!!,
            temperature = prefs.getFloat("temperature", d.temperature),
            maxTokens = prefs.getInt("max_tokens", d.maxTokens),
            nCtx = prefs.getInt("n_ctx", d.nCtx),
            enableThinking = prefs.getBoolean("thinking", d.enableThinking),
        )
    }

    private fun saveSettings(s: Settings) = prefs.edit()
        .putString("system_prompt", s.systemPrompt)
        .putFloat("temperature", s.temperature)
        .putInt("max_tokens", s.maxTokens)
        .putInt("n_ctx", s.nCtx)
        .putBoolean("thinking", s.enableThinking)
        .apply()
}
