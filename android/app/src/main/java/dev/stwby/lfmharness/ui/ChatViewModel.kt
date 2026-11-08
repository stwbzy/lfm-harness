package dev.stwby.lfmharness.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
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
    val model: ModelState = ModelState.None,
    val generating: Boolean = false,
    val settings: Settings = Settings(),
    val error: String? = null,
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("harness", Context.MODE_PRIVATE)
    private val modelsDir = File(app.filesDir, "models").apply { mkdirs() }
    private val _state = MutableStateFlow(UiState(settings = loadSettings()))
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var nextId = 0L
    private var genJob: Job? = null

    init {
        val path = prefs.getString("model_path", null)
        if (path != null && File(path).isFile) {
            if (LlmEngine.isLoaded) _state.update { it.copy(model = ModelState.Ready(File(path).name)) }
            else loadModel(File(path))
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
        val user = UiMessage(nextId++, Role.User, text.trim())
        val reply = UiMessage(nextId++, Role.Assistant, streaming = true)
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
            try {
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
            } catch (e: Exception) {
                updateMessage(reply.id) { it.copy(streaming = false) }
                _state.update { it.copy(error = e.message) }
            } finally {
                _state.update { it.copy(generating = false) }
            }
        }
    }

    fun stop() = LlmEngine.cancel()

    fun newChat() {
        if (_state.value.generating) return
        _state.update { it.copy(messages = emptyList(), error = null) }
        viewModelScope.launch { LlmEngine.reset() }
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
