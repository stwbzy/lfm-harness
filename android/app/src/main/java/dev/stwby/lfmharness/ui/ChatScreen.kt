package dev.stwby.lfmharness.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material3.Checkbox
import androidx.compose.ui.text.font.FontFamily
import dev.stwby.lfmharness.data.Conversation
import dev.stwby.lfmharness.data.Note
import dev.stwby.lfmharness.markdown.MarkdownText
import kotlinx.coroutines.launch
import java.text.DateFormat
import java.util.Date
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.stwby.lfmharness.engine.Role

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()
    var showSettings by remember { mutableStateOf(false) }
    val snackbar = remember { SnackbarHostState() }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(vm::importModel)
    }

    LaunchedEffect(state.error) {
        state.error?.let {
            snackbar.showSnackbar(it)
            vm.dismissError()
        }
    }

    val drawer = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    ModalNavigationDrawer(
        drawerState = drawer,
        drawerContent = {
            ConversationDrawer(
                conversations = state.conversations,
                currentId = state.currentId,
                enabled = !state.generating,
                onNew = { vm.newChat(); scope.launch { drawer.close() } },
                onSelect = { vm.selectConversation(it); scope.launch { drawer.close() } },
                onRename = vm::renameConversation,
                onDelete = vm::deleteConversation,
                onNotes = { vm.showNotes(); scope.launch { drawer.close() } },
            )
        },
    ) {
    Scaffold(
        modifier = Modifier.imePadding(),
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = { scope.launch { drawer.open() } }) {
                        Icon(Icons.Default.Menu, "Chats")
                    }
                },
                title = {
                    Column {
                        Text(
                            state.conversations.firstOrNull { it.id == state.currentId }?.title ?: "Nuevo chat",
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(modelLabel(state.model), style = MaterialTheme.typography.labelSmall)
                    }
                },
                actions = {
                    IconButton(onClick = vm::newChat, enabled = !state.generating) {
                        Icon(Icons.Default.Add, "Nuevo chat")
                    }
                    IconButton(onClick = { picker.launch(arrayOf("*/*")) }, enabled = !state.generating) {
                        Icon(Icons.Default.FolderOpen, "Cargar modelo")
                    }
                    IconButton(onClick = { showSettings = true }) {
                        Icon(Icons.Default.Settings, "Ajustes")
                    }
                },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            when (val m = state.model) {
                is ModelState.Importing -> LinearProgressIndicator(
                    progress = { m.progress }, modifier = Modifier.fillMaxWidth(),
                )
                is ModelState.Loading -> LinearProgressIndicator(Modifier.fillMaxWidth())
                else -> {}
            }
            Box(Modifier.weight(1f)) {
                if (state.messages.isEmpty()) {
                    EmptyHint(state.model, onPick = { picker.launch(arrayOf("*/*")) })
                } else {
                    MessageList(state.messages)
                }
            }
            InputBar(
                enabled = state.model is ModelState.Ready,
                generating = state.generating,
                onSend = vm::send,
                onStop = vm::stop,
            )
        }
    }
    }

    if (showSettings) {
        SettingsDialog(state.settings, onDismiss = { showSettings = false }) {
            vm.updateSettings(it)
            showSettings = false
        }
    }
    state.confirm?.let { c ->
        AlertDialog(
            onDismissRequest = {},
            title = { Text("¿Permitir ${TOOL_LABELS[c.name]?.substringBefore(" (") ?: c.name}?") },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState())) {
                    Text("El modelo quiere usar la herramienta «${c.name}» con:")
                    Spacer(Modifier.size(8.dp))
                    c.args.forEach { (k, v) ->
                        Text(k, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                        Text(v, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
                        Spacer(Modifier.size(6.dp))
                    }
                }
            },
            confirmButton = { TextButton(onClick = { vm.answerConfirm(true) }) { Text("Permitir") } },
            dismissButton = { TextButton(onClick = { vm.answerConfirm(false) }) { Text("Rechazar") } },
        )
    }
    state.notes?.let { notes ->
        NotesDialog(notes, onDelete = vm::deleteNote, onDismiss = vm::hideNotes)
    }
}

@Composable
private fun NotesDialog(notes: List<Note>, onDelete: (String) -> Unit, onDismiss: () -> Unit) {
    var open by remember { mutableStateOf<String?>(null) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Notas (${notes.size})") },
        text = {
            if (notes.isEmpty()) {
                Text("No hay notas. Pídele al modelo que guarde una.")
            } else {
                LazyColumn {
                    items(notes, key = { it.title }) { n ->
                        Column(Modifier.fillMaxWidth().clickable { open = if (open == n.title) null else n.title }.padding(vertical = 8.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(n.title, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                                IconButton(onClick = { onDelete(n.title) }) { Icon(Icons.Default.Delete, "Borrar nota") }
                            }
                            Text(
                                n.content,
                                maxLines = if (open == n.title) Int.MAX_VALUE else 2,
                                overflow = TextOverflow.Ellipsis,
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        HorizontalDivider()
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Cerrar") } },
    )
}

@Composable
private fun ConversationDrawer(
    conversations: List<Conversation>,
    currentId: Long?,
    enabled: Boolean,
    onNew: () -> Unit,
    onSelect: (Long) -> Unit,
    onRename: (Long, String) -> Unit,
    onDelete: (Long) -> Unit,
    onNotes: () -> Unit,
) {
    var renaming by remember { mutableStateOf<Conversation?>(null) }
    var deleting by remember { mutableStateOf<Conversation?>(null) }
    val dateFmt = remember { DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT) }

    ModalDrawerSheet(Modifier.widthIn(max = 320.dp)) {
        Text("Chats", style = MaterialTheme.typography.titleLarge, modifier = Modifier.padding(16.dp))
        NavigationDrawerItem(
            label = { Text("Nuevo chat") },
            icon = { Icon(Icons.Default.Add, null) },
            selected = currentId == null,
            onClick = { if (enabled) onNew() },
            modifier = Modifier.padding(horizontal = 12.dp),
        )
        NavigationDrawerItem(
            label = { Text("Notas") },
            icon = { Icon(Icons.Default.Description, null) },
            selected = false,
            onClick = onNotes,
            modifier = Modifier.padding(horizontal = 12.dp),
        )
        HorizontalDivider(Modifier.padding(vertical = 8.dp))
        if (conversations.isEmpty()) {
            Text(
                "Aún no hay conversaciones.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 28.dp),
            )
        }
        LazyColumn {
            items(conversations, key = { it.id }) { c ->
                var menu by remember { mutableStateOf(false) }
                NavigationDrawerItem(
                    label = {
                        Column {
                            Text(c.title, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(
                                dateFmt.format(Date(c.updatedAt)),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    },
                    selected = c.id == currentId,
                    onClick = { if (enabled) onSelect(c.id) },
                    badge = {
                        Box {
                            IconButton(onClick = { menu = true }, enabled = enabled) {
                                Icon(Icons.Default.MoreVert, "Opciones")
                            }
                            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                                DropdownMenuItem(
                                    text = { Text("Renombrar") },
                                    onClick = { menu = false; renaming = c },
                                )
                                DropdownMenuItem(
                                    text = { Text("Borrar") },
                                    onClick = { menu = false; deleting = c },
                                )
                            }
                        }
                    },
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
            }
        }
    }

    renaming?.let { c ->
        var title by remember(c.id) { mutableStateOf(c.title) }
        AlertDialog(
            onDismissRequest = { renaming = null },
            title = { Text("Renombrar chat") },
            text = { OutlinedTextField(title, { title = it }, singleLine = true) },
            confirmButton = {
                TextButton(onClick = { onRename(c.id, title); renaming = null }, enabled = title.isNotBlank()) {
                    Text("Guardar")
                }
            },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text("Cancelar") } },
        )
    }
    deleting?.let { c ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("¿Borrar chat?") },
            text = { Text("\"${c.title}\" se borrará permanentemente.") },
            confirmButton = {
                TextButton(onClick = { onDelete(c.id); deleting = null }) { Text("Borrar") }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancelar") } },
        )
    }
}

private fun modelLabel(m: ModelState) = when (m) {
    ModelState.None -> "Sin modelo"
    is ModelState.Importing -> "Importando… ${(m.progress * 100).toInt()}%"
    is ModelState.Loading -> "Cargando ${m.name}…"
    is ModelState.Ready -> m.name
}

@Composable
private fun EmptyHint(model: ModelState, onPick: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        if (model is ModelState.None) {
            Text("Elige un archivo .gguf (p. ej. LFM3-1.7B Q4_K_M) para empezar.")
            Spacer(Modifier.size(16.dp))
            Button(onClick = onPick) { Text("Cargar modelo") }
        } else {
            Text("Escribe un mensaje.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun MessageList(messages: List<UiMessage>) {
    val listState = rememberLazyListState()
    val last = messages.lastOrNull()
    // Follow the streaming reply.
    // `last.id` changes when the stored messages replace the streamed ones.
    LaunchedEffect(messages.size, last?.id, last?.content?.length, last?.reasoning?.length) {
        listState.scrollToItem(messages.lastIndex, Int.MAX_VALUE)
    }
    LazyColumn(
        state = listState,
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxSize(),
    ) {
        items(messages, key = { it.id }) { MessageBubble(it) }
    }
}

@Composable
private fun ToolCard(m: UiMessage) {
    val colors = MaterialTheme.colorScheme
    val isResult = m.kind == Kind.ToolResult
    val bg = when {
        isResult && !m.toolOk -> colors.errorContainer
        isResult -> colors.secondaryContainer
        else -> colors.tertiaryContainer
    }
    var expanded by remember { mutableStateOf(false) }
    Column(
        Modifier
            .widthIn(max = 340.dp)
            .background(bg, RoundedCornerShape(10.dp))
            .clickable(enabled = isResult) { expanded = !expanded }
            .padding(horizontal = 10.dp, vertical = 6.dp),
    ) {
        if (isResult) {
            Text(
                "${if (m.toolOk) "✓" else "✗"} ${m.toolName ?: "herramienta"}",
                style = MaterialTheme.typography.labelMedium,
            )
            Text(
                m.content,
                fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.bodySmall,
                maxLines = if (expanded) Int.MAX_VALUE else 3,
                overflow = TextOverflow.Ellipsis,
            )
        } else {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Build, null, Modifier.size(14.dp))
                Spacer(Modifier.size(6.dp))
                Text(
                    if (m.toolName == "…") "Preparando herramienta…" else "${m.toolName}(${m.toolArgs.orEmpty()})",
                    style = MaterialTheme.typography.labelMedium,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }
    }
}

@Composable
private fun MessageBubble(m: UiMessage) {
    if (m.kind != Kind.Normal) {
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            // Text the model wrote before deciding to call the tool.
            if (m.kind == Kind.ToolCall && (m.content.isNotBlank() || m.reasoning.isNotBlank())) {
                MessageBubble(m.copy(kind = Kind.Normal, streaming = false, stats = null))
            }
            ToolCard(m)
        }
        return
    }
    val isUser = m.role == Role.User
    val colors = MaterialTheme.colorScheme
    Box(Modifier.fillMaxWidth(), contentAlignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart) {
        Column(
            Modifier
                .widthIn(max = 340.dp)
                .background(
                    if (isUser) colors.primaryContainer else colors.surfaceVariant,
                    RoundedCornerShape(16.dp),
                )
                .padding(12.dp),
        ) {
            if (m.reasoning.isNotBlank()) Reasoning(m.reasoning, expandedByDefault = m.streaming && m.content.isEmpty())
            when {
                m.content.isNotEmpty() && isUser -> SelectionContainer { Text(m.content) }
                m.content.isNotEmpty() -> MarkdownText(m.content)
                m.streaming && m.reasoning.isEmpty() -> CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
            }
            m.stats?.let {
                Text(it, style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant,
                    modifier = Modifier.padding(top = 6.dp))
            }
        }
    }
}

@Composable
private fun Reasoning(text: String, expandedByDefault: Boolean) {
    var expanded by remember { mutableStateOf(false) }
    val show = expanded || expandedByDefault
    Column(Modifier.padding(bottom = 6.dp)) {
        Text(
            if (show) "▾ Razonamiento" else "▸ Razonamiento",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.clickable { expanded = !show },
        )
        AnimatedVisibility(show) {
            Text(
                text,
                style = MaterialTheme.typography.bodySmall.copy(fontStyle = FontStyle.Italic),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun InputBar(enabled: Boolean, generating: Boolean, onSend: (String) -> Unit, onStop: () -> Unit) {
    var text by remember { mutableStateOf("") }
    Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.weight(1f),
            placeholder = { Text(if (enabled) "Mensaje" else "Carga un modelo") },
            enabled = enabled,
            maxLines = 6,
        )
        Spacer(Modifier.size(8.dp))
        if (generating) {
            FilledIconButton(onClick = onStop) { Icon(Icons.Default.Stop, "Detener") }
        } else {
            FilledIconButton(
                onClick = { onSend(text); text = "" },
                enabled = enabled && text.isNotBlank(),
            ) { Icon(Icons.AutoMirrored.Filled.Send, "Enviar") }
        }
    }
}

@Composable
private fun SettingsDialog(initial: Settings, onDismiss: () -> Unit, onSave: (Settings) -> Unit) {
    var s by remember { mutableStateOf(initial) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Ajustes") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                OutlinedTextField(
                    value = s.systemPrompt,
                    onValueChange = { s = s.copy(systemPrompt = it) },
                    label = { Text("Prompt de sistema") },
                    minLines = 3,
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.size(12.dp))
                Text("Temperatura: %.2f".format(s.temperature))
                Slider(s.temperature, { s = s.copy(temperature = it) }, valueRange = 0f..1.5f)
                Text("Máx. tokens de respuesta: ${s.maxTokens}")
                Slider(s.maxTokens.toFloat(), { s = s.copy(maxTokens = it.toInt()) }, valueRange = 128f..4096f, steps = 30)
                Text("Contexto: ${s.nCtx} tokens (recarga el modelo)")
                Slider(
                    s.nCtx.toFloat(),
                    { s = s.copy(nCtx = (it / 1024).toInt().coerceAtLeast(1) * 1024) },
                    valueRange = 1024f..16384f,
                    steps = 14,
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Razonamiento (thinking)", Modifier.weight(1f))
                    Switch(s.enableThinking, { s = s.copy(enableThinking = it) })
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Modo código automático")
                        Text(
                            "Respuesta en un solo bloque, muestreo preciso; Rust se revisa con syn",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(s.codeMode, { s = s.copy(codeMode = it) })
                }
                HorizontalDivider(Modifier.padding(vertical = 8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Herramientas")
                        Text(
                            "Sin red ni archivos; máx. 3 llamadas por mensaje",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(s.toolsEnabled, { s = s.copy(toolsEnabled = it) })
                }
                if (s.toolsEnabled) {
                    TOOL_LABELS.forEach { (name, label) ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(
                                checked = name in s.enabledTools,
                                onCheckedChange = { on ->
                                    s = s.copy(enabledTools = if (on) s.enabledTools + name else s.enabledTools - name)
                                },
                            )
                            Text(label, style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = { onSave(s) }) { Text("Guardar") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancelar") } },
    )
}
