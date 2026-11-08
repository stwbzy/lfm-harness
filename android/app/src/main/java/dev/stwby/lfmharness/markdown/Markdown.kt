package dev.stwby.lfmharness.markdown

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withLink
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import org.json.JSONArray
import org.json.JSONObject

/** JNI surface of `core/src/markdown.rs`. */
internal object NativeMarkdown {
    init {
        System.loadLibrary("harness_core")
    }

    @JvmStatic external fun parse(text: String): String
}

private const val BOLD = 1
private const val ITALIC = 2
private const val CODE = 4
private const val STRIKE = 8
private const val LINK = 16

data class MdSpan(val text: String, val flags: Int, val url: String?)

sealed interface MdBlock {
    data class Paragraph(val spans: List<MdSpan>) : MdBlock
    data class Heading(val level: Int, val spans: List<MdSpan>) : MdBlock
    data class Code(val lang: String, val text: String) : MdBlock
    data class ListBlock(val start: Long?, val items: List<List<MdBlock>>) : MdBlock
    data class Quote(val blocks: List<MdBlock>) : MdBlock
    data object Rule : MdBlock
    data class Table(val head: List<List<MdSpan>>, val rows: List<List<List<MdSpan>>>) : MdBlock
}

fun parseMarkdown(text: String): List<MdBlock> =
    runCatching { blocks(JSONArray(NativeMarkdown.parse(text))) }
        // Never lose the text if parsing fails for any reason.
        .getOrElse { listOf(MdBlock.Paragraph(listOf(MdSpan(text, 0, null)))) }

private fun blocks(a: JSONArray): List<MdBlock> = List(a.length()) { block(a.getJSONObject(it)) }

private fun spans(a: JSONArray): List<MdSpan> = List(a.length()) { i ->
    a.getJSONObject(i).run { MdSpan(getString("x"), optInt("f", 0), optString("u").ifEmpty { null }) }
}

private fun block(o: JSONObject): MdBlock = when (o.getString("t")) {
    "p" -> MdBlock.Paragraph(spans(o.getJSONArray("s")))
    "h" -> MdBlock.Heading(o.getInt("l"), spans(o.getJSONArray("s")))
    "code" -> MdBlock.Code(o.getString("lang"), o.getString("text"))
    "list" -> {
        val items = o.getJSONArray("items")
        MdBlock.ListBlock(
            if (o.isNull("start")) null else o.getLong("start"),
            List(items.length()) { blocks(items.getJSONArray(it)) },
        )
    }
    "quote" -> MdBlock.Quote(blocks(o.getJSONArray("b")))
    "hr" -> MdBlock.Rule
    "table" -> {
        val head = o.getJSONArray("head")
        val rows = o.getJSONArray("rows")
        MdBlock.Table(
            List(head.length()) { spans(head.getJSONArray(it)) },
            List(rows.length()) { r -> rows.getJSONArray(r).let { row -> List(row.length()) { spans(row.getJSONArray(it)) } } },
        )
    }
    else -> MdBlock.Paragraph(emptyList())
}

/** Renders Markdown (parsed in Rust) with Material 3 styles; text is selectable. */
@Composable
fun MarkdownText(text: String, modifier: Modifier = Modifier) {
    val blocks = remember(text) { parseMarkdown(text) }
    SelectionContainer(modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            blocks.forEach { Block(it) }
        }
    }
}

@Composable
private fun Block(b: MdBlock) {
    val type = MaterialTheme.typography
    val colors = MaterialTheme.colorScheme
    when (b) {
        is MdBlock.Paragraph -> Text(annotated(b.spans), style = type.bodyLarge)
        is MdBlock.Heading -> Text(
            annotated(b.spans),
            style = when (b.level) {
                1 -> type.titleLarge
                2 -> type.titleMedium
                else -> type.titleSmall
            }.copy(fontWeight = FontWeight.Bold),
        )
        is MdBlock.Code -> Column(
            Modifier
                .fillMaxWidth()
                .background(colors.surface.copy(alpha = 0.6f), RoundedCornerShape(8.dp))
                .padding(8.dp),
        ) {
            if (b.lang.isNotEmpty()) {
                Text(b.lang, style = type.labelSmall, color = colors.onSurfaceVariant)
            }
            Box(Modifier.horizontalScroll(rememberScrollState())) {
                Text(b.text, fontFamily = FontFamily.Monospace, style = type.bodySmall, softWrap = false)
            }
        }
        is MdBlock.ListBlock -> Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            b.items.forEachIndexed { i, item ->
                Row {
                    val marker = b.start?.let { "${it + i}." } ?: "•"
                    Text(marker, style = type.bodyLarge, modifier = Modifier.width(if (b.start != null) 28.dp else 16.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) { item.forEach { Block(it) } }
                }
            }
        }
        is MdBlock.Quote -> Row(Modifier.height(IntrinsicSize.Min)) {
            Box(Modifier.width(3.dp).fillMaxHeight().background(colors.primary.copy(alpha = 0.6f)))
            Column(Modifier.padding(start = 8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                b.blocks.forEach { Block(it) }
            }
        }
        MdBlock.Rule -> HorizontalDivider(Modifier.padding(vertical = 4.dp))
        is MdBlock.Table -> Column(Modifier.horizontalScroll(rememberScrollState())) {
            TableRow(b.head, header = true)
            HorizontalDivider()
            b.rows.forEach { TableRow(it, header = false) }
        }
    }
}

@Composable
private fun TableRow(cells: List<List<MdSpan>>, header: Boolean) {
    Row {
        cells.forEach { cell ->
            Text(
                annotated(cell),
                style = MaterialTheme.typography.bodyMedium.let { if (header) it.copy(fontWeight = FontWeight.Bold) else it },
                modifier = Modifier.width(120.dp).padding(4.dp),
            )
        }
    }
}

@Composable
private fun annotated(spans: List<MdSpan>): AnnotatedString {
    val colors = MaterialTheme.colorScheme
    val codeBg = colors.surface.copy(alpha = 0.6f)
    val linkColor = colors.primary
    return remember(spans, codeBg, linkColor) {
        buildAnnotatedString {
            spans.forEach { s -> appendSpan(s, codeBg, linkColor) }
        }
    }
}

private fun AnnotatedString.Builder.appendSpan(s: MdSpan, codeBg: Color, linkColor: Color) {
    val style = SpanStyle(
        fontWeight = if (s.flags and BOLD != 0) FontWeight.Bold else null,
        fontStyle = if (s.flags and ITALIC != 0) FontStyle.Italic else null,
        fontFamily = if (s.flags and CODE != 0) FontFamily.Monospace else null,
        background = if (s.flags and CODE != 0) codeBg else Color.Unspecified,
        textDecoration = if (s.flags and STRIKE != 0) TextDecoration.LineThrough else null,
    )
    if (s.flags and LINK != 0 && s.url != null) {
        val linkStyle = TextLinkStyles(SpanStyle(color = linkColor, textDecoration = TextDecoration.Underline))
        withLink(LinkAnnotation.Url(s.url, linkStyle)) { withStyle(style) { append(s.text) } }
    } else {
        withStyle(style) { append(s.text) }
    }
}

