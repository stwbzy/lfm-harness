package dev.stwby.lfmharness.engine

/** Receives a turn's output from [NativeEngine.generate]. Called on the generating thread. */
interface TurnCallback {
    fun onText(text: String)
    fun onThinkEnd()
    fun onToolCallStarted()
    fun onToolCall(name: String, argsJson: String)
    fun onToolResult(name: String, ok: Boolean, content: String)
    /** A new assistant message starts (after a tool result). */
    fun onNewMessage()
    /** Asked before tools that modify data; may block until the user answers. */
    fun confirmTool(name: String, argsJson: String): Boolean
}

/** Raw JNI surface of the Rust core (`core/src/jni.rs`). Use [LlmEngine] instead. */
internal object NativeEngine {
    init {
        System.loadLibrary("harness_core")
    }

    @JvmStatic external fun load(path: String, nCtx: Int, nThreads: Int): Long
    @JvmStatic external fun free(handle: Long)
    @JvmStatic external fun cancel(handle: Long)
    @JvmStatic external fun reset(handle: Long)

    /** Runs one turn (tool loop included); blocks until done and returns JSON. */
    @JvmStatic external fun generate(
        handle: Long,
        storeHandle: Long,
        messagesJson: String,
        paramsJson: String,
        callback: TurnCallback,
    ): String
}
