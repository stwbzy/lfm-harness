package dev.stwby.lfmharness.engine

/** Receives streamed output from [NativeEngine.generate]. Called on the generating thread. */
interface TokenCallback {
    fun onText(text: String)
    fun onThinkEnd()
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

    /** Blocks until done; returns stats JSON. Throws RuntimeException on failure. */
    @JvmStatic external fun generate(
        handle: Long,
        messagesJson: String,
        paramsJson: String,
        callback: TokenCallback,
    ): String
}
