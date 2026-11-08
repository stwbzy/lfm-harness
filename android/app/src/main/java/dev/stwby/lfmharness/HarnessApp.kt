package dev.stwby.lfmharness

import android.app.Application
import dev.stwby.lfmharness.data.ChatRepository
import java.io.File

class HarnessApp : Application() {
    val chats: ChatRepository by lazy { ChatRepository(File(filesDir, "chats.db").absolutePath) }
}
