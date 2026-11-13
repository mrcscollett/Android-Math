package com.androidmath.app

import android.content.Context
import com.androidmath.core.History
import com.androidmath.core.HistoryEntry

/** Reads and writes the history list. The list logic itself lives in [History]. */
class HistoryStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("history", Context.MODE_PRIVATE)

    fun load(): List<HistoryEntry> = History.decode(prefs.getString(KEY, null))

    fun save(list: List<HistoryEntry>) {
        prefs.edit().putString(KEY, History.encode(list)).apply()
    }

    private companion object {
        const val KEY = "entries"
    }
}
