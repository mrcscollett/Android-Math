package com.androidmath.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json

/**
 * One solved problem: enough to reopen it.
 *
 * [mode] is the section key ("first", "second", "numeric", "laplace",
 * "inverse", "system", "fourier", "boundary", "eigen") and [fields] holds
 * every input box by name. [key] identifies the problem, so solving the
 * same inputs again moves the entry to the top instead of adding a copy.
 */
@Serializable
data class HistoryEntry(
    val key: String,
    val mode: String,
    val section: String,
    val label: String,
    val fields: Map<String, String>,
    val time: Long,
    val favorite: Boolean = false,
)

/** List logic for the history, free of Android types so it can be unit tested. */
object History {
    const val MAX_RECENT = 50

    /** Identity of a problem: its section and its non-empty inputs, order-independent. */
    fun keyOf(mode: String, fields: Map<String, String>): String =
        mode + "|" + fields.filterValues { it.isNotBlank() }.toSortedMap()
            .entries.joinToString("|") { it.key + "=" + it.value.trim() }

    fun entry(
        mode: String, section: String, label: String, fields: Map<String, String>, time: Long,
    ): HistoryEntry = HistoryEntry(
        keyOf(mode, fields), mode, section, label, fields.filterValues { it.isNotBlank() }, time,
    )

    /**
     * Puts [entry] first. An existing entry with the same key is replaced
     * but keeps its favorite mark. Only the newest [MAX_RECENT] entries that
     * are not favorites are kept; favorites are never dropped.
     */
    fun add(list: List<HistoryEntry>, entry: HistoryEntry): List<HistoryEntry> {
        val old = list.firstOrNull { it.key == entry.key }
        val merged = entry.copy(favorite = old?.favorite ?: entry.favorite)
        val rest = list.filter { it.key != entry.key }
        var recent = 0
        return (listOf(merged) + rest).filter { it.favorite || recent++ < MAX_RECENT }
    }

    fun toggleFavorite(list: List<HistoryEntry>, key: String): List<HistoryEntry> =
        list.map { if (it.key == key) it.copy(favorite = !it.favorite) else it }

    /** Removes everything that is not a favorite. */
    fun clearRecent(list: List<HistoryEntry>): List<HistoryEntry> = list.filter { it.favorite }

    fun favorites(list: List<HistoryEntry>): List<HistoryEntry> = list.filter { it.favorite }

    fun recent(list: List<HistoryEntry>): List<HistoryEntry> = list.filter { !it.favorite }

    private val json = Json { ignoreUnknownKeys = true }
    private val serializer = ListSerializer(HistoryEntry.serializer())

    fun encode(list: List<HistoryEntry>): String = json.encodeToString(serializer, list)

    /** Reads a stored list; anything unreadable gives an empty history rather than a crash. */
    fun decode(text: String?): List<HistoryEntry> {
        if (text.isNullOrBlank()) return emptyList()
        return try {
            json.decodeFromString(serializer, text)
        } catch (e: Exception) {
            emptyList()
        }
    }
}
