package com.androidmath.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

/** The physics scenarios the engine offers; the app builds its forms from this. */
@Serializable
data class PhysicsCatalog(
    val ok: Boolean = true,
    val scenarios: List<Scenario> = emptyList(),
    /** Names of scenarios that are planned but not built yet. */
    val upcoming: List<String> = emptyList(),
) {
    fun scenario(id: String): Scenario? = scenarios.firstOrNull { it.id == id }
}

@Serializable
data class Scenario(
    val id: String,
    val title: String,
    val description: String = "",
    val law: String = "",
    val fields: List<PhysicsField> = emptyList(),
    val effects: List<PhysicsEffect> = emptyList(),
    val questions: List<PhysicsQuestion> = emptyList(),
    val presets: List<PhysicsPreset> = emptyList(),
) {
    fun question(id: String): PhysicsQuestion? = questions.firstOrNull { it.id == id }

    /** The option each effect starts on. */
    fun defaultEffects(): Map<String, String> = effects.associate { it.id to it.default }

    /**
     * The boxes to show for the chosen effects and question, in the scenario's
     * order: the base fields the question uses, the fields of every active
     * effect option, and the question's own parameters.
     */
    fun visibleFields(chosen: Map<String, String>, questionId: String): List<PhysicsField> {
        val q = question(questionId)
        val wanted = HashSet<String>()
        for (f in fields) if (f.group == "base") wanted.add(f.id)
        for (e in effects) {
            val option = e.options.firstOrNull { it.id == (chosen[e.id] ?: e.default) }
            if (option != null) wanted.addAll(option.fields)
        }
        if (q != null) {
            wanted.removeAll(q.hides.toSet())
            wanted.addAll(q.params)
        }
        return fields.filter { it.id in wanted }
    }
}

@Serializable
data class PhysicsField(
    val id: String,
    val label: String,
    val symbol: String = "",
    val quantity: String = "",
    val units: List<UnitChoice> = emptyList(),
    /** Value text and unit id shown when the form opens, or null for an empty box. */
    @SerialName("default") val initial: List<String>? = null,
    /** "base" (always shown), "effect" (shown with its effect) or "param" (shown with its question). */
    val group: String = "base",
    val hint: String = "",
) {
    val initialValue: String get() = initial?.getOrNull(0) ?: ""
    val initialUnit: String get() = initial?.getOrNull(1) ?: (units.firstOrNull()?.id ?: "")

    fun unitLabel(unitId: String): String = units.firstOrNull { it.id == unitId }?.label ?: unitId
}

@Serializable
data class UnitChoice(val id: String, val label: String = "")

@Serializable
data class PhysicsEffect(
    val id: String,
    val label: String,
    val description: String = "",
    /** "switch" (off or on) or "choice" (one of several options). */
    val kind: String = "switch",
    val default: String = "off",
    val options: List<EffectOption> = emptyList(),
)

@Serializable
data class EffectOption(val id: String, val label: String = "", val fields: List<String> = emptyList())

@Serializable
data class PhysicsQuestion(
    val id: String,
    val label: String,
    val params: List<String> = emptyList(),
    /** Fields this question works out, shown as "to find" instead of as inputs. */
    val finds: List<String> = emptyList(),
    val hides: List<String> = emptyList(),
)

@Serializable
data class PhysicsPreset(
    val id: String,
    val label: String,
    val description: String = "",
    /** Field id to [value text, unit id]. */
    val values: Map<String, List<String>> = emptyMap(),
    val effects: Map<String, String> = emptyMap(),
    val question: String? = null,
)

/** What a solved scenario gives the result screen besides its steps. */
@Serializable
data class PhysicsData(
    val scenario: String = "",
    val headline: Headline = Headline(),
    @SerialName("x_label") val xLabel: String = "",
    @SerialName("y_label") val yLabel: String = "",
    /** The solution as [t, value] samples. */
    val curve: List<List<Double>> = emptyList(),
    @SerialName("curve_label") val curveLabel: String = "",
    /** The same quantity with every optional effect switched off (empty when none is on). */
    val ideal: List<List<Double>> = emptyList(),
    @SerialName("ideal_label") val idealLabel: String = "",
    val marker: PhysicsMarker? = null,
    @SerialName("t_max") val tMax: Double = 1.0,
    val animation: PhysicsAnimation? = null,
) {
    /** Lowest and highest value over both curves and the marker, widened if flat. */
    fun valueRange(): Pair<Double, Double> {
        var lo = Double.POSITIVE_INFINITY
        var hi = Double.NEGATIVE_INFINITY
        for (p in curve + ideal) {
            val v = p.getOrElse(1) { Double.NaN }
            if (v.isFinite()) {
                lo = minOf(lo, v); hi = maxOf(hi, v)
            }
        }
        marker?.let { lo = minOf(lo, it.y); hi = maxOf(hi, it.y) }
        if (lo > hi) return -1.0 to 1.0
        if (hi - lo < 1e-12 * maxOf(1.0, kotlin.math.abs(hi))) return (lo - 1.0) to (hi + 1.0)
        return lo to hi
    }
}

@Serializable
data class Headline(val label: String = "", val text: String = "")

@Serializable
data class PhysicsMarker(val t: Double = 0.0, val y: Double = 0.0, val label: String = "")

@Serializable
data class PhysicsAnimation(
    /** "fall", "spring" or "capacitor": which picture to draw. */
    val kind: String = "",
    val points: List<List<Double>> = emptyList(),
    val min: Double = 0.0,
    val max: Double = 1.0,
    val label: String = "",
) {
    /** The animated quantity at time [t], by linear interpolation of the samples. */
    fun valueAt(t: Double): Double = Samples.valueAt(points, t)

    /** [valueAt] scaled to 0..1 between [min] and [max]. */
    fun fractionAt(t: Double): Double {
        val span = max - min
        if (!(span > 0)) return 0.0
        return ((valueAt(t) - min) / span).coerceIn(0.0, 1.0)
    }
}

object Samples {
    /** Linear interpolation in a list of [t, value] pairs sorted by t. */
    fun valueAt(points: List<List<Double>>, t: Double): Double {
        if (points.isEmpty()) return 0.0
        if (t <= points.first()[0]) return points.first()[1]
        if (t >= points.last()[0]) return points.last()[1]
        var lo = 0
        var hi = points.size - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) / 2
            if (points[mid][0] <= t) lo = mid else hi = mid
        }
        val t0 = points[lo][0]
        val t1 = points[hi][0]
        if (t1 == t0) return points[lo][1]
        val w = (t - t0) / (t1 - t0)
        return points[lo][1] + w * (points[hi][1] - points[lo][1])
    }

    /** [t, value] pairs flattened to [t0, v0, t1, v1, ...]. */
    fun flatten(points: List<List<Double>>): DoubleArray {
        val out = DoubleArray(points.size * 2)
        for ((i, p) in points.withIndex()) {
            out[2 * i] = p.getOrElse(0) { Double.NaN }
            out[2 * i + 1] = p.getOrElse(1) { Double.NaN }
        }
        return out
    }
}

/** What the user entered for a scenario, as sent to the engine and kept in the history. */
data class PhysicsInputs(
    /** Field id to (value text, unit id); a blank value means unknown. */
    val values: Map<String, Pair<String, String>>,
    val effects: Map<String, String>,
    val question: String,
) {
    fun encode(): String = buildJsonObject {
        putJsonObject("values") {
            for ((id, entry) in values) {
                putJsonObject(id) {
                    put("value", entry.first)
                    put("unit", entry.second)
                }
            }
        }
        putJsonObject("effects") { for ((id, option) in effects) put(id, option) }
        put("question", question)
    }.toString()

    /** A one-line description for the history list: the entered values, in field order. */
    fun label(scenario: Scenario): String {
        val q = scenario.question(question)?.label ?: question
        val parts = ArrayList<String>()
        for (f in scenario.visibleFields(effects, question)) {
            val entry = values[f.id] ?: continue
            if (entry.first.isBlank()) continue
            val unit = f.unitLabel(entry.second)
            parts.add(f.label + " " + entry.first + (if (unit.isEmpty()) "" else " $unit"))
        }
        return q + ": " + parts.joinToString(", ")
    }

    companion object {
        /** Reads [encode]'s output; returns null if the text is not usable. */
        fun decode(text: String): PhysicsInputs? = try {
            val root = Json.parseToJsonElement(text).jsonObject
            val values = LinkedHashMap<String, Pair<String, String>>()
            (root["values"] as? JsonObject)?.forEach { (id, entry) ->
                val obj = entry.jsonObject
                values[id] = (obj["value"] as? JsonPrimitive)?.content.orEmpty() to
                    (obj["unit"] as? JsonPrimitive)?.content.orEmpty()
            }
            val effects = LinkedHashMap<String, String>()
            (root["effects"] as? JsonObject)?.forEach { (id, option) -> effects[id] = option.jsonPrimitive.content }
            PhysicsInputs(values, effects, (root["question"] as? JsonPrimitive)?.content.orEmpty())
        } catch (e: Exception) {
            null
        }
    }
}
