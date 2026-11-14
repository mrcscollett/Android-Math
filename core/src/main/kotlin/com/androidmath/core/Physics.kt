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

/** The physics systems the engine offers; the app builds its forms from this. */
@Serializable
data class PhysicsCatalog(
    val ok: Boolean = true,
    val scenarios: List<Scenario> = emptyList(),
    /** Names of scenarios that are planned but not built yet. */
    val upcoming: List<String> = emptyList(),
) {
    /** The system with this id. Ids of the earlier situation-based models open the system that replaced them. */
    fun scenario(id: String): Scenario? {
        val current = LEGACY_IDS[id] ?: id
        return scenarios.firstOrNull { it.id == current }
    }

    /** The switches that turn the general system into the old model [id] (empty for a current id). */
    fun legacyEffects(id: String): Map<String, String> = LEGACY_EFFECTS[id] ?: emptyMap()

    companion object {
        /** Old model id to the general system that now covers it (saved history, old links). */
        val LEGACY_IDS: Map<String, String> = mapOf(
            "falling" to "line",
            "spring" to "line",
            "projectile" to "plane",
            "pendulum" to "rotation",
            "cooling" to "change",
            "decay" to "change",
            "rc" to "circuit",
            "rl" to "circuit",
        )

        private val LEGACY_EFFECTS: Map<String, Map<String, String>> = mapOf(
            "falling" to mapOf("gravity" to "vertical"),
            "spring" to mapOf("spring" to "on"),
            "projectile" to mapOf("gravity" to "on"),
            "pendulum" to mapOf("gravity" to "point"),
            "cooling" to mapOf("what" to "temp", "law" to "level"),
            "decay" to mapOf("what" to "amt", "law" to "decay"),
            "rc" to mapOf("res" to "on", "cap" to "on"),
            "rl" to mapOf("res" to "on", "ind" to "on"),
        )
    }
}

@Serializable
data class Scenario(
    val id: String,
    val title: String,
    val description: String = "",
    val law: String = "",
    val fields: List<PhysicsField> = emptyList(),
    val effects: List<PhysicsEffect> = emptyList(),
    val presets: List<PhysicsPreset> = emptyList(),
) {
    fun field(id: String): PhysicsField? = fields.firstOrNull { it.id == id }

    /** The option each effect starts on. */
    fun defaultEffects(): Map<String, String> = effects.associate { it.id to it.default }

    /**
     * The variables that exist for the chosen effects, in the model's order:
     * every variable that does not belong to an effect, plus the variables of
     * each effect option that is switched on. A variable can also carry its
     * own conditions on the switches ([PhysicsField.show], [PhysicsField.hide]).
     */
    fun visibleFields(chosen: Map<String, String>): List<PhysicsField> {
        val current = HashMap<String, String>()
        val wanted = HashSet<String>()
        for (e in effects) {
            val optionId = chosen[e.id] ?: e.default
            current[e.id] = optionId
            val option = e.options.firstOrNull { it.id == optionId }
            if (option != null) wanted.addAll(option.fields)
        }
        return fields.filter { f ->
            val listed = f.group != "effect" || f.id in wanted
            val shown = f.show.all { it.holds(current) }
            val hidden = f.hide.isNotEmpty() && f.hide.all { it.holds(current) }
            listed && shown && !hidden
        }
    }

    /** [visibleFields] split into titled sections, in the order the form shows them. */
    fun sections(chosen: Map<String, String>): List<FieldSection> {
        val visible = visibleFields(chosen)
        val out = ArrayList<FieldSection>()
        for ((group, title) in GROUPS) {
            val rows = visible.filter { it.group == group }
            if (rows.isNotEmpty()) out.add(FieldSection(group, title, rows))
        }
        val known = GROUPS.map { it.first }.toSet()
        val rest = visible.filter { it.group !in known }
        if (rest.isNotEmpty()) out.add(FieldSection("other", "Other", rest))
        return out
    }

    companion object {
        /** Group id to section title, in display order. */
        val GROUPS: List<Pair<String, String>> = listOf(
            "setup" to "The setup",
            "initial" to "At the start",
            "effect" to "Added effects",
            "moment" to "At a moment",
            "moment2" to "At a second moment",
            "derived" to "Other quantities",
        )
    }
}

/** One titled block of variable rows in the form. */
data class FieldSection(val group: String, val title: String, val fields: List<PhysicsField>)

@Serializable
data class PhysicsField(
    val id: String,
    val label: String,
    val symbol: String = "",
    val quantity: String = "",
    val units: List<UnitChoice> = emptyList(),
    /** Value text (null for an empty box) and unit id shown when the form opens. */
    @SerialName("default") val initial: List<String?>? = null,
    /** "setup", "initial", "effect" (exists only with its effect), "moment", "moment2" or "derived". */
    val group: String = "setup",
    val hint: String = "",
    /** The row exists only while every one of these conditions on the switches holds. */
    val show: List<FieldCondition> = emptyList(),
    /** The row is removed while every one of these conditions holds (ignored when empty). */
    val hide: List<FieldCondition> = emptyList(),
) {
    val initialValue: String get() = initial?.getOrNull(0) ?: ""
    val initialUnit: String get() = initial?.getOrNull(1) ?: (units.firstOrNull()?.id ?: "")

    fun unitLabel(unitId: String): String = units.firstOrNull { it.id == unitId }?.label ?: unitId
}

/** A condition on one switch: its chosen option is one of [options]. */
@Serializable
data class FieldCondition(
    val effect: String,
    @SerialName("in") val options: List<String> = emptyList(),
) {
    fun holds(chosen: Map<String, String>): Boolean = chosen[effect] in options
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
data class PhysicsPreset(
    val id: String,
    val label: String,
    val description: String = "",
    /** Field id to [value text, unit id]; an empty value leaves the box blank but sets its unit. */
    val values: Map<String, List<String>> = emptyMap(),
    val effects: Map<String, String> = emptyMap(),
)

/** What a solved scenario gives the result screen besides its steps. */
@Serializable
data class PhysicsData(
    val scenario: String = "",
    val headline: Headline = Headline(),
    /** The unknowns that were worked out, in the order they were solved. */
    val found: List<FoundValue> = emptyList(),
    /** The unknowns the entered values do not determine, each with what else is needed. */
    val missing: List<MissingValue> = emptyList(),
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
data class FoundValue(
    val id: String = "",
    val label: String = "",
    /** LaTeX symbol. */
    val symbol: String = "",
    /** The value with its unit, ready to show. */
    val text: String = "",
    /** The value in SI units. */
    val value: Double = 0.0,
)

@Serializable
data class MissingValue(val id: String = "", val label: String = "", val reason: String = "")

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

/** What the user entered for a model, as sent to the engine and kept in the history. */
data class PhysicsInputs(
    /** Field id to (value text, unit id); a blank value means unknown. */
    val values: Map<String, Pair<String, String>>,
    val effects: Map<String, String>,
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
    }.toString()

    /** The variables with a value typed in, among those that exist for the chosen effects. */
    fun knowns(scenario: Scenario): List<PhysicsField> =
        scenario.visibleFields(effects).filter { !values[it.id]?.first.isNullOrBlank() }

    /** A one-line description for the history list: the entered values, in field order. */
    fun label(scenario: Scenario): String {
        val parts = ArrayList<String>()
        for (f in knowns(scenario)) {
            val entry = values[f.id] ?: continue
            val unit = f.unitLabel(entry.second)
            parts.add(f.label + " " + entry.first.trim() + (if (unit.isEmpty()) "" else " $unit"))
        }
        return if (parts.isEmpty()) "No values entered" else "Known: " + parts.joinToString(", ")
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
            PhysicsInputs(values, effects)
        } catch (e: Exception) {
            null
        }
    }
}
