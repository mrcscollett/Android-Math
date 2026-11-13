package com.androidmath.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** The parts of the engine's JSON the native UI needs. Steps are rendered by the web view. */
@Serializable
data class SolveResult(
    val ok: Boolean,
    val error: String? = null,
    val indep: String = "x",
    @SerialName("input_latex") val inputLatex: String = "",
    @SerialName("normal_latex") val normalLatex: String = "",
    @SerialName("rhs_expr") val rhsExpr: String? = null,
    val ic: InitialCondition? = null,
    val summary: String = "",
    @SerialName("other_types") val otherTypes: List<String> = emptyList(),
    val methods: List<Method> = emptyList(),
) {
    val applicableMethods: List<Method> get() = methods.filter { it.applicable }
}

@Serializable
data class InitialCondition(val x0: Double, val y0: Double, val latex: String = "")

@Serializable
data class Method(
    val id: String,
    val name: String,
    val applicable: Boolean,
    val reason: String = "",
    @SerialName("general_latex") val generalLatex: String? = null,
    @SerialName("particular_latex") val particularLatex: String? = null,
    val plot: PlotExprs? = null,
    val verified: Boolean? = null,
    @SerialName("verify_note") val verifyNote: String? = null,
)

@Serializable
data class PlotExprs(
    @SerialName("general_expr") val generalExpr: String? = null,
    @SerialName("particular_expr") val particularExpr: String? = null,
)

@Serializable
data class PreviewResult(val ok: Boolean, val latex: String = "", val error: String? = null)

object EngineJson {
    private val json = Json { ignoreUnknownKeys = true }

    fun solveResult(text: String): SolveResult = try {
        json.decodeFromString(SolveResult.serializer(), text)
    } catch (e: Exception) {
        SolveResult(ok = false, error = "The solver returned something unexpected.")
    }

    fun preview(text: String): PreviewResult = try {
        json.decodeFromString(PreviewResult.serializer(), text)
    } catch (e: Exception) {
        PreviewResult(ok = false, error = "The solver returned something unexpected.")
    }
}
