package com.androidmath.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** The parts of the engine's JSON the native UI needs. Steps are rendered by the web view. */
@Serializable
data class SolveResult(
    val ok: Boolean,
    val error: String? = null,
    /** "symbolic" for step-by-step solutions, "numeric" for the numerical methods. */
    val kind: String = "symbolic",
    val order: Int = 1,
    val indep: String = "x",
    @SerialName("input_latex") val inputLatex: String = "",
    @SerialName("normal_latex") val normalLatex: String = "",
    @SerialName("rhs_expr") val rhsExpr: String? = null,
    val ic: InitialCondition? = null,
    val summary: String = "",
    @SerialName("other_types") val otherTypes: List<String> = emptyList(),
    val methods: List<Method> = emptyList(),
    /** For systems: the right-hand sides of x' and y' as plot expressions in x and y. */
    val field: VectorField? = null,
    val equilibria: List<Equilibrium> = emptyList(),
    /** Half-length of the interval of interest (Fourier, boundary, eigenvalue problems). */
    val span: Double? = null,
    /** For kind "pde": "heat", "wave" or "laplace". */
    val pde: String = "",
    /** For PDEs: the region the solution is drawn on and its range of values. */
    val domain: Domain? = null,
    /** For heat and wave problems: the initial profile f(x). */
    @SerialName("initial_expr") val initialExpr: String? = null,
    /** True when the solution is a function of two space variables, drawn as a heat map. */
    val surface: Boolean = false,
) {
    val applicableMethods: List<Method> get() = methods.filter { it.applicable }
}

@Serializable
data class Domain(
    @SerialName("x_min") val xMin: Double = 0.0,
    @SerialName("x_max") val xMax: Double = 1.0,
    @SerialName("y_min") val yMin: Double = 0.0,
    @SerialName("y_max") val yMax: Double = 1.0,
    /** Length of the time interval worth showing (heat and wave problems). */
    @SerialName("t_max") val tMax: Double = 0.0,
    @SerialName("u_min") val uMin: Double = -1.0,
    @SerialName("u_max") val uMax: Double = 1.0,
)

@Serializable
data class VectorField(val f: String? = null, val g: String? = null)

@Serializable
data class Equilibrium(val x: Double, val y: Double, val label: String = "")

@Serializable
data class InitialCondition(
    val x0: Double,
    val y0: Double,
    /** y'(x0), present for second-order problems. */
    val y1: Double? = null,
    val latex: String = "",
)

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
    /** For systems: x(t) and y(t) of the solution, written with x standing for t. */
    @SerialName("x_expr") val xExpr: String? = null,
    @SerialName("y_expr") val yExpr: String? = null,
    /** Points computed by a numerical method, as [x, y] pairs. */
    val points: List<List<Double>> = emptyList(),
    /** Curves to step through with a slider: Fourier partial sums or eigenfunctions. */
    val partials: List<PartialSum> = emptyList(),
) {
    /** The points flattened to [x0, y0, x1, y1, ...]. */
    fun pointArray(): DoubleArray {
        val out = DoubleArray(points.size * 2)
        for ((i, p) in points.withIndex()) {
            out[2 * i] = p.getOrElse(0) { Double.NaN }
            out[2 * i + 1] = p.getOrElse(1) { Double.NaN }
        }
        return out
    }
}

/** One selectable curve: [n] terms of a Fourier series, or the n-th eigenfunction. */
@Serializable
data class PartialSum(val n: Int, val expr: String)

@Serializable
data class PreviewResult(
    val ok: Boolean,
    val latex: String = "",
    val error: String? = null,
    val order: Int = 1,
)

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
