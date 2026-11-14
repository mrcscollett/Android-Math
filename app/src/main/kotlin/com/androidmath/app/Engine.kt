package com.androidmath.app

import android.content.Context
import com.androidmath.core.EngineJson
import com.androidmath.core.PhysicsCatalog
import com.androidmath.core.PreviewResult
import com.androidmath.core.SolveResult
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/** Bridge to the Python solver. All calls run on one dedicated thread. */
object Engine {
    // SymPy recurses deeply, so the engine thread gets a large stack.
    private val dispatcher = Executors.newSingleThreadExecutor { task ->
        Thread(null, task, "engine", 16L * 1024 * 1024)
    }.asCoroutineDispatcher()

    private fun module(context: Context): PyObject {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context.applicationContext))
        }
        return Python.getInstance().getModule("deengine")
    }

    /** Imports SymPy and runs one solve so the first real request is quick. */
    suspend fun warmUp(context: Context): Boolean = withContext(dispatcher) {
        try {
            module(context).callAttr("warmup")
            true
        } catch (e: Throwable) {
            android.util.Log.e("Engine", "warm-up failed", e)
            false
        }
    }

    /** [kind] is "ode" for an equation, "forward" for f(t), "inverse" for F(s). */
    suspend fun preview(context: Context, text: String, kind: String): PreviewResult = withContext(dispatcher) {
        try {
            EngineJson.preview(module(context).callAttr("preview", text, kind).toString())
        } catch (e: Throwable) {
            PreviewResult(ok = false, error = "The solver could not start.")
        }
    }

    /** Returns the raw JSON (for the steps view) and its parsed form. */
    suspend fun solve(
        context: Context, text: String, x0: String, y0: String, yPrime0: String,
    ): Pair<String, SolveResult> = call(context, "solve", text, x0, y0, yPrime0)

    /** Euler, improved Euler and RK4 tables for y' = f(x, y), y(x0) = y0. */
    suspend fun numeric(
        context: Context, text: String, x0: String, y0: String, step: String, count: String,
    ): Pair<String, SolveResult> = call(context, "numeric", text, x0, y0, step, count)

    /** Laplace transform of f(t), or the inverse transform of F(s). */
    suspend fun transform(context: Context, text: String, inverse: Boolean): Pair<String, SolveResult> =
        call(context, "transform", text, if (inverse) "inverse" else "forward")

    /** Solves or analyses the planar system x' = f(x, y), y' = g(x, y). */
    suspend fun system(
        context: Context, f: String, g: String, x0: String, y0: String, h: String, z0: String,
    ): Pair<String, SolveResult> = call(context, "system", f, g, x0, y0, h, z0)

    /** Fourier series of f on (-L, L) and the half-range series on (0, L). */
    suspend fun fourier(
        context: Context, f: String, left: String, length: String,
    ): Pair<String, SolveResult> = call(context, "fourier", f, left, length)

    /** Boundary value problem; each kind is "y" or "dy". */
    suspend fun boundary(
        context: Context, text: String,
        a: String, aKind: String, aValue: String,
        b: String, bKind: String, bValue: String,
    ): Pair<String, SolveResult> = call(context, "boundary", text, a, aKind, aValue, b, bKind, bValue)

    /** y'' + lambda y = 0 on [0, L]; [kind] is dd, nn, dn or nd. */
    suspend fun eigenProblem(context: Context, kind: String, length: String): Pair<String, SolveResult> =
        call(context, "eigen_problem", kind, length)

    /**
     * Heat, wave or Laplace equation. [kind] is "heat", "wave" or "laplace";
     * p1 and p2 are L and k, L and c, or a and b; [bc] is the boundary-condition
     * code (heat) or the side carrying the data (Laplace).
     */
    suspend fun pde(
        context: Context, kind: String, f: String, g: String, p1: String, p2: String,
        bc: String, t1: String, t2: String,
    ): Pair<String, SolveResult> = call(context, "pde", kind, f, g, p1, p2, bc, t1, t2)

    /**
     * A math tool. [name] is linalg, calculus, series or complex; [operation]
     * picks what it does; [extra1] and [extra2] are its small boxes (the
     * right-hand side b, limits of integration, a centre and an order, or n).
     */
    suspend fun tool(
        context: Context, name: String, operation: String, text: String, extra1: String, extra2: String,
    ): Pair<String, SolveResult> = call(context, "tool", name, operation, text, extra1, extra2)

    /**
     * A physics scenario. [inputs] is the form as JSON: the values with their
     * units, the option chosen for each effect, and the question.
     */
    suspend fun physics(context: Context, scenario: String, inputs: String): Pair<String, SolveResult> =
        call(context, "physics", scenario, inputs)

    /** The scenarios with their fields, effects, questions and presets. */
    suspend fun physicsCatalog(context: Context): PhysicsCatalog = withContext(dispatcher) {
        try {
            EngineJson.catalog(module(context).callAttr("physics_catalog").toString())
        } catch (e: Throwable) {
            android.util.Log.e("Engine", "physics_catalog failed", e)
            PhysicsCatalog(ok = false)
        }
    }

    private suspend fun call(
        context: Context, function: String, vararg args: String,
    ): Pair<String, SolveResult> = withContext(dispatcher) {
        try {
            val json = module(context).callAttr(function, *args).toString()
            json to EngineJson.solveResult(json)
        } catch (e: Throwable) {
            android.util.Log.e("Engine", "$function failed", e)
            "" to SolveResult(ok = false, error = "The solver could not start.")
        }
    }
}
