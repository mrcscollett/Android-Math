package com.androidmath.app

import android.content.Context
import com.androidmath.core.EngineJson
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
        context: Context, f: String, g: String, x0: String, y0: String,
    ): Pair<String, SolveResult> = call(context, "system", f, g, x0, y0)

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
