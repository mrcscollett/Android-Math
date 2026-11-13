package com.androidmath.core

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.min

/** Numerical solution of y' = f(x, y) for drawing solution curves. */
object Ode {
    /**
     * Integrates from (x0, y0) in both directions with classical RK4 and
     * returns the curve as [x0, y0, x1, y1, ...] sorted by x.
     *
     * Integration stops at the x-range, or when the solution leaves a band
     * of three window-heights around the window (a blow-up) or becomes
     * undefined. Steps are subdivided where the slope is steep.
     */
    fun solve(
        f: Expr, x0: Double, y0: Double,
        xMin: Double, xMax: Double, yMin: Double, yMax: Double,
        steps: Int = 600,
    ): DoubleArray {
        val backward = trace(f, x0, y0, xMin, xMax, yMin, yMax, steps, -1.0)
        val forward = trace(f, x0, y0, xMin, xMax, yMin, yMax, steps, 1.0)
        val out = DoubleArray(backward.size + forward.size + 2)
        var k = 0
        var i = backward.size - 2
        while (i >= 0) {
            out[k++] = backward[i]; out[k++] = backward[i + 1]
            i -= 2
        }
        out[k++] = x0; out[k++] = y0
        forward.copyInto(out, k)
        return out
    }

    private fun trace(
        f: Expr, x0: Double, y0: Double,
        xMin: Double, xMax: Double, yMin: Double, yMax: Double,
        steps: Int, direction: Double,
    ): DoubleArray {
        val h = direction * (xMax - xMin) / steps
        val height = yMax - yMin
        val lo = yMin - 3 * height
        val hi = yMax + 3 * height
        val maxRise = height / 25
        val points = ArrayList<Double>()
        var x = x0
        var y = y0
        var count = 0
        while (count++ < steps * 2) {
            // Shorten the last step so the curve ends exactly at the window edge.
            val remaining = if (direction > 0) xMax - x else xMin - x
            if (abs(remaining) < abs(h) * 1e-9 || remaining * direction <= 0) break
            val step = if (abs(remaining) < abs(h)) remaining else h
            val slope = f.eval(x, y, 0.0)
            if (!slope.isFinite()) break
            val pieces = min(64.0, ceil(abs(slope * step) / maxRise).coerceAtLeast(1.0)).toInt()
            val hs = step / pieces
            var ok = true
            for (p in 0 until pieces) {
                // A steep slope that flips sign within one small step means the
                // step straddles a singularity of f, not a turning point.
                val before = f.eval(x, y, 0.0)
                val next = rk4(f, x, y, hs)
                val after = f.eval(x + hs, next, 0.0)
                val straddles = before * after < 0 &&
                    min(abs(before), abs(after)) * abs(hs) > maxRise / 8
                x += hs
                y = next
                if (straddles || !y.isFinite() || !after.isFinite() || y < lo || y > hi) {
                    ok = false
                    break
                }
            }
            if (!ok) break
            points.add(x); points.add(y)
        }
        return points.toDoubleArray()
    }

    /** One classical Runge-Kutta step. */
    fun rk4(f: Expr, x: Double, y: Double, h: Double): Double {
        val k1 = f.eval(x, y, 0.0)
        val k2 = f.eval(x + h / 2, y + h / 2 * k1, 0.0)
        val k3 = f.eval(x + h / 2, y + h / 2 * k2, 0.0)
        val k4 = f.eval(x + h, y + h * k3, 0.0)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    }
}
