package com.androidmath.core

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToLong

/** The visible part of the plane, in world coordinates. */
data class Viewport(val xMin: Double, val xMax: Double, val yMin: Double, val yMax: Double) {
    val width get() = xMax - xMin
    val height get() = yMax - yMin

    fun toPxX(x: Double, widthPx: Float): Float = ((x - xMin) / width * widthPx).toFloat()
    fun toPxY(y: Double, heightPx: Float): Float = ((yMax - y) / height * heightPx).toFloat()
    fun toWorldX(px: Float, widthPx: Float): Double = xMin + px / widthPx * width
    fun toWorldY(py: Float, heightPx: Float): Double = yMax - py / heightPx * height

    /** Moves the view by a drag of (dxPx, dyPx) screen pixels. */
    fun pannedBy(dxPx: Float, dyPx: Float, widthPx: Float, heightPx: Float): Viewport {
        val dx = -dxPx / widthPx * width
        val dy = dyPx / heightPx * height
        return Viewport(xMin + dx, xMax + dx, yMin + dy, yMax + dy)
    }

    /** Zooms by [factor] (>1 zooms in), keeping the world point under (px, py) fixed. */
    fun zoomedBy(factor: Float, px: Float, py: Float, widthPx: Float, heightPx: Float): Viewport {
        val newWidth = (width / factor).coerceIn(MIN_SPAN, MAX_SPAN)
        val newHeight = height * (newWidth / width)
        val cx = toWorldX(px, widthPx)
        val cy = toWorldY(py, heightPx)
        val fx = (px / widthPx).toDouble()
        val fy = (py / heightPx).toDouble()
        val left = cx - fx * newWidth
        val top = cy + fy * newHeight
        return Viewport(left, left + newWidth, top - newHeight, top)
    }

    companion object {
        const val MIN_SPAN = 1e-3
        const val MAX_SPAN = 1e5

        /** A window centred on (cx, cy) whose units are square for the given pixel size. */
        fun centred(cx: Double, cy: Double, halfWidth: Double, widthPx: Float, heightPx: Float): Viewport {
            val halfHeight = if (widthPx > 0f) halfWidth * heightPx / widthPx else halfWidth
            return Viewport(cx - halfWidth, cx + halfWidth, cy - halfHeight, cy + halfHeight)
        }
    }
}

object Plot {
    /** Round tick positions (multiples of 1, 2 or 5 times a power of ten) covering [min, max]. */
    fun ticks(min: Double, max: Double, target: Int = 6): List<Double> {
        val step = tickStep(min, max, target)
        val first = kotlin.math.ceil(min / step - 1e-9)
        val out = ArrayList<Double>()
        var i = 0
        while (true) {
            val v = (first + i) * step
            if (v > max + step * 1e-9 || out.size > 200) break
            out.add(if (abs(v) < step * 1e-9) 0.0 else v)
            i++
        }
        return out
    }

    fun tickStep(min: Double, max: Double, target: Int = 6): Double {
        val raw = (max - min) / target.coerceAtLeast(1)
        if (!(raw > 0) || !raw.isFinite()) return 1.0
        val magnitude = 10.0.pow(floor(log10(raw)))
        val scaled = raw / magnitude
        val nice = when {
            scaled < 1.5 -> 1.0
            scaled < 3.5 -> 2.0
            scaled < 7.5 -> 5.0
            else -> 10.0
        }
        return nice * magnitude
    }

    /** Label for a tick value, with just enough decimals for the given step. */
    fun tickLabel(value: Double, step: Double): String {
        if (value == 0.0) return "0"
        val decimals = (-floor(log10(step))).toInt().coerceIn(0, 6)
        if (abs(value) >= 1e5 || abs(value) < 1e-4) {
            return "%.1e".format(value).replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
        }
        return if (decimals == 0) value.roundToLong().toString() else "%.${decimals}f".format(value)
    }

    /**
     * Short line segments showing the slope f(x, y) on a grid.
     * Returns pixel coordinates [x1, y1, x2, y2, ...]. Every segment has the
     * same on-screen length, so steep and shallow slopes are equally visible.
     */
    fun slopeField(
        f: Expr, vp: Viewport, widthPx: Float, heightPx: Float,
        spacingPx: Float, lengthPx: Float,
    ): FloatArray {
        if (widthPx <= 0f || heightPx <= 0f || spacingPx <= 0f) return FloatArray(0)
        val nx = (widthPx / spacingPx).toInt().coerceAtLeast(1)
        val ny = (heightPx / spacingPx).toInt().coerceAtLeast(1)
        val out = FloatArray(nx * ny * 4)
        var k = 0
        val sx = widthPx / vp.width
        val sy = heightPx / vp.height
        for (i in 0 until nx) {
            val px = (i + 0.5f) * widthPx / nx
            val x = vp.toWorldX(px, widthPx)
            for (j in 0 until ny) {
                val py = (j + 0.5f) * heightPx / ny
                val y = vp.toWorldY(py, heightPx)
                val m = f.eval(x, y, 0.0)
                if (m.isNaN()) continue
                var dx: Double
                var dy: Double
                if (m.isInfinite()) {
                    dx = 0.0; dy = 1.0
                } else {
                    dx = sx; dy = -m * sy
                    val norm = hypot(dx, dy)
                    if (norm == 0.0 || !norm.isFinite()) {
                        dx = 0.0; dy = 1.0
                    } else {
                        dx /= norm; dy /= norm
                    }
                }
                val hx = (dx * lengthPx / 2).toFloat()
                val hy = (dy * lengthPx / 2).toFloat()
                out[k++] = px - hx; out[k++] = py - hy
                out[k++] = px + hx; out[k++] = py + hy
            }
        }
        return if (k == out.size) out else out.copyOf(k)
    }

    /**
     * Samples y = g(x) across the window and returns polylines in pixels,
     * each as [x0, y0, x1, y1, ...]. The curve is broken wherever the
     * function is undefined or jumps across the window (an asymptote).
     */
    fun curve(g: (Double) -> Double, vp: Viewport, widthPx: Float, heightPx: Float, samples: Int): List<FloatArray> {
        val xs = DoubleArray(samples + 1) { vp.xMin + vp.width * it / samples }
        val ys = DoubleArray(samples + 1) { g(xs[it]) }
        return polylines(xs, ys, vp, widthPx, heightPx)
    }

    /** Converts world points [x0, y0, x1, y1, ...] to pixel polylines, clipped to the window. */
    fun path(points: DoubleArray, vp: Viewport, widthPx: Float, heightPx: Float): List<FloatArray> {
        val n = points.size / 2
        val xs = DoubleArray(n) { points[2 * it] }
        val ys = DoubleArray(n) { points[2 * it + 1] }
        return polylines(xs, ys, vp, widthPx, heightPx)
    }

    private fun polylines(xs: DoubleArray, ys: DoubleArray, vp: Viewport, widthPx: Float, heightPx: Float): List<FloatArray> {
        val lo = vp.yMin - 2 * vp.height
        val hi = vp.yMax + 2 * vp.height
        val result = ArrayList<FloatArray>()
        var current = ArrayList<Float>()

        fun flush() {
            if (current.size >= 4) result.add(current.toFloatArray())
            current = ArrayList()
        }

        fun add(x: Double, y: Double) {
            current.add(vp.toPxX(x, widthPx))
            current.add(vp.toPxY(y.coerceIn(lo, hi), heightPx))
        }

        var prevInside = false
        var prevFinite = false
        for (i in xs.indices) {
            val y = ys[i]
            val finite = y.isFinite()
            val inside = finite && y > lo && y < hi
            if (inside) {
                // Entering from off-window: start at the clamped previous point
                // unless it was on the opposite side of an asymptote.
                if (!prevInside && prevFinite && i > 0) add(xs[i - 1], ys[i - 1])
                add(xs[i], y)
            } else {
                if (prevInside && finite) add(xs[i], y)
                flush()
            }
            prevInside = inside
            prevFinite = finite
        }
        flush()
        return result
    }
}
