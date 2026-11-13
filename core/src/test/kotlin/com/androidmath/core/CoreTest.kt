package com.androidmath.core

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.E
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tan

class ExprTest {
    private fun eval(text: String, x: Double = 0.0, y: Double = 0.0, c: Double = 0.0): Double =
        Expr.parseOrNull(text)!!.eval(x, y, c)

    @Test fun arithmeticAndPrecedence() {
        assertEquals(7.0, eval("1 + 2*3"), 0.0)
        assertEquals(-9.0, eval("-3**2"), 0.0)
        assertEquals(512.0, eval("2**3**2"), 0.0)
        assertEquals(0.25, eval("2**-2"), 0.0)
        assertEquals(1.5, eval("x/2 - 1/4 + C", x = 2.0, c = 0.75), 1e-12)
        assertEquals(2.5e-3, eval("2.5e-3"), 0.0)
    }

    @Test fun engineOutputs() {
        assertEquals(exp(0.5), eval("exp(x**2/2)", x = 1.0), 1e-12)
        assertEquals(-sqrt(2.0), eval("-sqrt(x**2 + 1)", x = 1.0), 1e-12)
        assertEquals(3 * exp(-2.0) + 0.25, eval("C*exp(-2*x) + x/2 - 1/4", x = 1.0, c = 3.0), 1e-12)
        assertEquals(2.0, eval("x*(C + log(Abs(x)))", x = -1.0, c = -2.0), 1e-12)
        assertEquals(PI + E, eval("pi + E"), 1e-12)
        assertEquals(5.0, eval("x**2 + y**2", x = 1.0, y = 2.0), 0.0)
        assertEquals(tan(1.0), eval("tan(C + x)", x = 0.25, c = 0.75), 1e-12)
    }

    @Test fun twoConstants() {
        val e = Expr.parseOrNull("C*exp(-2*x) + C2*exp(-x) + exp(x)/6")!!
        assertEquals(2.0 + 3.0 + 1.0 / 6, e.eval(0.0, 0.0, 2.0, 3.0), 1e-12)
        assertEquals(2.0 + 1.0 / 6, e.eval(0.0, 0.0, 2.0), 1e-12)
        assertEquals(4.0, Expr.parseOrNull("C2**2")!!.eval(0.0, 0.0, 0.0, 2.0), 0.0)
    }

    @Test fun unsupportedInputIsRejected() {
        assertNull(Expr.parseOrNull("erf(x)"))
        assertNull(Expr.parseOrNull("x +"))
        assertNull(Expr.parseOrNull("(x"))
        assertNull(Expr.parseOrNull("z"))
        assertNull(Expr.parseOrNull(""))
        assertNull(Expr.parseOrNull(null))
    }

    @Test fun undefinedPointsGiveNaN() {
        assertTrue(eval("sqrt(x)", x = -1.0).isNaN())
        assertTrue(eval("1/x", x = 0.0).isInfinite())
    }
}

class OdeTest {
    private fun at(curve: DoubleArray, x: Double): Double {
        var best = 0
        for (i in 0 until curve.size / 2) if (abs(curve[2 * i] - x) < abs(curve[2 * best] - x)) best = i
        assertEquals(x, curve[2 * best], 1e-6)
        return curve[2 * best + 1]
    }

    @Test fun matchesExactSolution() {
        val f = Expr.parseOrNull("x*y")!!
        val curve = Ode.solve(f, 0.0, 1.0, -2.0, 2.0, -5.0, 10.0, steps = 400)
        assertEquals(exp(0.5), at(curve, 1.0), 1e-7)
        assertEquals(exp(0.5), at(curve, -1.0), 1e-7)
        assertEquals(-2.0, curve[0], 1e-9)
        for (i in 1 until curve.size / 2) assertTrue(curve[2 * i] > curve[2 * i - 2])
    }

    @Test fun stopsAtBlowUp() {
        // y' = y^2, y(0) = 1 blows up at x = 1.
        val f = Expr.parseOrNull("y**2")!!
        val curve = Ode.solve(f, 0.0, 1.0, -4.0, 4.0, -4.0, 4.0)
        val lastX = curve[curve.size - 2]
        assertTrue(lastX < 1.0)
        assertTrue(lastX > 0.8)
        assertTrue(curve.all { it.isFinite() })
    }

    @Test fun stopsAtSingularityOfTheEquation() {
        // y' = 1/x, y(1) = 0 is ln(x); it must not continue across x = 0.
        val f = Expr.parseOrNull("1/x")!!
        val curve = Ode.solve(f, 1.0, 0.0, -3.0, 3.0, -3.0, 3.0)
        assertTrue(curve.all { it.isFinite() })
        assertTrue(curve[0] > 0.0)
        assertEquals(kotlin.math.ln(2.0), at(curve, 2.0), 1e-7)
    }

    @Test fun passesThroughTurningPoints() {
        // y' = -x, y(0) = 2 is 2 - x^2/2: the slope changes sign smoothly.
        val f = Expr.parseOrNull("-x")!!
        val curve = Ode.solve(f, 0.0, 2.0, -3.0, 3.0, -3.0, 3.0)
        assertEquals(-3.0, curve[0], 1e-9)
        assertEquals(3.0, curve[curve.size - 2], 1e-9)
    }
}

class PlotTest {
    @Test fun niceTicks() {
        assertEquals(listOf(-4.0, -2.0, 0.0, 2.0, 4.0), Plot.ticks(-5.0, 5.0, 5))
        assertEquals(listOf(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), Plot.ticks(0.0, 1.0, 5).map { Math.round(it * 10) / 10.0 })
        assertEquals("0.2", Plot.tickLabel(0.2, 0.2))
        assertEquals("-3", Plot.tickLabel(-3.0, 1.0))
        assertEquals("0", Plot.tickLabel(0.0, 0.5))
        assertEquals("2.50", Plot.tickLabel(2.5, 0.05))
    }

    @Test fun viewportMapping() {
        val vp = Viewport(-2.0, 2.0, -1.0, 1.0)
        assertEquals(100f, vp.toPxX(0.0, 200f), 1e-4f)
        assertEquals(0f, vp.toPxY(1.0, 100f), 1e-4f)
        assertEquals(0.5, vp.toWorldY(25f, 100f), 1e-9)
        val zoomed = vp.zoomedBy(2f, 150f, 25f, 200f, 100f)
        assertEquals(2.0, zoomed.width, 1e-9)
        assertEquals(vp.toWorldX(150f, 200f), zoomed.toWorldX(150f, 200f), 1e-9)
        assertEquals(vp.toWorldY(25f, 100f), zoomed.toWorldY(25f, 100f), 1e-9)
        val panned = vp.pannedBy(50f, 0f, 200f, 100f)
        assertEquals(-3.0, panned.xMin, 1e-9)
        val sq = Viewport.centred(0.0, 0.0, 5.0, 400f, 200f)
        assertEquals(sq.width / 400, sq.height / 200, 1e-12)
    }

    @Test fun viewportFitsPoints() {
        val vp = Viewport.fitting(doubleArrayOf(0.0, 1.0, 0.5, 1.2, 1.0, 1.65), 400f, 800f)
        assertTrue(vp.xMin < 0.0 && vp.xMax > 1.0)
        assertTrue(vp.yMin < 1.0 && vp.yMax > 1.65)
        assertEquals(vp.width / 400, vp.height / 800, 1e-12)
        assertEquals(0.5, (vp.xMin + vp.xMax) / 2, 1e-12)
        val empty = Viewport.fitting(DoubleArray(0), 400f, 400f)
        assertEquals(10.0, empty.width, 1e-12)
    }

    @Test fun slopeFieldSegmentsHaveFixedLength() {
        val f = Expr.parseOrNull("x*y")!!
        val segs = Plot.slopeField(f, Viewport(-3.0, 3.0, -3.0, 3.0), 300f, 300f, 30f, 12f)
        assertEquals(10 * 10 * 4, segs.size)
        for (i in 0 until segs.size / 4) {
            val len = Math.hypot((segs[4 * i + 2] - segs[4 * i]).toDouble(), (segs[4 * i + 3] - segs[4 * i + 1]).toDouble())
            assertEquals(12.0, len, 1e-3)
        }
    }

    @Test fun curveBreaksAtAsymptotes() {
        val vp = Viewport(-4.0, 4.0, -4.0, 4.0)
        val pieces = Plot.curve({ tan(it) }, vp, 800f, 800f, 800)
        assertEquals(3, pieces.size)
        val all = pieces.flatMap { it.toList() }
        assertTrue(all.all { it.isFinite() && abs(it) < 5000f })
        val gap = Plot.curve({ sqrt(it) }, vp, 800f, 800f, 800)
        assertEquals(1, gap.size)
        assertTrue(gap[0][0] >= 399f)
    }
}

class ModelTest {
    private fun fixture(name: String) = javaClass.getResource("/$name")!!.readText()

    @Test fun parsesBothMethods() {
        val r = EngineJson.solveResult(fixture("both_methods_ivp.json"))
        assertTrue(r.ok)
        assertEquals("x*y", r.rhsExpr)
        assertEquals(listOf("separable", "linear"), r.applicableMethods.map { it.id })
        assertEquals(1.0, r.ic!!.y0, 0.0)
        for (m in r.applicableMethods) {
            assertEquals(true, m.verified)
            val particular = Expr.parseOrNull(m.plot!!.particularExpr)!!
            assertEquals(exp(0.5), particular.eval(1.0, 0.0, 0.0), 1e-12)
            assertNotNull(Expr.parseOrNull(m.plot!!.generalExpr))
        }
        assertNotNull(Expr.parseOrNull(r.rhsExpr))
    }

    @Test fun parsesGeneralOnly() {
        val r = EngineJson.solveResult(fixture("linear_general.json"))
        assertNull(r.ic)
        val linear = r.methods.first { it.id == "linear" }
        assertNull(linear.particularLatex)
        assertNull(linear.plot!!.particularExpr)
        assertFalse(r.methods.first { it.id == "separable" }.applicable)
    }

    @Test fun parsesNoMethodAndErrors() {
        val none = EngineJson.solveResult(fixture("none_applicable.json"))
        assertTrue(none.ok)
        assertTrue(none.applicableMethods.isEmpty())
        val err = EngineJson.solveResult(fixture("error.json"))
        assertFalse(err.ok)
        assertTrue(err.error!!.contains("Unknown symbol"))
        assertFalse(EngineJson.solveResult("not json").ok)
        assertEquals("y' = x", EngineJson.preview("""{"ok": true, "latex": "y' = x"}""").latex)
    }

    @Test fun parsesSecondOrder() {
        val r = EngineJson.solveResult(fixture("second_order_ivp.json"))
        assertTrue(r.ok)
        assertEquals(2, r.order)
        assertNull(r.rhsExpr)
        assertEquals(0.0, r.ic!!.y1!!, 0.0)
        assertEquals(listOf("undetermined", "variation", "laplace"), r.applicableMethods.map { it.id })
        val values = r.applicableMethods.map {
            Expr.parseOrNull(it.plot!!.particularExpr)!!.eval(1.0, 0.0, 0.0)
        }
        for (v in values) assertEquals(values[0], v, 1e-12)
        val general = EngineJson.solveResult(fixture("second_order_general.json"))
        val expr = Expr.parseOrNull(general.applicableMethods[0].plot!!.generalExpr)!!
        // C1 cos x + C2 sin x at x = 0 is C1.
        assertEquals(3.0, expr.eval(0.0, 0.0, 3.0, 7.0), 1e-12)
    }

    @Test fun parsesNumeric() {
        val r = EngineJson.solveResult(fixture("numeric.json"))
        assertEquals("numeric", r.kind)
        assertEquals(listOf("euler", "heun", "rk4"), r.methods.map { it.id })
        val pts = r.methods[0].plot!!.pointArray()
        assertEquals(10, pts.size)
        assertEquals(0.0, pts[0], 0.0)
        assertEquals(1.0, pts[8], 1e-12)
        assertEquals(1.419434, pts[9], 1e-6)
        assertNotNull(Expr.parseOrNull(r.methods[0].plot!!.particularExpr))
        assertEquals(2, EngineJson.preview("""{"ok": true, "latex": "", "order": 2}""").order)
    }
}
