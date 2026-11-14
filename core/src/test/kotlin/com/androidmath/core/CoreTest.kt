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

    @Test fun unitStep() {
        val e = Expr.parseOrNull("sin(x - 2)*Heaviside(x - 2)")!!
        assertEquals(0.0, e.eval(1.0, 0.0, 0.0), 0.0)
        assertEquals(kotlin.math.sin(1.0), e.eval(3.0, 0.0, 0.0), 1e-12)
    }

    @Test fun floorAndPeriodicExtension() {
        assertEquals(-2.0, Expr.parseOrNull("floor(x)")!!.eval(-1.5, 0.0, 0.0), 0.0)
        // x reduced to [-pi, pi): the sawtooth has period 2 pi.
        val saw = Expr.parseOrNull("x - 2*pi*floor((x + pi)/(2*pi))")!!
        assertEquals(1.0, saw.eval(1.0, 0.0, 0.0), 1e-12)
        assertEquals(1.0, saw.eval(1.0 + 2 * PI, 0.0, 0.0), 1e-12)
        assertEquals(-1.0, saw.eval(-1.0 - 4 * PI, 0.0, 0.0), 1e-12)
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

    @Test fun closedOrbitOfACenter() {
        // x' = y, y' = -x: circles around the origin, traversed clockwise.
        val f = Expr.parseOrNull("y")!!
        val g = Expr.parseOrNull("-x")!!
        val orbit = Ode.trajectory(f, g, 2.0, 0.0, -5.0, 5.0, -5.0, 5.0)
        assertTrue(orbit.size > 200)
        for (i in 0 until orbit.size / 2) {
            assertEquals(2.0, kotlin.math.hypot(orbit[2 * i], orbit[2 * i + 1]), 1e-6)
        }
        val start = orbit.size / 2 / 2   // the middle of the array is the starting point
        assertTrue(orbit[2 * (start + 5) + 1] < 0.0)   // forward in time y decreases first
    }

    @Test fun orbitStopsAtEquilibriumAndOffWindow() {
        // x' = -x, y' = -y: straight into the origin forwards, off the window backwards.
        val f = Expr.parseOrNull("-x")!!
        val g = Expr.parseOrNull("-y")!!
        val orbit = Ode.trajectory(f, g, 1.0, 1.0, -3.0, 3.0, -3.0, 3.0)
        assertTrue(orbit.all { it.isFinite() })
        val last = orbit.size - 2
        assertTrue(kotlin.math.hypot(orbit[last], orbit[last + 1]) < 0.05)
        assertTrue(orbit[0] > 3.0 && orbit[0] < 9.5)
        assertEquals(orbit[0], orbit[1], 1e-9)
        assertEquals(2, Ode.trajectory(f, g, 0.0, 0.0, -3.0, 3.0, -3.0, 3.0).size)
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

    @Test fun vectorFieldArrows() {
        val f = Expr.parseOrNull("y")!!
        val g = Expr.parseOrNull("-x")!!
        val arrows = Plot.vectorField(f, g, Viewport(-3.0, 3.0, -3.0, 3.0), 300f, 300f, 30f, 12f)
        assertEquals(10 * 10 * 4, arrows.size)
        // At the top-left sample (x < 0, y > 0) the field (y, -x) points right and up on screen.
        assertTrue(arrows[2] > arrows[0])
        assertTrue(arrows[3] < arrows[1])
        val still = Plot.vectorField(Expr.parseOrNull("0*x")!!, Expr.parseOrNull("0*y")!!, Viewport(-1.0, 1.0, -1.0, 1.0), 100f, 100f, 50f, 10f)
        assertEquals(0, still.size)
        val circle = Plot.parametric({ kotlin.math.cos(it) }, { kotlin.math.sin(it) }, 0.0, PI, 2)
        assertEquals(6, circle.size)
        assertEquals(-1.0, circle[4], 1e-12)
        val pieces = Plot.path(doubleArrayOf(0.0, 0.0, 1e9, 1.0, 0.5, 0.5), Viewport(-1.0, 1.0, -1.0, 1.0), 100f, 100f)
        assertTrue(pieces.flatMap { it.toList() }.all { abs(it) < 1000f })
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
        assertEquals(listOf("euler", "heun", "rk4", "midpoint", "ab4", "rkf45"), r.methods.map { it.id })
        val pts = r.methods[0].plot!!.pointArray()
        assertEquals(10, pts.size)
        assertEquals(0.0, pts[0], 0.0)
        assertEquals(1.0, pts[8], 1e-12)
        assertEquals(1.419434, pts[9], 1e-6)
        assertNotNull(Expr.parseOrNull(r.methods[0].plot!!.particularExpr))
        assertEquals(2, EngineJson.preview("""{"ok": true, "latex": "", "order": 2}""").order)
    }

    @Test fun parsesTransform() {
        val r = EngineJson.solveResult(fixture("inverse_laplace.json"))
        assertEquals("transform", r.kind)
        assertNull(r.ic)
        val m = r.applicableMethods.single()
        assertEquals("fractions", m.id)
        val f = Expr.parseOrNull(m.plot!!.particularExpr)!!
        assertEquals(0.0, f.eval(1.0, 0.0, 0.0), 0.0)
        assertEquals(kotlin.math.sin(1.0), f.eval(3.0, 0.0, 0.0), 1e-12)
    }

    @Test fun parsesSystems() {
        val linear = EngineJson.solveResult(fixture("system_linear.json"))
        assertEquals("system", linear.kind)
        assertEquals(listOf("eigen", "expm"), linear.applicableMethods.map { it.id })
        val f = Expr.parseOrNull(linear.field!!.f)!!
        assertEquals(-5.0, f.eval(1.0, 2.0, 0.0), 0.0)
        assertEquals("spiral sink", linear.equilibria.single().label)
        val xt = Expr.parseOrNull(linear.applicableMethods[0].plot!!.xExpr)!!
        val yt = Expr.parseOrNull(linear.applicableMethods[0].plot!!.yExpr)!!
        assertEquals(1.0, xt.eval(0.0, 0.0, 0.0), 1e-12)
        assertEquals(1.0, yt.eval(0.0, 0.0, 0.0), 1e-12)
        val nonlinear = EngineJson.solveResult(fixture("system_nonlinear.json"))
        assertEquals(2, nonlinear.equilibria.size)
        assertNull(nonlinear.applicableMethods.single().plot!!.xExpr)
        assertNotNull(Expr.parseOrNull(nonlinear.field!!.g))
    }

    @Test fun parsesFourier() {
        val r = EngineJson.solveResult(fixture("fourier.json"))
        assertEquals("fourier", r.kind)
        assertEquals(PI, r.span!!, 1e-12)
        assertEquals(listOf("full", "sine", "cosine"), r.applicableMethods.map { it.id })
        val full = r.applicableMethods[0].plot!!
        assertEquals(listOf(1, 2, 3, 5, 10, 20), full.partials.map { it.n })
        val f = Expr.parseOrNull(full.particularExpr)!!
        assertEquals(1.0, f.eval(1.0, 0.0, 0.0), 1e-12)
        // Twenty terms of the sawtooth series are close to f away from the jumps.
        val sum = Expr.parseOrNull(full.partials.last().expr)!!
        assertEquals(1.0, sum.eval(1.0, 0.0, 0.0), 0.1)
        assertEquals(f.eval(8.0, 0.0, 0.0), sum.eval(8.0, 0.0, 0.0), 0.1)
        val one = Expr.parseOrNull(full.partials.first().expr)!!
        assertEquals(2 * kotlin.math.sin(1.0), one.eval(1.0, 0.0, 0.0), 1e-9)
    }

    @Test fun parsesBoundaryAndEigen() {
        val bvp = EngineJson.solveResult(fixture("boundary.json"))
        assertEquals("boundary", bvp.kind)
        val y = Expr.parseOrNull(bvp.applicableMethods.single().plot!!.particularExpr)!!
        assertEquals(1.0, y.eval(PI / 2, 0.0, 0.0), 1e-12)
        val eigen = EngineJson.solveResult(fixture("eigen.json"))
        assertEquals("eigen", eigen.kind)
        val functions = eigen.applicableMethods.single().plot!!.partials
        assertEquals(listOf(1, 2, 3, 4, 5), functions.map { it.n })
        for (p in functions) assertEquals(0.0, Expr.parseOrNull(p.expr)!!.eval(PI, 0.0, 0.0), 1e-9)
    }

    @Test fun parsesForcedAndThreeBySystems() {
        val forced = EngineJson.solveResult(fixture("system_forced.json"))
        assertEquals("system", forced.kind)
        assertNull(forced.field)
        val m = forced.applicableMethods.single()
        assertEquals("variation", m.id)
        assertEquals(1.0, Expr.parseOrNull(m.plot!!.xExpr)!!.eval(2.0, 0.0, 0.0), 1e-12)
        assertEquals(-2.0, Expr.parseOrNull(m.plot!!.yExpr)!!.eval(2.0, 0.0, 0.0), 1e-12)
        val three = EngineJson.solveResult(fixture("system_three.json"))
        assertNull(three.field)
        assertTrue(three.equilibria.isEmpty())
        assertEquals(listOf("eigen", "expm"), three.applicableMethods.map { it.id })
        assertTrue(three.applicableMethods[0].particularLatex!!.contains("z(t)"))
    }
}

class HistoryTest {
    private fun entry(equation: String, time: Long, mode: String = "first") =
        History.entry(mode, "First-order equation", equation, mapOf("equation" to equation, "x0" to ""), time)

    @Test fun newestFirstAndNoDuplicates() {
        var list = emptyList<HistoryEntry>()
        list = History.add(list, entry("y' = y", 1))
        list = History.add(list, entry("y' = x", 2))
        list = History.add(list, entry("y' = y", 3))
        assertEquals(listOf("y' = y", "y' = x"), list.map { it.label })
        assertEquals(3L, list[0].time)
        // Blank fields do not affect identity, and are not stored.
        assertEquals(History.keyOf("first", mapOf("equation" to "y' = y")), list[0].key)
        assertFalse(list[0].fields.containsKey("x0"))
        // The same text in another section is a different problem.
        assertEquals(3, History.add(list, entry("y' = y", 4, mode = "numeric")).size)
    }

    @Test fun favoritesSurviveCapClearAndResolve() {
        var list = History.add(emptyList(), entry("keep", 0))
        list = History.toggleFavorite(list, list[0].key)
        for (i in 1..History.MAX_RECENT + 10) list = History.add(list, entry("p$i", i.toLong()))
        assertEquals(History.MAX_RECENT + 1, list.size)
        assertEquals("keep", History.favorites(list).single().label)
        assertEquals("p${History.MAX_RECENT + 10}", History.recent(list).first().label)
        assertEquals("p11", History.recent(list).last().label)
        // Solving a favorite again keeps the star and moves it to the top.
        list = History.add(list, entry("keep", 999))
        assertTrue(list[0].favorite)
        assertEquals(999L, list[0].time)
        assertEquals(listOf("keep"), History.clearRecent(list).map { it.label })
        val off = History.toggleFavorite(list, list[0].key)
        assertTrue(History.favorites(off).isEmpty())
    }

    @Test fun roundTripsThroughJson() {
        val list = History.add(
            History.add(emptyList(), entry("y' = \"quoted\" + x", 5)),
            History.entry("system", "Systems", "x' = y, y' = -x", mapOf("equation" to "y", "equation2" to "-x"), 6),
        )
        assertEquals(list, History.decode(History.encode(list)))
        assertTrue(History.decode(null).isEmpty())
        assertTrue(History.decode("not json").isEmpty())
        assertTrue(History.decode("").isEmpty())
    }
}

class PdeTest {
    private fun fixture(name: String) = javaClass.getResource("/$name")!!.readText()

    @Test fun parsesHeat() {
        val r = EngineJson.solveResult(fixture("pde_heat.json"))
        assertEquals("pde", r.kind)
        assertEquals("heat", r.pde)
        assertFalse(r.surface)
        val d = r.domain!!
        assertEquals(1.0, d.xMax, 0.0)
        assertTrue(d.tMax > 0.0 && d.uMax > d.uMin)
        val initial = Expr.parseOrNull(r.initialExpr)!!
        assertEquals(0.25, initial.eval(0.5, 0.0, 0.0), 1e-12)
        // y stands for time: at t = 0 the series reproduces f, later it has decayed.
        val u = Expr.parseOrNull(r.applicableMethods.single().plot!!.particularExpr)!!
        assertEquals(0.25, u.eval(0.5, 0.0, 0.0), 1e-3)
        assertTrue(u.eval(0.5, d.tMax, 0.0) < 0.01)
        assertEquals(0.0, u.eval(0.0, 0.1, 0.0), 1e-9)
    }

    @Test fun parsesWaveWithBothMethods() {
        val r = EngineJson.solveResult(fixture("pde_wave.json"))
        assertEquals("wave", r.pde)
        assertEquals(listOf("separation", "dalembert"), r.applicableMethods.map { it.id })
        for (m in r.applicableMethods) {
            val u = Expr.parseOrNull(m.plot!!.particularExpr)!!
            // u = cos(2 pi t) sin(pi x): half a period later the string is inverted.
            assertEquals(kotlin.math.sin(PI * 0.3), u.eval(0.3, 0.0, 0.0), 1e-9)
            assertEquals(-kotlin.math.sin(PI * 0.3), u.eval(0.3, 0.5, 0.0), 1e-9)
        }
    }

    @Test fun parsesLaplaceAndBuildsHeatMap() {
        val r = EngineJson.solveResult(fixture("pde_laplace.json"))
        assertEquals("laplace", r.pde)
        assertTrue(r.surface)
        assertNull(r.initialExpr)
        val d = r.domain!!
        assertEquals(2.0, d.yMax, 0.0)
        val u = Expr.parseOrNull(r.applicableMethods.single().plot!!.particularExpr)!!
        val exact = kotlin.math.sinh(PI * 1.0) / kotlin.math.sinh(PI * 2.0) * kotlin.math.sin(PI * 0.5)
        assertEquals(exact, u.eval(0.5, 1.0, 0.0), 1e-9)
        val map = Plot.heatMap(u, d.xMin, d.xMax, d.yMin, d.yMax, 8, 16, d.uMin, d.uMax)
        assertEquals(8 * 16, map.size)
        assertTrue(map.all { it in 0f..1f })
        // Hottest along the top edge (row 0), cold at the bottom.
        assertTrue(map[3] > 0.8f)
        assertTrue(map[15 * 8 + 3] < 0.05f)
        assertTrue(map[3] > map[0])
    }

    @Test fun heatMapClampsAndFlagsUndefined() {
        val plane = Expr.parseOrNull("x + y")!!
        val map = Plot.heatMap(plane, 0.0, 2.0, 0.0, 2.0, 2, 2, 1.0, 3.0)
        // Cell centres: top row y = 1.5, x = 0.5 and 1.5; bottom row y = 0.5.
        assertEquals(0.5f, map[0], 1e-6f)
        assertEquals(1.0f, map[1], 1e-6f)
        assertEquals(0.0f, map[2], 1e-6f)
        assertEquals(0.5f, map[3], 1e-6f)
        val undefined = Plot.heatMap(Expr.parseOrNull("sqrt(x - 5)")!!, 0.0, 1.0, 0.0, 1.0, 2, 1, 0.0, 1.0)
        assertTrue(undefined.all { it.isNaN() })
        val flat = Plot.heatMap(Expr.parseOrNull("3")!!, 0.0, 1.0, 0.0, 1.0, 1, 1, 3.0, 3.0)
        assertEquals(0f, flat[0], 0f)
    }

    @Test fun parsesSecondPdeFixtures() {
        val moving = EngineJson.solveResult(fixture("pde_characteristics.json"))
        assertEquals("characteristics", moving.pde)
        val u = Expr.parseOrNull(moving.applicableMethods.single().plot!!.particularExpr)!!
        // e^(-(x - 2t)^2) with y standing for t: the peak has moved to x = 2 at t = 1.
        assertEquals(1.0, u.eval(2.0, 1.0, 0.0), 1e-12)
        assertTrue(moving.domain!!.tMax > 0.0)
        assertNotNull(Expr.parseOrNull(moving.initialExpr))

        val type = EngineJson.solveResult(fixture("pde_classify.json"))
        assertEquals("classify", type.pde)
        assertNull(type.applicableMethods.single().plot!!.particularExpr)

        val grid = EngineJson.solveResult(fixture("pde_fd.json"))
        assertEquals(listOf("ftcs", "cn"), grid.applicableMethods.map { it.id })
        val plot = grid.applicableMethods[0].plot!!
        assertEquals(11, plot.levels.size)
        assertEquals(0, plot.levelNear(0.004))
        assertEquals(3, plot.levelNear(0.061))
        assertEquals(10, plot.levelNear(5.0))
        val points = plot.levels[1].pointArray()
        assertEquals(10, points.size)
        assertEquals(0.25, points[2], 1e-12)
        val exact = Expr.parseOrNull(plot.particularExpr)!!
        assertEquals(exact.eval(0.5, 0.2, 0.0), plot.levels[10].pointArray()[5], 0.02)

        val disk = EngineJson.solveResult(fixture("pde_disk.json"))
        assertEquals("disk", disk.domain!!.mask)
        assertTrue(disk.surface)
        val harmonic = Expr.parseOrNull(disk.applicableMethods.single().plot!!.particularExpr)!!
        assertEquals(0.5, harmonic.eval(0.0, 0.0, 0.0), 1e-9)
        assertTrue(harmonic.eval(0.0, 0.8, 0.0) > 0.8)
        assertTrue(harmonic.eval(0.0, -0.8, 0.0) < 0.2)
    }

    @Test fun heatMapDiskMask() {
        val one = Expr.parseOrNull("1")!!
        val map = Plot.heatMap(one, -1.0, 1.0, -1.0, 1.0, 4, 4, 0.0, 1.0, disk = true)
        // Corner cells lie outside the unit circle; the four centre cells are inside.
        assertTrue(map[0].isNaN() && map[3].isNaN() && map[12].isNaN() && map[15].isNaN())
        assertEquals(1f, map[5], 0f)
        assertEquals(1f, map[10], 0f)
        assertEquals(12, map.count { !it.isNaN() })
        assertEquals(16, Plot.heatMap(one, -1.0, 1.0, -1.0, 1.0, 4, 4, 0.0, 1.0).count { !it.isNaN() })
    }

    @Test fun parsesMathTools() {
        val linear = EngineJson.solveResult(fixture("tool_linalg.json"))
        assertEquals("tool", linear.kind)
        assertEquals("linalg", linear.tool)
        assertEquals("Linear system", linear.title)
        assertEquals(listOf("gauss", "cramer", "inverse"), linear.applicableMethods.map { it.id })
        assertNull(linear.rhsExpr)

        val integral = EngineJson.solveResult(fixture("tool_calculus.json"))
        assertEquals("Integral", integral.title)
        val integrand = Expr.parseOrNull(integral.applicableMethods[0].plot!!.particularExpr)!!
        assertEquals(2.0 * kotlin.math.cos(4.0), integrand.eval(2.0, 0.0, 0.0), 1e-12)

        val taylor = EngineJson.solveResult(fixture("tool_series.json"))
        val plot = taylor.applicableMethods.single().plot!!
        assertTrue(plot.partials.size >= 3)
        val best = Expr.parseOrNull(plot.partials.last().expr)!!
        assertEquals(kotlin.math.sin(0.5), best.eval(0.5, 0.0, 0.0), 1e-6)
        assertTrue(plot.partials.zipWithNext().all { (a, b) -> a.n < b.n })

        val roots = EngineJson.solveResult(fixture("tool_complex.json"))
        val points = roots.applicableMethods.single().plot!!.pointArray()
        assertEquals(8, points.size)                     // three roots and the first one repeated
        for (i in 0 until 3) {
            assertEquals(2.0, kotlin.math.hypot(points[2 * i], points[2 * i + 1]), 1e-9)
        }
        assertEquals(points[0], points[6], 0.0)
        assertEquals(points[1], points[7], 0.0)
    }

    @Test fun parsesSpecialFunctionsTransformsAndRecurrences() {
        val legendre = EngineJson.solveResult(fixture("tool_special.json"))
        val curves = legendre.applicableMethods.single().plot!!.partials
        assertEquals(listOf(0, 1, 2, 3), curves.map { it.n })
        assertEquals("P\u2083", curves[3].label)
        // P3(x) = (5x^3 - 3x)/2
        assertEquals(-0.4375, Expr.parseOrNull(curves[3].expr)!!.eval(0.5, 0.0, 0.0), 1e-12)

        val bessel = EngineJson.solveResult(fixture("tool_bessel.json"))
        assertEquals("J", bessel.legends["exact"])
        assertEquals("Y", bessel.legends["points"])
        val j0 = Expr.parseOrNull(bessel.applicableMethods.single().plot!!.particularExpr)!!
        assertEquals(1.0, j0.eval(0.0, 0.0, 0.0), 1e-12)
        assertEquals(0.0, j0.eval(2.404825557695773, 0.0, 0.0), 1e-9)      // first zero of J0
        assertEquals(-0.2459357644513483, j0.eval(10.0, 0.0, 0.0), 1e-7)
        assertFalse(j0.eval(31.0, 0.0, 0.0).isFinite())                    // blank beyond the range
        assertTrue(bessel.applicableMethods.single().plot!!.points.size > 100)

        val heat = EngineJson.solveResult(fixture("tool_ftransform.json"))
        val times = heat.applicableMethods.single().plot!!.partials
        assertTrue(times.size >= 4)
        assertTrue(times.all { it.label!!.startsWith("t = ") })
        val early = Expr.parseOrNull(times.first().expr)!!
        val late = Expr.parseOrNull(times.last().expr)!!
        // A pulse of height 1 on (-1, 1) spreads out: the centre cools, the outside warms.
        assertTrue(early.eval(0.0, 0.0, 0.0) > 0.99)
        assertTrue(late.eval(0.0, 0.0, 0.0) < early.eval(0.0, 0.0, 0.0))
        assertTrue(late.eval(3.0, 0.0, 0.0) > early.eval(3.0, 0.0, 0.0))

        val fibonacci = EngineJson.solveResult(fixture("tool_recurrence.json"))
        assertEquals(listOf("characteristic", "iteration"), fibonacci.applicableMethods.map { it.id })
        val terms = fibonacci.applicableMethods[0].plot!!.pointArray()
        assertEquals(55.0, terms[2 * 10 + 1], 1e-9)
        assertEquals("Terms of the sequence", fibonacci.legends["points"])
    }
}
