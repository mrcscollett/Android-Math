package com.androidmath.app

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.androidmath.core.History
import com.androidmath.core.HistoryEntry
import com.androidmath.core.Method
import com.androidmath.core.SolveResult
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

enum class Screen { Home, Input, Result }

/** PDE tools whose number boxes hold something other than a length and a speed. */
private val PDE_TOOLS = setOf("characteristics", "classify", "fd", "disk")

/** Which section of the app the input screen is serving. */
enum class Mode(val title: String) {
    FirstOrder("First-order equation"),
    SecondOrder("Second-order equation"),
    Numerical("Numerical methods"),
    Laplace("Laplace transforms"),
    Systems("System of equations"),
    Fourier("Fourier series"),
    Boundary("Boundary value problems"),
    Pde("Partial differential equations"),
    LinearAlgebra("Linear algebra"),
    Calculus("Calculus"),
    Series("Series"),
    Complex("Complex numbers"),
    Special("Special functions"),
    FourierTransform("Fourier transforms"),
    Recurrence("Difference equations"),
    Dynamics("Stability and dynamics"),
    Vector("Vector calculus"),
}

/** The engine's name for a math-tool section, or null for the other sections. */
fun Mode.toolName(): String? = when (this) {
    Mode.LinearAlgebra -> "linalg"
    Mode.Calculus -> "calculus"
    Mode.Series -> "series"
    Mode.Complex -> "complex"
    Mode.Special -> "special"
    Mode.FourierTransform -> "ftransform"
    Mode.Recurrence -> "recurrence"
    Mode.Dynamics -> "dynamics"
    Mode.Vector -> "vector"
    else -> null
}

/** The operation a math tool opens with. */
private fun Mode.firstOperation(): String = when (this) {
    Mode.LinearAlgebra -> "rref"
    Mode.Calculus -> "integral"
    Mode.Series -> "taylor"
    Mode.Complex -> "evaluate"
    Mode.Special -> "bessel"
    Mode.FourierTransform -> "forward"
    Mode.Recurrence -> "solve"
    Mode.Dynamics -> "lyapunov"
    Mode.Vector -> "gradient"
    else -> ""
}

class SolverViewModel(app: Application) : AndroidViewModel(app) {
    var screen by mutableStateOf(Screen.Home)
    var mode by mutableStateOf(Mode.FirstOrder)
        private set

    var equation by mutableStateOf(TextFieldValue(""))
        private set
    /** In the systems section: the right-hand side of y' (the first field holds x'). */
    var equation2 by mutableStateOf(TextFieldValue(""))
        private set
    /** In the systems section: the optional right-hand side of z' (a 3x3 linear system). */
    var equation3 by mutableStateOf(TextFieldValue(""))
        private set
    /** z(0), used by 3x3 systems. */
    var z0 by mutableStateOf("")
    /** Which equation field the math keys type into (0, 1 or 2). */
    var activeField by mutableStateOf(0)

    /** Solved problems, newest first; favorites are kept until unstarred. */
    var history by mutableStateOf<List<HistoryEntry>>(emptyList())
        private set
    private val historyStore = HistoryStore(app)
    var x0 by mutableStateOf("")
    var y0 by mutableStateOf("")
    /** y'(x0), used by second-order problems. */
    var yPrime0 by mutableStateOf("")
    var stepSize by mutableStateOf("0.1")
    var stepCount by mutableStateOf("10")
    /** In the Laplace section: false transforms f(t), true inverts F(s). */
    var inverse by mutableStateOf(false)
        private set

    var previewLatex by mutableStateOf("")
        private set
    var previewError by mutableStateOf<String?>(null)
        private set
    /** Order of the equation as the engine reads it (1 to 4). */
    var previewOrder by mutableStateOf(1)
        private set
    var engineReady by mutableStateOf(false)
        private set
    var solving by mutableStateOf(false)
        private set
    var inputError by mutableStateOf<String?>(null)
        private set

    var resultJson by mutableStateOf("")
        private set
    var result by mutableStateOf<SolveResult?>(null)
        private set
    var methodId by mutableStateOf<String?>(null)
    var tab by mutableStateOf(0)

    /** Points the user tapped on the graph: world (x, y) pairs. */
    val tappedPoints = mutableStateListOf<Pair<Double, Double>>()
    var constant by mutableStateOf(1f)
    var constant2 by mutableStateOf(1f)
    var viewResetKey by mutableStateOf(0)

    private var previewJob: Job? = null

    init {
        history = historyStore.load()
        viewModelScope.launch {
            Engine.warmUp(getApplication())
            engineReady = true
        }
    }

    val selectedMethod: Method?
        get() {
            val methods = result?.applicableMethods ?: return null
            return methods.firstOrNull { it.id == methodId } ?: methods.firstOrNull()
        }

    /** True when the problem takes y'(x0) as well: the second-order section, or y'' typed elsewhere. */
    val isSecondOrder: Boolean
        get() = mode == Mode.SecondOrder ||
            (mode == Mode.FirstOrder && (previewOrder >= 2 || equation.text.contains("''")))

    /** Boundary section: false solves a boundary value problem, true the eigenvalue problem. */
    var eigenMode by mutableStateOf(false)
        private set
    /** Boundary value problem: the two points, whether each condition is on y', and the values. */
    var pointA by mutableStateOf("")
    var pointB by mutableStateOf("")
    var slopeAtA by mutableStateOf(false)
    var slopeAtB by mutableStateOf(false)
    var valueA by mutableStateOf("")
    var valueB by mutableStateOf("")
    /** Eigenvalue problem: dd, nn, dn or nd (d: y = 0, n: y' = 0 at 0 and at L). */
    var eigenKind by mutableStateOf("dd")
    /** Which partial sum or eigenfunction the graph shows. */
    var partialIndex by mutableStateOf(0)

    /** PDE section: heat, wave, laplace, characteristics, classify, fd or disk. */
    var pdeKind by mutableStateOf("heat")
        private set
    /** Finite differences: the number of time steps. */
    var pdeSteps by mutableStateOf("10")
    /** Heat equation boundary conditions: 00, nn, 0n, n0 or tt (ends held at T1 and T2). */
    var pdeBc by mutableStateOf("00")
    /** Laplace's equation: the side that carries f (top, bottom, left or right). */
    var pdeSide by mutableStateOf("top")
        private set
    var pdeT1 by mutableStateOf("")
    var pdeT2 by mutableStateOf("")
    /** The time shown on the heat and wave graphs, and whether it is being animated. */
    var pdeTime by mutableStateOf(0f)
    var pdePlaying by mutableStateOf(false)

    /** Math tools: the chosen operation (rref, det, integral, taylor, roots, ...). */
    var toolOp by mutableStateOf("")
        private set

    /** Switches a math tool to another operation; its small boxes start empty. */
    fun chooseTool(operation: String) {
        if (operation == toolOp) return
        toolOp = operation
        x0 = ""; y0 = ""
        inputError = null
        refreshPreview()
    }

    /** True when the screen has a second expression field (y' of a system, left half of f, g of a wave). */
    private val twoFields: Boolean
        get() = mode == Mode.Systems || mode == Mode.Fourier || mode == Mode.Pde

    /** Switches the PDE section between the heat, wave and Laplace equations. */
    fun choosePde(kind: String) {
        if (kind == pdeKind) return
        // The number boxes mean different things in these four tools (coefficients,
        // grid sizes, a radius), so they are not carried over from another problem.
        if (kind in PDE_TOOLS || pdeKind in PDE_TOOLS) {
            x0 = ""; y0 = ""; pdeT1 = ""; pdeT2 = ""
        }
        pdeKind = kind
        // Only the wave equation has a second function (the initial velocity).
        equation2 = TextFieldValue("")
        activeField = 0
        inputError = null
        refreshPreview()
    }

    /** Laplace: which side carries the data. Left and right take a function of y. */
    fun chooseSide(side: String) {
        if (side == pdeSide) return
        pdeSide = side
        inputError = null
        refreshPreview()
    }

    fun chooseEigen(value: Boolean) {
        if (value == eigenMode) return
        eigenMode = value
        inputError = null
    }

    private val previewKind: String
        get() = when {
            mode.toolName() != null -> mode.toolName() + ":" + toolOp
            mode == Mode.Systems -> "system"
            mode == Mode.Fourier -> "fourier"
            mode == Mode.Pde -> when {
                pdeKind == "laplace" && (pdeSide == "left" || pdeSide == "right") -> "pdey"
                pdeKind == "disk" -> "pdet"     // the boundary values are a function of the angle t
                else -> "pde"
            }
            mode != Mode.Laplace -> "ode"
            inverse -> "inverse"
            else -> "forward"
        }

    /** Switches the Laplace section between transforming and inverting. */
    fun chooseInverse(value: Boolean) {
        if (value == inverse) return
        inverse = value
        // f(t) and F(s) are different expressions, so start the field afresh.
        equation = TextFieldValue("")
        previewLatex = ""; previewError = null; inputError = null
    }

    fun open(newMode: Mode) {
        if (newMode != mode) {
            mode = newMode
            equation = TextFieldValue("")
            equation2 = TextFieldValue("")
            equation3 = TextFieldValue("")
            activeField = 0
            x0 = ""; y0 = ""; yPrime0 = ""; z0 = ""
            pointA = ""; pointB = ""; valueA = ""; valueB = ""
            slopeAtA = false; slopeAtB = false
            pdeT1 = ""; pdeT2 = ""
            toolOp = newMode.firstOperation()
            previewLatex = ""; previewError = null; previewOrder = 1; inputError = null
        }
        screen = Screen.Input
    }

    fun onEquationChange(value: TextFieldValue) {
        val textChanged = value.text != equation.text
        equation = value
        if (textChanged) refreshPreview()
    }

    fun onEquation2Change(value: TextFieldValue) {
        val textChanged = value.text != equation2.text
        equation2 = value
        if (textChanged) refreshPreview()
    }

    fun onEquation3Change(value: TextFieldValue) {
        val textChanged = value.text != equation3.text
        equation3 = value
        if (textChanged) refreshPreview()
    }

    private fun refreshPreview() {
        inputError = null
        previewJob?.cancel()
        val text = when {
            mode == Mode.Systems -> equation.text + ";" + equation2.text + ";" + equation3.text
            twoFields -> equation.text + ";" + equation2.text
            else -> equation.text
        }
        val blank = equation.text.isBlank() && equation2.text.isBlank() && equation3.text.isBlank()
        previewJob = viewModelScope.launch {
            delay(250)
            val r = Engine.preview(getApplication(), if (blank) "" else text, previewKind)
            previewLatex = if (r.ok) r.latex else ""
            previewError = if (r.ok) null else r.error
            if (r.ok) previewOrder = r.order
        }
    }

    fun setEquationText(text: String) {
        onEquationChange(TextFieldValue(text, TextRange(text.length)))
    }

    fun setEquation2Text(text: String) {
        onEquation2Change(TextFieldValue(text, TextRange(text.length)))
    }

    fun setEquation3Text(text: String) {
        onEquation3Change(TextFieldValue(text, TextRange(text.length)))
    }

    /** Inserts text at the cursor of the active field (used by the math key row). */
    fun insert(piece: String) {
        val field = when {
            mode == Mode.Systems && activeField == 2 -> 2
            twoFields && activeField == 1 -> 1
            else -> 0
        }
        val current = when (field) {
            2 -> equation3
            1 -> equation2
            else -> equation
        }
        val start = current.selection.min
        val end = current.selection.max
        val text = current.text.substring(0, start) + piece + current.text.substring(end)
        val updated = TextFieldValue(text, TextRange(start + piece.length))
        when (field) {
            2 -> onEquation3Change(updated)
            1 -> onEquation2Change(updated)
            else -> onEquationChange(updated)
        }
    }

    /** The section key used by launch intents and the history ("first", "inverse", "eigen", ...). */
    private val modeKey: String
        get() = when (mode) {
            Mode.FirstOrder -> "first"
            Mode.SecondOrder -> "second"
            Mode.Numerical -> "numeric"
            Mode.Laplace -> if (inverse) "inverse" else "laplace"
            Mode.Systems -> "system"
            Mode.Fourier -> "fourier"
            Mode.Boundary -> if (eigenMode) "eigen" else "boundary"
            // "laplace" alone already names the Laplace-transform section.
            Mode.Pde -> if (pdeKind == "laplace") "laplace_eq" else pdeKind
            Mode.LinearAlgebra, Mode.Calculus, Mode.Series, Mode.Complex,
            Mode.Special, Mode.FourierTransform, Mode.Recurrence, Mode.Dynamics, Mode.Vector -> mode.toolName() ?: "first"
        }

    /** Every input box of the current section, by the names [restore] understands. */
    private fun snapshot(): Map<String, String> {
        val fields = LinkedHashMap<String, String>()
        fields["equation"] = equation.text
        when (mode) {
            Mode.FirstOrder, Mode.SecondOrder -> {
                fields["x0"] = x0; fields["y0"] = y0
                if (isSecondOrder) fields["yp0"] = yPrime0
            }
            Mode.Numerical -> {
                fields["x0"] = x0; fields["y0"] = y0
                fields["h"] = stepSize; fields["n"] = stepCount
            }
            Mode.Laplace -> {}
            Mode.LinearAlgebra, Mode.Calculus, Mode.Series, Mode.Complex,
            Mode.Special, Mode.FourierTransform, Mode.Recurrence, Mode.Dynamics, Mode.Vector -> {
                fields["op"] = toolOp
                fields["x0"] = x0; fields["y0"] = y0
            }
            Mode.Systems -> {
                fields["equation2"] = equation2.text
                fields["equation3"] = equation3.text
                fields["x0"] = x0; fields["y0"] = y0
                if (equation3.text.isNotBlank()) fields["z0"] = z0
            }
            Mode.Fourier -> {
                fields["equation2"] = equation2.text
                fields["x0"] = x0
            }
            Mode.Pde -> {
                fields["equation2"] = equation2.text
                fields["x0"] = x0; fields["y0"] = y0
                fields["bc"] = if (pdeKind == "laplace") pdeSide else pdeBc
                if ((pdeKind == "heat" && pdeBc == "tt") || pdeKind in PDE_TOOLS) {
                    fields["t1"] = pdeT1; fields["t2"] = pdeT2
                }
                if (pdeKind == "fd") fields["steps"] = pdeSteps
            }
            Mode.Boundary -> if (eigenMode) {
                fields["equation"] = ""
                fields["bc"] = eigenKind
                fields["x0"] = x0
            } else {
                fields["a"] = pointA; fields["va"] = valueA
                fields["ka"] = if (slopeAtA) "dy" else "y"
                fields["b"] = pointB; fields["vb"] = valueB
                fields["kb"] = if (slopeAtB) "dy" else "y"
            }
        }
        return fields
    }

    /** A one-line plain-text description of the current problem for the history list. */
    private fun historyLabel(): String = when (mode) {
        Mode.Systems -> "x' = ${equation.text},  y' = ${equation2.text}" +
            (if (equation3.text.isNotBlank()) ",  z' = ${equation3.text}" else "") +
            (if (x0.isNotBlank()) ",  x(0) = $x0,  y(0) = $y0" else "")
        Mode.Fourier -> "f(x) = ${equation.text}" +
            (if (equation2.text.isNotBlank()) " (x > 0), ${equation2.text} (x < 0)" else "") + ",  L = $x0"
        Mode.Laplace -> if (inverse) "Inverse of ${equation.text}" else "Transform of ${equation.text}"
        Mode.Numerical -> "${equation.text},  y($x0) = $y0,  h = $stepSize,  $stepCount steps"
        Mode.Boundary -> if (eigenMode) {
            "y'' + \u03bby = 0,  conditions $eigenKind,  L = $x0"
        } else {
            "${equation.text},  ${if (slopeAtA) "y'" else "y"}($pointA) = $valueA,  " +
                "${if (slopeAtB) "y'" else "y"}($pointB) = $valueB"
        }
        Mode.Pde -> when (pdeKind) {
            "wave" -> "Wave: f = ${equation.text.ifBlank { "0" }},  g = ${equation2.text.ifBlank { "0" }},  " +
                "L = $x0,  c = $y0"
            "laplace" -> "Laplace: u = ${equation.text} on the $pdeSide side,  a = $x0,  b = $y0"
            "characteristics" -> "Characteristics: ($x0) u_x + ($y0) u_t = (${pdeT1.ifBlank { "0" }}) u + " +
                "(${pdeT2.ifBlank { "0" }})" + (if (equation.text.isNotBlank()) ",  u(x,0) = ${equation.text}" else "")
            "classify" -> "Classify: A = ${x0.ifBlank { "0" }},  B = ${y0.ifBlank { "0" }},  " +
                "C = ${pdeT1.ifBlank { "0" }}"
            "fd" -> "Finite differences: f = ${equation.text},  L = $x0,  k = $y0,  N = $pdeT1,  " +
                "dt = $pdeT2,  $pdeSteps steps"
            "disk" -> "Disk: u = ${equation.text} on r = $x0"
            else -> "Heat: f = ${equation.text},  L = $x0,  k = $y0"
        }
        Mode.LinearAlgebra, Mode.Calculus, Mode.Series, Mode.Complex,
            Mode.Special, Mode.FourierTransform, Mode.Recurrence, Mode.Dynamics, Mode.Vector ->
            "$toolOp: " + equation.text.replace("\n", "; ") +
                (if (x0.isNotBlank()) ",  $x0" else "") + (if (y0.isNotBlank()) ",  $y0" else "")
        Mode.FirstOrder, Mode.SecondOrder -> equation.text +
            (if (x0.isNotBlank()) ",  y($x0) = $y0" else "") +
            (if (isSecondOrder && yPrime0.isNotBlank()) ",  y'($x0) = $yPrime0" else "")
    }

    private fun record() {
        history = History.add(
            history,
            History.entry(modeKey, mode.title, historyLabel(), snapshot(), System.currentTimeMillis()),
        )
        historyStore.save(history)
    }

    fun toggleFavorite(entry: HistoryEntry) {
        history = History.toggleFavorite(history, entry.key)
        historyStore.save(history)
    }

    fun clearRecent() {
        history = History.clearRecent(history)
        historyStore.save(history)
    }

    /**
     * Fills the input screen from saved fields and solves. This is the one
     * way a stored problem is reopened: launch intents and history entries
     * both come through here. [key] is a section key as in [modeKey].
     */
    fun restore(key: String, fields: Map<String, String>, openTab: Int = 0) {
        open(
            when (key) {
                "second" -> Mode.SecondOrder
                "numeric" -> Mode.Numerical
                "laplace", "inverse" -> Mode.Laplace
                "system" -> Mode.Systems
                "fourier" -> Mode.Fourier
                "boundary", "eigen" -> Mode.Boundary
                "heat", "wave", "laplace_eq", "characteristics", "classify", "fd", "disk" -> Mode.Pde
                "linalg" -> Mode.LinearAlgebra
                "calculus" -> Mode.Calculus
                "series" -> Mode.Series
                "complex" -> Mode.Complex
                "special" -> Mode.Special
                "ftransform" -> Mode.FourierTransform
                "recurrence" -> Mode.Recurrence
                "dynamics" -> Mode.Dynamics
                "vector" -> Mode.Vector
                else -> Mode.FirstOrder
            }
        )
        // A math tool: pick its operation before the boxes are filled (choosing clears them).
        if (mode.toolName() != null) fields["op"]?.let { chooseTool(it) }
        chooseInverse(key == "inverse")
        chooseEigen(key == "eigen")
        val pdeProblem = key == "heat" || key == "wave" || key == "laplace_eq" || key in PDE_TOOLS
        if (pdeProblem) choosePde(if (key == "laplace_eq") "laplace" else key)
        setEquationText(fields["equation"] ?: "")
        setEquation2Text(fields["equation2"] ?: "")
        setEquation3Text(fields["equation3"] ?: "")
        pointA = fields["a"] ?: ""
        valueA = fields["va"] ?: ""
        slopeAtA = fields["ka"] == "dy"
        pointB = fields["b"] ?: ""
        valueB = fields["vb"] ?: ""
        slopeAtB = fields["kb"] == "dy"
        val condition = fields["bc"]
        if (condition != null) {
            if (!pdeProblem) {
                eigenKind = condition
            } else if (key == "laplace_eq") {
                chooseSide(condition)
            } else {
                pdeBc = condition
            }
        }
        pdeT1 = fields["t1"] ?: ""
        pdeT2 = fields["t2"] ?: ""
        fields["steps"]?.let { pdeSteps = it }
        x0 = fields["x0"] ?: ""
        y0 = fields["y0"] ?: ""
        yPrime0 = fields["yp0"] ?: ""
        z0 = fields["z0"] ?: ""
        fields["h"]?.let { stepSize = it }
        fields["n"]?.let { stepCount = it }
        solve(openTab)
    }

    /** True when enough has been typed to press Solve. */
    val canSolve: Boolean
        get() = when {
            mode == Mode.Boundary && eigenMode -> x0.isNotBlank()
            mode == Mode.Systems -> equation.text.isNotBlank() && equation2.text.isNotBlank()
            mode == Mode.Pde && pdeKind == "wave" -> equation.text.isNotBlank() || equation2.text.isNotBlank()
            // Classification takes only coefficients; characteristics can do without f.
            mode == Mode.Pde && pdeKind == "classify" ->
                x0.isNotBlank() || y0.isNotBlank() || pdeT1.isNotBlank()
            mode == Mode.Pde && pdeKind == "characteristics" -> x0.isNotBlank() && y0.isNotBlank()
            else -> equation.text.isNotBlank()
        }

    fun solve(openTab: Int = 0) {
        if (solving || !canSolve) return
        solving = true
        inputError = null
        viewModelScope.launch {
            val mathTool = mode.toolName()
            val (json, parsed) = if (mathTool != null) {
                Engine.tool(getApplication(), mathTool, toolOp, equation.text, x0, y0)
            } else if (mode == Mode.Numerical) {
                Engine.numeric(getApplication(), equation.text, x0, y0, stepSize, stepCount)
            } else if (mode == Mode.Laplace) {
                Engine.transform(getApplication(), equation.text, inverse)
            } else if (mode == Mode.Systems) {
                Engine.system(
                    getApplication(), equation.text, equation2.text, x0, y0,
                    equation3.text, if (equation3.text.isNotBlank()) z0 else "",
                )
            } else if (mode == Mode.Fourier) {
                Engine.fourier(getApplication(), equation.text, equation2.text, x0)
            } else if (mode == Mode.Pde) {
                Engine.pde(
                    getApplication(), pdeKind, equation.text,
                    when (pdeKind) {
                        "wave" -> equation2.text
                        "fd" -> pdeSteps
                        else -> ""
                    },
                    x0, y0,
                    if (pdeKind == "laplace") pdeSide else pdeBc, pdeT1, pdeT2,
                )
            } else if (mode == Mode.Boundary && eigenMode) {
                Engine.eigenProblem(getApplication(), eigenKind, x0)
            } else if (mode == Mode.Boundary) {
                Engine.boundary(
                    getApplication(), equation.text,
                    pointA, if (slopeAtA) "dy" else "y", valueA,
                    pointB, if (slopeAtB) "dy" else "y", valueB,
                )
            } else {
                // y'(x0) only means something for a second-order equation.
                val slope = if (isSecondOrder || equation.text.contains("d2y")) yPrime0 else ""
                Engine.solve(getApplication(), equation.text, x0, y0, slope)
            }
            solving = false
            if (!parsed.ok) {
                inputError = parsed.error
                return@launch
            }
            record()
            resultJson = json
            result = parsed
            methodId = parsed.applicableMethods.firstOrNull()?.id
            // With no symbolic method the graph is the useful part.
            tab = if (parsed.applicableMethods.isEmpty() && parsed.rhsExpr != null) 1 else openTab
            tappedPoints.clear()
            constant = 1f
            constant2 = 1f
            partialIndex = 0
            pdeTime = 0f
            pdePlaying = false
            viewResetKey++
            screen = Screen.Result
        }
    }

    fun back(): Boolean = when (screen) {
        Screen.Result -> { screen = Screen.Input; true }
        Screen.Input -> { screen = Screen.Home; true }
        Screen.Home -> false
    }
}
