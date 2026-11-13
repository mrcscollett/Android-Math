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
import com.androidmath.core.Method
import com.androidmath.core.SolveResult
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

enum class Screen { Home, Input, Result }

/** Which section of the app the input screen is serving. */
enum class Mode(val title: String) {
    FirstOrder("First-order equation"),
    SecondOrder("Second-order equation"),
    Numerical("Numerical methods"),
    Laplace("Laplace transforms"),
}

class SolverViewModel(app: Application) : AndroidViewModel(app) {
    var screen by mutableStateOf(Screen.Home)
    var mode by mutableStateOf(Mode.FirstOrder)
        private set

    var equation by mutableStateOf(TextFieldValue(""))
        private set
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
    /** Order of the equation as the engine reads it (1 or 2). */
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
            (mode == Mode.FirstOrder && (previewOrder == 2 || equation.text.contains("''")))

    private val previewKind: String
        get() = when {
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
            x0 = ""; y0 = ""; yPrime0 = ""
            previewLatex = ""; previewError = null; previewOrder = 1; inputError = null
        }
        screen = Screen.Input
    }

    fun onEquationChange(value: TextFieldValue) {
        val textChanged = value.text != equation.text
        equation = value
        if (!textChanged) return
        inputError = null
        previewJob?.cancel()
        previewJob = viewModelScope.launch {
            delay(250)
            val r = Engine.preview(getApplication(), value.text, previewKind)
            previewLatex = if (r.ok) r.latex else ""
            previewError = if (r.ok) null else r.error
            if (r.ok) previewOrder = r.order
        }
    }

    fun setEquationText(text: String) {
        onEquationChange(TextFieldValue(text, TextRange(text.length)))
    }

    /** Inserts text at the cursor (used by the math key row). */
    fun insert(piece: String) {
        val current = equation
        val start = current.selection.min
        val end = current.selection.max
        val text = current.text.substring(0, start) + piece + current.text.substring(end)
        onEquationChange(TextFieldValue(text, TextRange(start + piece.length)))
    }

    fun solve(openTab: Int = 0) {
        if (solving || equation.text.isBlank()) return
        solving = true
        inputError = null
        viewModelScope.launch {
            val (json, parsed) = if (mode == Mode.Numerical) {
                Engine.numeric(getApplication(), equation.text, x0, y0, stepSize, stepCount)
            } else if (mode == Mode.Laplace) {
                Engine.transform(getApplication(), equation.text, inverse)
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
            resultJson = json
            result = parsed
            methodId = parsed.applicableMethods.firstOrNull()?.id
            // With no symbolic method the graph is the useful part.
            tab = if (parsed.applicableMethods.isEmpty() && parsed.rhsExpr != null) 1 else openTab
            tappedPoints.clear()
            constant = 1f
            constant2 = 1f
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
