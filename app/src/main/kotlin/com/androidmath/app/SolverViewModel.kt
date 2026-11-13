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
    Systems("System of two equations"),
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
    /** Which equation field the math keys type into (0 or 1). */
    var activeField by mutableStateOf(0)
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

    private val previewKind: String
        get() = when {
            mode == Mode.Systems -> "system"
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
            activeField = 0
            x0 = ""; y0 = ""; yPrime0 = ""
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

    private fun refreshPreview() {
        inputError = null
        previewJob?.cancel()
        val text = if (mode == Mode.Systems) equation.text + ";" + equation2.text else equation.text
        val blank = equation.text.isBlank() && equation2.text.isBlank()
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

    /** Inserts text at the cursor of the active field (used by the math key row). */
    fun insert(piece: String) {
        val second = mode == Mode.Systems && activeField == 1
        val current = if (second) equation2 else equation
        val start = current.selection.min
        val end = current.selection.max
        val text = current.text.substring(0, start) + piece + current.text.substring(end)
        val updated = TextFieldValue(text, TextRange(start + piece.length))
        if (second) onEquation2Change(updated) else onEquationChange(updated)
    }

    /** True when enough has been typed to press Solve. */
    val canSolve: Boolean
        get() = equation.text.isNotBlank() && (mode != Mode.Systems || equation2.text.isNotBlank())

    fun solve(openTab: Int = 0) {
        if (solving || !canSolve) return
        solving = true
        inputError = null
        viewModelScope.launch {
            val (json, parsed) = if (mode == Mode.Numerical) {
                Engine.numeric(getApplication(), equation.text, x0, y0, stepSize, stepCount)
            } else if (mode == Mode.Laplace) {
                Engine.transform(getApplication(), equation.text, inverse)
            } else if (mode == Mode.Systems) {
                Engine.system(getApplication(), equation.text, equation2.text, x0, y0)
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
