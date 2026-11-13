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

class SolverViewModel(app: Application) : AndroidViewModel(app) {
    var screen by mutableStateOf(Screen.Home)

    var equation by mutableStateOf(TextFieldValue(""))
        private set
    var x0 by mutableStateOf("")
    var y0 by mutableStateOf("")

    var previewLatex by mutableStateOf("")
        private set
    var previewError by mutableStateOf<String?>(null)
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

    fun onEquationChange(value: TextFieldValue) {
        val textChanged = value.text != equation.text
        equation = value
        if (!textChanged) return
        inputError = null
        previewJob?.cancel()
        previewJob = viewModelScope.launch {
            delay(250)
            val r = Engine.preview(getApplication(), value.text)
            previewLatex = if (r.ok) r.latex else ""
            previewError = if (r.ok) null else r.error
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
            val (json, parsed) = Engine.solve(getApplication(), equation.text, x0, y0)
            solving = false
            if (!parsed.ok) {
                inputError = parsed.error
                return@launch
            }
            resultJson = json
            result = parsed
            methodId = parsed.applicableMethods.firstOrNull()?.id
            // With no symbolic method the graph is the useful part.
            tab = if (parsed.applicableMethods.isEmpty()) 1 else openTab
            tappedPoints.clear()
            constant = 1f
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
