@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package com.androidmath.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.androidmath.app.Mode
import com.androidmath.app.SolverViewModel

private val KEYS = listOf(
    "y'", "y''", "y'''", "x", "y", "^", "(", ")", "*", "/", "+", "-", "=",
    "e^(", "sin(", "cos(", "tan(", "ln(", "sqrt(", "pi",
)

private val KEYS_T = listOf(
    "t", "^", "(", ")", "*", "/", "+", "-",
    "e^(", "sin(", "cos(", "sinh(", "cosh(", "sqrt(", "u(", "delta(", "pi",
)

private val KEYS_S = listOf("s", "^", "(", ")", "*", "/", "+", "-", "e^(-", "sqrt(", "pi")

private val FORWARD_EXAMPLES = listOf(
    Example("e^(2t) sin(3t)"), Example("t cos(2t)"), Example("t^2 + 2"),
    Example("(t-2) u(t-2)"), Example("5 e^(-t) - 3 sinh(2t)"),
)

private val INVERSE_EXAMPLES = listOf(
    Example("1/(s(s+1))"), Example("(s+3)/(s^2+2s+5)"), Example("(2s+1)/((s-1)(s^2+4))"),
    Example("e^(-2s)/(s^2+1)"), Example("1/(s^2+1)^2"),
)

/**
 * An example problem: equation, x0, y0 and (for second order) y'(x0).
 * For a system, [equation] is x', [second] is y', and x0, y0 are x(0), y(0).
 */
private data class Example(
    val equation: String, val x0: String = "", val y0: String = "", val slope: String = "",
    val second: String = "",
)

private val KEYS_XY = listOf(
    "x", "y", "^", "(", ")", "*", "/", "+", "-", "sin(", "cos(", "e^(", "sqrt(", "pi",
)

private val SYSTEM_EXAMPLES = listOf(
    Example("x + 2y", "1", "0", second = "3x + 2y"),
    Example("y", "1", "0", second = "-x"),
    Example("-x - 2y", "1", "1", second = "2x - y"),
    Example("x - y", second = "x + 3y"),
    Example("x(1 - y)", second = "y(x - 1)"),
    Example("y", second = "-sin(x)"),
)

private val FIRST_ORDER_EXAMPLES = listOf(
    Example("y' = x*y", "0", "1"),
    Example("y' + 2y = x"),
    Example("y' = y(1-y)", "0", "1/2"),
    Example("(2x + y) + (x + 2y) y' = 0"),
    Example("y' + y = x y^2", "0", "1"),
    Example("y' = (x^2 + y^2)/(x y)", "1", "2"),
    Example("y' = x^2 + y^2"),
)

private val SECOND_ORDER_EXAMPLES = listOf(
    Example("y'' + 3y' + 2y = e^x", "0", "1", "0"),
    Example("y'' + y = 0"),
    Example("y'' + 2y' + 5y = 0", "0", "0", "1"),
    Example("y'' + y = sin(x)", "0", "0", "0"),
    Example("y'' + y = tan(x)"),
    Example("x^2 y'' + x y' - y = 0", "1", "2", "0"),
    Example("y'' + y = u(t-2)", "0", "0", "0"),
    Example("y''' - 6y'' + 11y' - 6y = 0", "0", "1", "0, 0"),
)

private val NUMERICAL_EXAMPLES = listOf(
    Example("y' = x*y", "0", "1"),
    Example("y' = x^2 + y^2", "0", "0"),
    Example("y' = y - x^2 + 1", "0", "1/2"),
    Example("y' = sin(x) - y", "0", "1"),
)

@Composable
fun InputScreen(model: SolverViewModel) {
    val keyboard = LocalSoftwareKeyboardController.current
    val submit = {
        keyboard?.hide()
        model.solve()
    }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(model.mode.title) },
                navigationIcon = { TextButton(onClick = { model.back() }) { Text("Back") } },
            )
        },
    ) { padding ->
        // Centred and capped in width so the form is not stretched on tablets.
        Box(
            modifier = Modifier.fillMaxSize().padding(padding).imePadding(),
            contentAlignment = Alignment.TopCenter,
        ) {
        Column(
            modifier = Modifier
                .widthIn(max = 640.dp)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp),
        ) {
            val laplace = model.mode == Mode.Laplace
            val system = model.mode == Mode.Systems
            if (laplace) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = !model.inverse,
                        onClick = { model.chooseInverse(false) },
                        label = { Text("Transform f(t)") },
                    )
                    FilterChip(
                        selected = model.inverse,
                        onClick = { model.chooseInverse(true) },
                        label = { Text("Inverse of F(s)") },
                    )
                }
            }
            OutlinedTextField(
                value = model.equation,
                onValueChange = model::onEquationChange,
                modifier = Modifier.fillMaxWidth().onFocusChanged { if (it.isFocused) model.activeField = 0 },
                label = {
                    Text(
                        when {
                            system -> "x' ="
                            !laplace -> "Equation"
                            model.inverse -> "F(s)"
                            else -> "f(t)"
                        }
                    )
                },
                placeholder = {
                    Text(
                        when {
                            system -> "x + 2y"
                            laplace && model.inverse -> "1/(s(s+1))"
                            laplace -> "e^(2t) sin(3t)"
                            model.mode == Mode.SecondOrder -> "y'' + 3y' + 2y = e^x"
                            else -> "y' + 2y = x"
                        }
                    )
                },
                singleLine = true,
                textStyle = MaterialTheme.typography.bodyLarge.copy(fontFamily = FontFamily.Monospace),
                isError = model.previewError != null || model.inputError != null,
                keyboardOptions = KeyboardOptions(
                    capitalization = KeyboardCapitalization.None,
                    autoCorrectEnabled = false,
                    keyboardType = KeyboardType.Ascii,
                    imeAction = ImeAction.Done,
                ),
                keyboardActions = KeyboardActions(onDone = { submit() }),
            )

            if (system) {
                OutlinedTextField(
                    value = model.equation2,
                    onValueChange = model::onEquation2Change,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp)
                        .onFocusChanged { if (it.isFocused) model.activeField = 1 },
                    label = { Text("y' =") },
                    placeholder = { Text("3x + 2y") },
                    singleLine = true,
                    textStyle = MaterialTheme.typography.bodyLarge.copy(fontFamily = FontFamily.Monospace),
                    isError = model.previewError != null || model.inputError != null,
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.None,
                        autoCorrectEnabled = false,
                        keyboardType = KeyboardType.Ascii,
                        imeAction = ImeAction.Done,
                    ),
                    keyboardActions = KeyboardActions(onDone = { submit() }),
                )
            }
            // Wraps onto extra lines so every key is visible without scrolling.
            FlowRow(
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                val keys = when {
                    system -> KEYS_XY
                    !laplace -> KEYS
                    model.inverse -> KEYS_S
                    else -> KEYS_T
                }
                for (key in keys) {
                    FilledTonalButton(
                        onClick = { model.insert(key) },
                        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp),
                        modifier = Modifier.height(40.dp),
                    ) { Text(key, fontFamily = FontFamily.Monospace) }
                }
            }

            OutlinedCard(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                Box(Modifier.fillMaxWidth().height(64.dp), contentAlignment = Alignment.Center) {
                    // The web view stays in place; an error is drawn over it rather
                    // than swapping the view in and out as the user types.
                    val error = model.previewError
                    val hint = when {
                        error != null -> ""
                        model.engineReady -> if (laplace) "Your expression appears here as you type"
                        else if (system) "Your system appears here as you type"
                        else "Your equation appears here as you type"
                        else -> "Starting the solver\u2026"
                    }
                    MathWebView(
                        script = "showPreview(${jsString(if (error == null) model.previewLatex else "")}, ${jsString(hint)});",
                        modifier = Modifier.fillMaxSize(),
                    )
                    if (error != null) {
                        Text(
                            error,
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(horizontal = 12.dp),
                        )
                    }
                }
            }

            if (!laplace) {
                val numerical = model.mode == Mode.Numerical
                val second = model.isSecondOrder
                Text(
                    when {
                        system -> "Initial point at t = 0 (optional)"
                        numerical -> "Starting point"
                        second -> "Initial conditions (optional)"
                        else -> "Initial condition (optional)"
                    },
                    style = MaterialTheme.typography.labelLarge,
                    modifier = Modifier.padding(top = 16.dp, bottom = 4.dp),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    NumberField(model.x0, { model.x0 = it }, if (system) "x(0)" else "x\u2080")
                    NumberField(model.y0, { model.y0 = it }, if (system) "y(0)" else "y(x\u2080)")
                    if (second) {
                        // Third and fourth order take the remaining values as a comma-separated list.
                        if (model.previewOrder >= 3) {
                            NumberField(
                                model.yPrime0, { model.yPrime0 = it },
                                "y', y'', \u2026 at x\u2080", width = 200.dp,
                            )
                        } else {
                            NumberField(model.yPrime0, { model.yPrime0 = it }, "y'(x\u2080)")
                        }
                    }
                }
                if (numerical) {
                    Text(
                        "Step size and number of steps",
                        style = MaterialTheme.typography.labelLarge,
                        modifier = Modifier.padding(top = 16.dp, bottom = 4.dp),
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        NumberField(model.stepSize, { model.stepSize = it }, "h")
                        NumberField(model.stepCount, { model.stepCount = it }, "steps")
                    }
                }
            }

            model.inputError?.let {
                Text(
                    it,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 12.dp),
                )
            }

            Button(
                onClick = submit,
                enabled = !model.solving && model.canSolve,
                modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
            ) {
                if (model.solving) {
                    CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(if (model.engineReady) "Solving…" else "Starting the solver…")
                } else {
                    Text(
                        when (model.mode) {
                            Mode.Numerical -> "Compute step by step"
                            Mode.Laplace -> "Transform step by step"
                            else -> "Solve step by step"
                        }
                    )
                }
            }

            Text(
                "Examples",
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier.padding(top = 24.dp),
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                val examples = when (model.mode) {
                    Mode.FirstOrder -> FIRST_ORDER_EXAMPLES
                    Mode.SecondOrder -> SECOND_ORDER_EXAMPLES
                    Mode.Numerical -> NUMERICAL_EXAMPLES
                    Mode.Laplace -> if (model.inverse) INVERSE_EXAMPLES else FORWARD_EXAMPLES
                    Mode.Systems -> SYSTEM_EXAMPLES
                }
                for (example in examples) {
                    AssistChip(
                        onClick = {
                            model.setEquationText(example.equation)
                            if (system) model.setEquation2Text(example.second)
                            model.x0 = example.x0
                            model.y0 = example.y0
                            model.yPrime0 = example.slope
                        },
                        label = {
                            Text(
                                when {
                                    system && example.x0.isEmpty() ->
                                        "x' = ${example.equation},  y' = ${example.second}"
                                    system -> "x' = ${example.equation},  y' = ${example.second},  " +
                                        "x(0) = ${example.x0},  y(0) = ${example.y0}"
                                    example.x0.isEmpty() -> example.equation
                                    example.slope.contains(",") ->
                                        "${example.equation},  y(${example.x0}) = ${example.y0},  " +
                                            "y', y'' = ${example.slope}"
                                    example.slope.isEmpty() ->
                                        "${example.equation},  y(${example.x0}) = ${example.y0}"
                                    else -> "${example.equation},  y(${example.x0}) = ${example.y0},  " +
                                        "y'(${example.x0}) = ${example.slope}"
                                }
                            )
                        },
                    )
                }
            }
            Text(
                if (system) {
                    "Enter the right-hand sides in terms of x and y. Linear systems are solved " +
                        "exactly; nonlinear ones are analysed near their equilibrium points."
                } else if (laplace) {
                    "Use u(t-a) for a unit step switched on at t = a and delta(t-a) for an " +
                        "impulse. A delay in F(s) is written e^(-a s)."
                } else {
                    "Write derivatives as y' and y'' (or dy/dx). Use ^ for powers and * or a " +
                        "space between a variable and a function, as in x*sin(x). Step and " +
                        "impulse forcing is written u(t-a) and delta(t-a)."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 12.dp, bottom = 24.dp),
            )
        }
        }
    }
}

@Composable
private fun NumberField(value: String, onChange: (String) -> Unit, label: String, width: Dp = 116.dp) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        modifier = Modifier.width(width),
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(
            autoCorrectEnabled = false,
            keyboardType = KeyboardType.Ascii,
            imeAction = ImeAction.Next,
        ),
    )
}
