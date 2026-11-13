@file:OptIn(ExperimentalMaterial3Api::class)

package com.androidmath.app.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
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
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.androidmath.app.SolverViewModel

private val KEYS = listOf(
    "y'", "x", "y", "^", "(", ")", "*", "/", "+", "-", "=",
    "e^(", "sin(", "cos(", "tan(", "ln(", "sqrt(", "pi",
)

private val EXAMPLES = listOf(
    Triple("y' = x*y", "0", "1"),
    Triple("y' + 2y = x", "", ""),
    Triple("y' = y(1-y)", "0", "1/2"),
    Triple("x y' + y = x^2", "1", "1"),
    Triple("y' = x^2 + y^2", "", ""),
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
                title = { Text("First-order equation") },
                navigationIcon = { TextButton(onClick = { model.back() }) { Text("Back") } },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .imePadding()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp),
        ) {
            OutlinedTextField(
                value = model.equation,
                onValueChange = model::onEquationChange,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Equation") },
                placeholder = { Text("y' + 2y = x") },
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

            Row(
                modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                for (key in KEYS) {
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
                        model.engineReady -> "Your equation appears here as you type"
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

            Text(
                "Initial condition (optional)",
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier.padding(top = 16.dp, bottom = 4.dp),
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("y(")
                NumberField(model.x0, { model.x0 = it }, "x₀")
                Text(") = ")
                NumberField(model.y0, { model.y0 = it }, "y₀")
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
                enabled = !model.solving && model.equation.text.isNotBlank(),
                modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
            ) {
                if (model.solving) {
                    CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(if (model.engineReady) "Solving…" else "Starting the solver…")
                } else {
                    Text("Solve step by step")
                }
            }

            Text(
                "Examples",
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier.padding(top = 24.dp),
            )
            Row(
                modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                for ((equation, x0, y0) in EXAMPLES) {
                    AssistChip(
                        onClick = {
                            model.setEquationText(equation)
                            model.x0 = x0
                            model.y0 = y0
                        },
                        label = {
                            Text(if (x0.isEmpty()) equation else "$equation,  y($x0) = $y0")
                        },
                    )
                }
            }
            Text(
                "Write the derivative as y' or dy/dx. Use ^ for powers and * or a space " +
                    "between a variable and a function, as in x*sin(x).",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 12.dp, bottom = 24.dp),
            )
        }
    }
}

@Composable
private fun NumberField(value: String, onChange: (String) -> Unit, label: String) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        modifier = Modifier.width(104.dp),
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(
            autoCorrectEnabled = false,
            keyboardType = KeyboardType.Ascii,
            imeAction = ImeAction.Next,
        ),
    )
}
