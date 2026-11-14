@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package com.androidmath.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.androidmath.app.Mode
import com.androidmath.app.SolverViewModel

/** A ready-made problem for one operation of a math tool. */
private data class ToolExample(val label: String, val text: String, val box1: String = "", val box2: String = "")

/**
 * One operation of a math tool: what the main field holds, which small boxes
 * it needs (null when unused), and its keys, hint and examples.
 */
private data class ToolOp(
    val code: String,
    val label: String,
    val field: String,
    val placeholder: String,
    val button: String,
    val keys: List<String>,
    val hint: String,
    val examples: List<ToolExample>,
    val box1: String? = null,
    val box2: String? = null,
    val multiline: Boolean = false,
    val wideBox: Boolean = false,
)

private val MATRIX_KEYS = listOf(";", "-", "/", ".", "pi", "sqrt(")
private val FUNCTION_KEYS = listOf(
    "x", "^", "(", ")", "*", "/", "+", "-", "e^(", "sin(", "cos(", "tan(", "ln(", "sqrt(", "pi",
)
private val TERM_KEYS = listOf("n", "^", "(", ")", "*", "/", "+", "-", "!", "(-1)^n", "ln(", "sqrt(", "e^(")
private val COMPLEX_KEYS = listOf("i", "^", "(", ")", "*", "/", "+", "-", "e^(", "pi", "sqrt(")

private const val MATRIX_HINT =
    "Type the rows one per line, or separate them with a semicolon. Separate the entries of a " +
        "row with spaces. Entries can be fractions such as 1/2. Up to 4 by 4."
private const val FUNCTION_HINT =
    "Use ^ for powers and * or a space between a variable and a function, as in x*sin(x)."

private val LINEAR_ALGEBRA = listOf(
    ToolOp(
        "rref", "Row reduce", "Matrix A", "1 2 3; 4 5 6", "Row reduce", MATRIX_KEYS, MATRIX_HINT,
        listOf(ToolExample("3 by 3", "1 2 3; 4 5 6; 7 8 10"), ToolExample("Rank 1", "1 2; 2 4")),
        multiline = true,
    ),
    ToolOp(
        "solve", "Solve Ax = b", "Matrix A", "2 1; 1 3", "Solve the system", MATRIX_KEYS,
        "$MATRIX_HINT Give b as one number per row.",
        listOf(
            ToolExample("One solution", "2 1; 1 3", "5 10"),
            ToolExample("Three unknowns", "1 1 1; 0 2 5; 2 5 -1", "6 -4 27"),
            ToolExample("No solution", "1 2; 2 4", "3 7"),
            ToolExample("Infinitely many", "1 2; 2 4", "3 6"),
        ),
        box1 = "Right-hand side b", multiline = true, wideBox = true,
    ),
    ToolOp(
        "det", "Determinant", "Square matrix A", "1 2; 3 4", "Find the determinant", MATRIX_KEYS, MATRIX_HINT,
        listOf(ToolExample("2 by 2", "1 2; 3 4"), ToolExample("3 by 3", "2 0 1; 1 3 2; 1 1 4"),
            ToolExample("4 by 4", "1 2 3 4; 0 1 0 2; 2 0 1 1; 1 1 1 1")),
        multiline = true,
    ),
    ToolOp(
        "inverse", "Inverse", "Square matrix A", "2 1; 5 3", "Find the inverse", MATRIX_KEYS, MATRIX_HINT,
        listOf(ToolExample("2 by 2", "2 1; 5 3"), ToolExample("3 by 3", "1 2 3; 0 1 4; 5 6 0"),
            ToolExample("Singular", "1 2; 2 4")),
        multiline = true,
    ),
    ToolOp(
        "eigen", "Eigenvalues", "Square matrix A", "4 1; 2 3", "Find the eigenvalues", MATRIX_KEYS, MATRIX_HINT,
        listOf(ToolExample("Two real", "4 1; 2 3"), ToolExample("Complex", "0 1; -1 0"),
            ToolExample("Not diagonalizable", "2 1 0; 0 2 0; 0 0 3"), ToolExample("3 by 3", "2 0 0; 0 3 4; 0 4 9")),
        multiline = true,
    ),
    ToolOp(
        "spaces", "Rank, null space", "Matrix A", "1 2 3; 2 4 6; 1 0 1", "Find the subspaces", MATRIX_KEYS,
        MATRIX_HINT, listOf(ToolExample("Rank 2", "1 2 3; 2 4 6; 1 0 1")), multiline = true,
    ),
    ToolOp(
        "independence", "Independence", "Vectors as the columns", "1 0; 0 1; 1 1", "Test the vectors",
        MATRIX_KEYS, "Put each vector in a column: the first row holds the first entry of every vector.",
        listOf(ToolExample("Independent", "1 0; 0 1; 1 1"), ToolExample("Dependent", "1 2; 2 4; 3 6")),
        multiline = true,
    ),
    ToolOp(
        "wronskian", "Wronskian", "Functions of x, separated by commas", "e^x, e^(2x)", "Find the Wronskian",
        FUNCTION_KEYS + ",", "Enter two to four functions of x separated by commas.",
        listOf(ToolExample("Exponentials", "e^x, e^(2x)"), ToolExample("Sine and cosine", "sin(x), cos(x)"),
            ToolExample("Dependent", "x, 2x")),
    ),
)

private val CALCULUS = listOf(
    ToolOp(
        "integral", "Integral", "f(x)", "x e^x", "Integrate step by step", FUNCTION_KEYS, FUNCTION_HINT,
        listOf(ToolExample("By parts", "x e^x"), ToolExample("Substitution", "x cos(x^2)"),
            ToolExample("Partial fractions", "1/(x^2-1)"), ToolExample("Trig identity", "sin(x)^2"),
            ToolExample("Parts twice", "e^x sin(x)")),
    ),
    ToolOp(
        "definite", "Definite integral", "f(x)", "e^(-x)", "Evaluate the integral", FUNCTION_KEYS,
        "$FUNCTION_HINT A limit can be a number, pi, oo or -oo (infinity).",
        listOf(ToolExample("Area", "x^2", "0", "3"), ToolExample("Improper", "e^(-x)", "0", "oo"),
            ToolExample("Diverges", "1/x", "1", "oo"), ToolExample("Whole line", "1/(1+x^2)", "-oo", "oo")),
        box1 = "from", box2 = "to",
    ),
    ToolOp(
        "fractions", "Partial fractions", "Ratio of polynomials in x", "(5x-4)/(x^2-x-2)",
        "Decompose", FUNCTION_KEYS, "Enter a fraction whose top and bottom are polynomials in x.",
        listOf(ToolExample("Two linear factors", "(5x-4)/(x^2-x-2)"),
            ToolExample("Repeated and quadratic", "(2x+1)/((x-1)^2 (x^2+1))"),
            ToolExample("Divide first", "(x^3+1)/(x^2-4)")),
    ),
    ToolOp(
        "derivative", "Derivative", "f(x)", "x^2 sin(x)", "Differentiate", FUNCTION_KEYS, FUNCTION_HINT,
        listOf(ToolExample("Product", "x^2 sin(x)"), ToolExample("Quotient", "sin(x)/x"),
            ToolExample("Chain", "e^(3x^2)")),
    ),
)

private val SERIES = listOf(
    ToolOp(
        "taylor", "Taylor series", "f(x)", "sin(x)", "Find the series", FUNCTION_KEYS,
        "$FUNCTION_HINT Leave the centre empty for 0 and the order empty for 6.",
        listOf(ToolExample("e^x", "e^x"), ToolExample("sin x", "sin(x)", "", "7"),
            ToolExample("1/(1-x)", "1/(1-x)"), ToolExample("ln x about 1", "ln(x)", "1", "5")),
        box1 = "centre a", box2 = "order",
    ),
    ToolOp(
        "convergence", "Convergence tests", "General term a_n", "1/n^2", "Test the series", TERM_KEYS,
        "The series is the sum of a_n from n = 1. Write a factorial as n! and an alternating sign as (-1)^n.",
        listOf(ToolExample("Harmonic", "1/n"), ToolExample("p-series", "1/n^2"),
            ToolExample("Alternating", "(-1)^n/n"), ToolExample("Geometric", "(1/2)^n"),
            ToolExample("Factorial", "n!/n^n")),
    ),
    ToolOp(
        "radius", "Radius of convergence", "Coefficient c_n", "1/n", "Find the radius", TERM_KEYS,
        "The power series is the sum of c_n (x - a)^n. Leave the centre empty for 0.",
        listOf(ToolExample("1/n", "1/n"), ToolExample("1/n!", "1/n!"), ToolExample("2^n", "2^n"),
            ToolExample("Centre 1", "1/n^2", "1")),
        box1 = "centre a",
    ),
)

private val COMPLEX = listOf(
    ToolOp(
        "evaluate", "Simplify", "Expression with i", "(1 + 2i)/(3 - i)", "Work it out", COMPLEX_KEYS,
        "Write the imaginary unit as i. The result is given in rectangular, polar and exponential form.",
        listOf(ToolExample("Division", "(1 + 2i)/(3 - i)"), ToolExample("Power", "(1 + i)^8"),
            ToolExample("Euler", "e^(i pi/3)"), ToolExample("Root of a negative", "sqrt(-4)")),
    ),
    ToolOp(
        "roots", "n-th roots", "Complex number z", "-8", "Find the roots", COMPLEX_KEYS,
        "All n roots are listed and drawn in the complex plane. n is a whole number from 2 to 12.",
        listOf(ToolExample("Cube roots of 1", "1", "3"), ToolExample("Cube roots of -8", "-8", "3"),
            ToolExample("Square roots of 1+i", "1+i", "2")),
        box1 = "n",
    ),
)

private fun operationsFor(mode: Mode): List<ToolOp> = when (mode) {
    Mode.Calculus -> CALCULUS
    Mode.Series -> SERIES
    Mode.Complex -> COMPLEX
    else -> LINEAR_ALGEBRA
}

/** The input form shared by the linear algebra, calculus, series and complex-number tools. */
@Composable
fun ToolInputScreen(model: SolverViewModel) {
    val keyboard = LocalSoftwareKeyboardController.current
    val submit = {
        keyboard?.hide()
        model.solve()
    }
    val operations = operationsFor(model.mode)
    val op = operations.firstOrNull { it.code == model.toolOp } ?: operations[0]

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(model.mode.title) },
                navigationIcon = { TextButton(onClick = { model.back() }) { Text("Back") } },
            )
        },
    ) { padding ->
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
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (choice in operations) {
                        FilterChip(
                            selected = choice.code == op.code,
                            onClick = { model.chooseTool(choice.code) },
                            label = { Text(choice.label) },
                        )
                    }
                }

                OutlinedTextField(
                    value = model.equation,
                    onValueChange = model::onEquationChange,
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                    label = { Text(op.field) },
                    placeholder = { Text(op.placeholder) },
                    singleLine = !op.multiline,
                    minLines = if (op.multiline) 3 else 1,
                    textStyle = MaterialTheme.typography.bodyLarge.copy(fontFamily = FontFamily.Monospace),
                    isError = model.previewError != null || model.inputError != null,
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.None,
                        autoCorrectEnabled = false,
                        keyboardType = KeyboardType.Ascii,
                        // In a matrix, Enter starts a new row instead of solving.
                        imeAction = if (op.multiline) ImeAction.Default else ImeAction.Done,
                    ),
                    keyboardActions = KeyboardActions(onDone = { submit() }),
                )

                FlowRow(
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    for (key in op.keys) {
                        FilledTonalButton(
                            onClick = { model.insert(key) },
                            contentPadding = PaddingValues(horizontal = 12.dp),
                            modifier = Modifier.height(40.dp),
                        ) { Text(key, fontFamily = FontFamily.Monospace) }
                    }
                }

                OutlinedCard(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                    Box(
                        Modifier.fillMaxWidth().height(if (op.multiline) 112.dp else 64.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        val error = model.previewError
                        val hint = when {
                            error != null -> ""
                            model.engineReady -> "What you type appears here"
                            else -> "Starting the solver…"
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

                val first = op.box1
                val second = op.box2
                if (first != null) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        modifier = Modifier.padding(top = 12.dp),
                    ) {
                        ToolBox(
                            model.x0, { model.x0 = it }, first,
                            if (op.wideBox) Modifier.fillMaxWidth() else Modifier.width(140.dp),
                        )
                        if (second != null) {
                            ToolBox(model.y0, { model.y0 = it }, second, Modifier.width(140.dp))
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
                        Text(if (model.engineReady) "Working…" else "Starting the solver…")
                    } else {
                        Text(op.button)
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
                    for (example in op.examples) {
                        AssistChip(
                            onClick = {
                                model.setEquationText(example.text)
                                model.x0 = example.box1
                                model.y0 = example.box2
                            },
                            label = { Text(example.label) },
                        )
                    }
                }
                Text(
                    op.hint,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 12.dp, bottom = 24.dp),
                )
            }
        }
    }
}

@Composable
private fun ToolBox(value: String, onChange: (String) -> Unit, label: String, modifier: Modifier) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        modifier = modifier,
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(
            autoCorrectEnabled = false,
            keyboardType = KeyboardType.Ascii,
            imeAction = ImeAction.Next,
        ),
    )
}
