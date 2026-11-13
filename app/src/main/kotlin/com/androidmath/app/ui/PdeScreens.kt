@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package com.androidmath.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
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
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.androidmath.app.SolverViewModel
import com.androidmath.core.Domain
import com.androidmath.core.Expr
import com.androidmath.core.Plot
import com.androidmath.core.Viewport
import kotlinx.coroutines.delay

/** One example problem of the PDE section. */
private data class PdeExample(
    val label: String, val f: String, val g: String, val p1: String, val p2: String,
    val bc: String = "", val t1: String = "", val t2: String = "",
)

private val HEAT_EXAMPLES = listOf(
    PdeExample("f = x(1-x), ends at 0", "x(1-x)", "", "1", "1", "00"),
    PdeExample("f = x, insulated ends", "x", "", "pi", "1", "nn"),
    PdeExample("Cold rod, ends at 0 and 100", "0", "", "1", "1", "tt", "0", "100"),
    PdeExample("f = 1, right end insulated", "1", "", "2", "1", "0n"),
)

private val WAVE_EXAMPLES = listOf(
    PdeExample("f = x(1-x), at rest", "x(1-x)", "", "1", "1"),
    PdeExample("Plucked string", "x (1 - u(x - 1/2)) + (1 - x) u(x - 1/2)", "", "1", "1"),
    PdeExample("Struck string, g = sin(pi x)", "", "sin(pi x)", "1", "1"),
    PdeExample("One mode, c = 2", "sin(pi x)", "", "1", "2"),
)

private val LAPLACE_EXAMPLES = listOf(
    PdeExample("sin(pi x) on the top", "sin(pi x)", "", "1", "1", "top"),
    PdeExample("x(2-x) on the bottom", "x(2-x)", "", "2", "1", "bottom"),
    PdeExample("y(1-y) on the left", "y(1-y)", "", "1", "1", "left"),
    PdeExample("1 on the right", "1", "", "1", "3", "right"),
)

private val HEAT_CONDITIONS = listOf(
    "00" to "Both ends at 0",
    "nn" to "Both ends insulated",
    "0n" to "Left at 0, right insulated",
    "n0" to "Left insulated, right at 0",
    "tt" to "Ends held at T₁ and T₂",
)

private val SIDES = listOf(
    "top" to "Top (y = b)", "bottom" to "Bottom (y = 0)",
    "left" to "Left (x = 0)", "right" to "Right (x = a)",
)

@Composable
fun PdeInputScreen(model: SolverViewModel) {
    val keyboard = LocalSoftwareKeyboardController.current
    val submit = {
        keyboard?.hide()
        model.solve()
    }
    val kind = model.pdeKind
    val wave = kind == "wave"
    val laplace = kind == "laplace"
    // On the left and right sides of the rectangle the data is a function of y.
    val variable = if (laplace && (model.pdeSide == "left" || model.pdeSide == "right")) "y" else "x"

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
                    FilterChip(
                        selected = kind == "heat",
                        onClick = { model.choosePde("heat") },
                        label = { Text("Heat") },
                    )
                    FilterChip(
                        selected = wave,
                        onClick = { model.choosePde("wave") },
                        label = { Text("Wave") },
                    )
                    FilterChip(
                        selected = laplace,
                        onClick = { model.choosePde("laplace") },
                        label = { Text("Laplace") },
                    )
                }
                Text(
                    when (kind) {
                        "wave" -> "u_tt = c² u_xx on 0 < x < L, both ends fixed"
                        "laplace" -> "u_xx + u_yy = 0 on 0 < x < a, 0 < y < b"
                        else -> "u_t = k u_xx on 0 < x < L"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 8.dp),
                )

                if (laplace) {
                    Text("Side that carries the values f", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        for ((code, label) in SIDES) {
                            FilterChip(
                                selected = model.pdeSide == code,
                                onClick = { model.chooseSide(code) },
                                label = { Text(label) },
                            )
                        }
                    }
                }

                PdeExpressionField(
                    value = model.equation,
                    onChange = model::onEquationChange,
                    label = when (kind) {
                        "wave" -> "Initial displacement f(x)"
                        "laplace" -> "Boundary values f($variable)"
                        else -> "Initial temperature f(x)"
                    },
                    placeholder = if (variable == "y") "y(1-y)" else "x(1-x)",
                    isError = model.previewError != null || model.inputError != null,
                    onFocus = { model.activeField = 0 },
                    onDone = { submit() },
                )
                if (wave) {
                    PdeExpressionField(
                        value = model.equation2,
                        onChange = model::onEquation2Change,
                        label = "Initial velocity g(x)  (optional)",
                        placeholder = "0",
                        isError = model.previewError != null || model.inputError != null,
                        onFocus = { model.activeField = 1 },
                        onDone = { submit() },
                    )
                }

                FlowRow(
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    val keys = listOf(variable, "^", "(", ")", "*", "/", "+", "-",
                        "sin(", "cos(", "e^(", "sqrt(", "u(", "pi")
                    for (key in keys) {
                        FilledTonalButton(
                            onClick = { model.insert(key) },
                            contentPadding = PaddingValues(horizontal = 12.dp),
                            modifier = Modifier.height(40.dp),
                        ) { Text(key, fontFamily = FontFamily.Monospace) }
                    }
                }

                OutlinedCard(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                    Box(Modifier.fillMaxWidth().height(64.dp), contentAlignment = Alignment.Center) {
                        val error = model.previewError
                        val hint = when {
                            error != null -> ""
                            model.engineReady -> "Your function appears here as you type"
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

                Text(
                    when (kind) {
                        "wave" -> "Length and wave speed"
                        "laplace" -> "Size of the rectangle"
                        else -> "Length and diffusivity"
                    },
                    style = MaterialTheme.typography.labelLarge,
                    modifier = Modifier.padding(top = 16.dp, bottom = 4.dp),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    PdeNumberField(model.x0, { model.x0 = it }, if (laplace) "a" else "L")
                    PdeNumberField(
                        model.y0, { model.y0 = it },
                        when (kind) {
                            "wave" -> "c"
                            "laplace" -> "b"
                            else -> "k"
                        },
                    )
                }

                if (kind == "heat") {
                    Text(
                        "Boundary conditions",
                        style = MaterialTheme.typography.labelLarge,
                        modifier = Modifier.padding(top = 16.dp),
                    )
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        for ((code, label) in HEAT_CONDITIONS) {
                            FilterChip(
                                selected = model.pdeBc == code,
                                onClick = { model.pdeBc = code },
                                label = { Text(label) },
                            )
                        }
                    }
                    if (model.pdeBc == "tt") {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            modifier = Modifier.padding(top = 4.dp),
                        ) {
                            PdeNumberField(model.pdeT1, { model.pdeT1 = it }, "T₁")
                            PdeNumberField(model.pdeT2, { model.pdeT2 = it }, "T₂")
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
                        Text("Solve step by step")
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
                    val examples = when (kind) {
                        "wave" -> WAVE_EXAMPLES
                        "laplace" -> LAPLACE_EXAMPLES
                        else -> HEAT_EXAMPLES
                    }
                    for (example in examples) {
                        AssistChip(
                            onClick = {
                                // Choose the side first: it decides whether f is read in x or y.
                                if (laplace) model.chooseSide(example.bc) else if (kind == "heat") model.pdeBc = example.bc
                                model.setEquationText(example.f)
                                model.setEquation2Text(example.g)
                                model.x0 = example.p1
                                model.y0 = example.p2
                                model.pdeT1 = example.t1
                                model.pdeT2 = example.t2
                            },
                            label = { Text(example.label) },
                        )
                    }
                }
                Text(
                    "Use ^ for powers and * or a space between a variable and a function. A " +
                        "function given by two formulas can be written with the unit step, as in " +
                        "the plucked-string example: u(x - 1/2) is 0 before x = 1/2 and 1 after.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 12.dp, bottom = 24.dp),
                )
            }
        }
    }
}

@Composable
private fun PdeExpressionField(
    value: androidx.compose.ui.text.input.TextFieldValue,
    onChange: (androidx.compose.ui.text.input.TextFieldValue) -> Unit,
    label: String,
    placeholder: String,
    isError: Boolean,
    onFocus: () -> Unit,
    onDone: () -> Unit,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp)
            .onFocusChanged { if (it.isFocused) onFocus() },
        label = { Text(label) },
        placeholder = { Text(placeholder) },
        singleLine = true,
        textStyle = MaterialTheme.typography.bodyLarge.copy(fontFamily = FontFamily.Monospace),
        isError = isError,
        keyboardOptions = KeyboardOptions(
            capitalization = KeyboardCapitalization.None,
            autoCorrectEnabled = false,
            keyboardType = KeyboardType.Ascii,
            imeAction = ImeAction.Done,
        ),
        keyboardActions = KeyboardActions(onDone = { onDone() }),
    )
}

@Composable
private fun PdeNumberField(value: String, onChange: (String) -> Unit, label: String) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        modifier = Modifier.width(116.dp),
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(
            autoCorrectEnabled = false,
            keyboardType = KeyboardType.Ascii,
            imeAction = ImeAction.Next,
        ),
    )
}

/** The Graph tab of a PDE result: an animated profile, or a heat map for Laplace's equation. */
@Composable
fun PdeTab(model: SolverViewModel) {
    val result = model.result ?: return
    val method = model.selectedMethod
    val solution = remember(method) { Expr.parseOrNull(method?.plot?.particularExpr) }
    val initial = remember(result.initialExpr) { Expr.parseOrNull(result.initialExpr) }
    val domain = result.domain

    if (solution == null || domain == null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(
                "There is no formula to graph for this method.",
                modifier = Modifier.padding(24.dp),
            )
        }
        return
    }
    if (result.surface) {
        HeatMapTab(solution, domain)
        return
    }

    val wave = result.pde == "wave"
    val tMax = domain.tMax.toFloat().coerceAtLeast(1e-6f)
    // Heat: the whole decay in about six seconds. Wave: one period in about five.
    val frameStep = if (wave) tMax / 2f / 150f else tMax / 180f
    LaunchedEffect(model.pdePlaying, model.viewResetKey) {
        while (model.pdePlaying) {
            delay(33)
            val next = model.pdeTime + frameStep
            if (next < tMax) {
                model.pdeTime = next
            } else if (wave) {
                model.pdeTime = next - tMax
            } else {
                model.pdeTime = tMax
                model.pdePlaying = false
            }
        }
    }

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            ProfileView(
                solution = solution,
                initial = initial,
                domain = domain,
                time = model.pdeTime.toDouble(),
                modifier = Modifier.fillMaxSize(),
            )
        }
        Column(
            modifier = Modifier
                .align(Alignment.CenterHorizontally)
                .widthIn(max = 760.dp)
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(
                    onClick = {
                        if (!model.pdePlaying && !wave && model.pdeTime >= tMax) model.pdeTime = 0f
                        model.pdePlaying = !model.pdePlaying
                    },
                ) { Text(if (model.pdePlaying) "Pause" else "Play") }
                Text("t = ${"%.3f".format(model.pdeTime)}", modifier = Modifier.width(96.dp))
                Slider(
                    value = model.pdeTime.coerceIn(0f, tMax),
                    onValueChange = {
                        model.pdePlaying = false
                        model.pdeTime = it
                    },
                    valueRange = 0f..tMax,
                    modifier = Modifier.weight(1f),
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                PdeLegend(MaterialTheme.colorScheme.primary, if (wave) "String at time t" else "Temperature at time t")
                if (initial != null) {
                    Spacer(Modifier.width(16.dp))
                    PdeLegend(MaterialTheme.colorScheme.outline, "Initial profile")
                }
            }
        }
    }
}

@Composable
private fun PdeLegend(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).background(color))
        Spacer(Modifier.width(6.dp))
        Text(label, style = MaterialTheme.typography.labelMedium)
    }
}

/** u(x, t) over the interval at one instant, on axes that stay fixed while t changes. */
@Composable
private fun ProfileView(
    solution: Expr, initial: Expr?, domain: Domain, time: Double, modifier: Modifier = Modifier,
) {
    val colors = MaterialTheme.colorScheme
    val measurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 11.sp, color = colors.onSurfaceVariant)
    val padX = (domain.xMax - domain.xMin) * 0.08
    val padU = (domain.uMax - domain.uMin) * 0.15
    val vp = Viewport(domain.xMin - padX, domain.xMax + padX, domain.uMin - padU, domain.uMax + padU)

    Canvas(modifier = modifier.clipToBounds()) {
        val w = size.width
        val h = size.height
        if (w <= 0f || h <= 0f) return@Canvas
        val hair = 1.dp.toPx()

        val xStep = Plot.tickStep(vp.xMin, vp.xMax, (w / 80.dp.toPx()).toInt().coerceAtLeast(2))
        val uStep = Plot.tickStep(vp.yMin, vp.yMax, (h / 70.dp.toPx()).toInt().coerceAtLeast(2))
        for (x in Plot.ticks(vp.xMin, vp.xMax, (w / 80.dp.toPx()).toInt().coerceAtLeast(2))) {
            val px = vp.toPxX(x, w)
            drawLine(colors.outlineVariant, Offset(px, 0f), Offset(px, h), hair)
            val layout = measurer.measure(Plot.tickLabel(x, xStep), labelStyle)
            val left = px - layout.size.width / 2f
            if (left >= 0f && left + layout.size.width <= w) {
                drawText(layout, topLeft = Offset(left, h - layout.size.height - 2.dp.toPx()))
            }
        }
        for (u in Plot.ticks(vp.yMin, vp.yMax, (h / 70.dp.toPx()).toInt().coerceAtLeast(2))) {
            val py = vp.toPxY(u, h)
            drawLine(colors.outlineVariant, Offset(0f, py), Offset(w, py), hair)
            val layout = measurer.measure(Plot.tickLabel(u, uStep), labelStyle)
            val top = py - layout.size.height / 2f
            if (top >= 0f && top + layout.size.height <= h - 16.dp.toPx()) {
                drawText(layout, topLeft = Offset(3.dp.toPx(), top))
            }
        }
        // The zero level and the two ends of the interval.
        val zero = vp.toPxY(0.0, h)
        if (zero in 0f..h) drawLine(colors.onSurface, Offset(0f, zero), Offset(w, zero), 1.5.dp.toPx())
        val leftEnd = vp.toPxX(domain.xMin, w)
        val rightEnd = vp.toPxX(domain.xMax, w)
        drawLine(colors.onSurface, Offset(leftEnd, 0f), Offset(leftEnd, h), 1.5.dp.toPx())
        drawLine(colors.onSurface, Offset(rightEnd, 0f), Offset(rightEnd, h), 1.5.dp.toPx())

        val samples = (w / 2).toInt().coerceIn(200, 800)
        val inside = { x: Double -> x >= domain.xMin - 1e-9 && x <= domain.xMax + 1e-9 }
        if (initial != null) {
            val faint = Plot.curve(
                { x -> if (inside(x)) initial.eval(x, 0.0, 0.0) else Double.NaN }, vp, w, h, samples,
            )
            drawCurves(faint, colors.outline, 1.5.dp.toPx())
        }
        val now = Plot.curve(
            { x -> if (inside(x)) solution.eval(x, time, 0.0) else Double.NaN }, vp, w, h, samples,
        )
        drawCurves(now, colors.primary, 3.dp.toPx())
    }
}

private fun DrawScope.drawCurves(lines: List<FloatArray>, color: Color, width: Float) {
    if (lines.isEmpty()) return
    val path = Path()
    for (line in lines) {
        if (line.size < 4) continue
        path.moveTo(line[0], line[1])
        var i = 2
        while (i + 1 < line.size) {
            path.lineTo(line[i], line[i + 1])
            i += 2
        }
    }
    drawPath(path, color, style = Stroke(width, cap = StrokeCap.Round, join = StrokeJoin.Round))
}

/** Laplace's equation: u over the rectangle as coloured cells, keeping the rectangle's shape. */
@Composable
private fun HeatMapTab(solution: Expr, domain: Domain) {
    val colors = MaterialTheme.colorScheme
    val cold = colors.surfaceVariant
    val hot = colors.primary
    val spanX = (domain.xMax - domain.xMin).coerceAtLeast(1e-9)
    val spanY = (domain.yMax - domain.yMin).coerceAtLeast(1e-9)
    val nx = if (spanX >= spanY) 72 else (72 * spanX / spanY).toInt().coerceIn(8, 72)
    val ny = if (spanY >= spanX) 72 else (72 * spanY / spanX).toInt().coerceIn(8, 72)
    val cells = remember(solution, domain) {
        Plot.heatMap(solution, domain.xMin, domain.xMax, domain.yMin, domain.yMax, nx, ny, domain.uMin, domain.uMax)
    }

    Column(Modifier.fillMaxSize()) {
        Canvas(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            val scale = minOf(size.width / spanX.toFloat(), size.height / spanY.toFloat())
            val rectW = spanX.toFloat() * scale
            val rectH = spanY.toFloat() * scale
            val left = (size.width - rectW) / 2f
            val top = (size.height - rectH) / 2f
            val cellW = rectW / nx
            val cellH = rectH / ny
            for (row in 0 until ny) {
                for (col in 0 until nx) {
                    val value = cells[row * nx + col]
                    if (value.isNaN()) continue
                    drawRect(
                        color = lerp(cold, hot, value),
                        topLeft = Offset(left + col * cellW, top + row * cellH),
                        // A hair of overlap hides the seams between cells.
                        size = Size(cellW + 1f, cellH + 1f),
                    )
                }
            }
            drawRect(
                color = colors.onSurface,
                topLeft = Offset(left, top),
                size = Size(rectW, rectH),
                style = Stroke(1.5.dp.toPx()),
            )
        }
        Column(
            modifier = Modifier
                .align(Alignment.CenterHorizontally)
                .widthIn(max = 760.dp)
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                PdeLegend(cold, "u = ${"%.3g".format(domain.uMin)}")
                Spacer(Modifier.width(16.dp))
                PdeLegend(hot, "u = ${"%.3g".format(domain.uMax)}")
            }
            Text(
                "The rectangle is 0 < x < ${"%.3g".format(domain.xMax)} across and " +
                    "0 < y < ${"%.3g".format(domain.yMax)} up. Darker means larger u.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
