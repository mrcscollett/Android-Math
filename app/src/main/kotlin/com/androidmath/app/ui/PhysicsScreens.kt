@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package com.androidmath.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.androidmath.app.SolverViewModel
import com.androidmath.core.PhysicsAnimation
import com.androidmath.core.PhysicsData
import com.androidmath.core.PhysicsEffect
import com.androidmath.core.PhysicsField
import com.androidmath.core.Plot
import com.androidmath.core.Samples
import com.androidmath.core.Viewport
import kotlinx.coroutines.delay

/**
 * The form for a physics model: a list of its variables, each with a value box
 * and a unit picker. The user fills in what they know; every box left empty is
 * an unknown, and Solve works out each one the known values determine.
 * Everything shown comes from the engine's catalog: the variables and their
 * sections, the optional effects (each adds its own variables), the unit
 * choices and the real examples.
 */
@Composable
fun PhysicsInputScreen(model: SolverViewModel) {
    val scenario = model.physicsScenario
    val keyboard = LocalSoftwareKeyboardController.current
    val submit = {
        keyboard?.hide()
        model.solve()
    }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(scenario?.title ?: "Physics") },
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
                if (scenario == null) {
                    Text("Starting the solver…", modifier = Modifier.padding(top = 24.dp))
                } else {
                    Text(
                        scenario.description,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )

                    Text(
                        "Fill in the values you know and leave the rest empty. " +
                            "Solve finds every empty one that your values determine.",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(top = 8.dp),
                    )

                    SectionLabel("Fill in a real example")
                    FlowRow(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        for (preset in scenario.presets) {
                            AssistChip(onClick = { model.applyPreset(preset) }, label = { Text(preset.label) })
                        }
                        AssistChip(onClick = { model.clearPhysics() }, label = { Text("Clear all") })
                    }

                    SectionLabel("The equation (" + scenario.law + ")")
                    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
                        Box(Modifier.fillMaxWidth().height(72.dp), contentAlignment = Alignment.Center) {
                            val hint = if (model.engineReady) "" else "Starting the solver…"
                            MathWebView(
                                script = "showPreview(${jsString(model.previewLatex)}, ${jsString(hint)});",
                                modifier = Modifier.fillMaxSize(),
                            )
                        }
                    }

                    if (scenario.effects.isNotEmpty()) {
                        SectionLabel("Optional effects")
                        for (effect in scenario.effects) {
                            EffectRow(effect, model.physicsEffects[effect.id] ?: effect.default) { option ->
                                model.setPhysicsEffect(effect.id, option)
                            }
                        }
                    }

                    val effects = HashMap<String, String>(model.physicsEffects)
                    for (section in scenario.sections(effects)) {
                        SectionLabel(section.title)
                        for (field in section.fields) {
                            FieldRow(
                                field = field,
                                value = model.physicsValues[field.id] ?: "",
                                unit = model.physicsUnits[field.id] ?: field.initialUnit,
                                onValue = { model.setPhysicsValue(field.id, it) },
                                onUnit = { model.setPhysicsUnit(field.id, it) },
                            )
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
                        modifier = Modifier.fillMaxWidth().padding(top = 16.dp, bottom = 24.dp),
                    ) {
                        if (model.solving) {
                            CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                            Text(if (model.engineReady) "Solving…" else "Starting the solver…")
                        } else {
                            Text("Solve step by step")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SectionLabel(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.labelLarge,
        modifier = Modifier.padding(top = 16.dp, bottom = 4.dp),
    )
}

/** One optional effect: a switch, or a row of chips when it has several alternatives. */
@Composable
private fun EffectRow(effect: PhysicsEffect, chosen: String, onChoose: (String) -> Unit) {
    if (effect.kind == "switch") {
        Row(
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f).padding(end = 12.dp)) {
                Text(effect.label, style = MaterialTheme.typography.bodyLarge)
                Text(
                    effect.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(checked = chosen == "on", onCheckedChange = { onChoose(if (it) "on" else "off") })
        }
    } else {
        Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
            Text(effect.label, style = MaterialTheme.typography.bodyLarge)
            Text(
                effect.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (option in effect.options) {
                    FilterChip(
                        selected = option.id == chosen,
                        onClick = { onChoose(option.id) },
                        label = { Text(option.label) },
                    )
                }
            }
        }
    }
}

/** One variable: its value box (empty = unknown), a clear button and the unit picker. */
@Composable
private fun FieldRow(
    field: PhysicsField, value: String, unit: String,
    onValue: (String) -> Unit, onUnit: (String) -> Unit,
) {
    Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = value,
                onValueChange = onValue,
                modifier = Modifier.weight(1f),
                label = { Text(field.label) },
                placeholder = { Text("unknown") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    capitalization = KeyboardCapitalization.None,
                    autoCorrectEnabled = false,
                    keyboardType = KeyboardType.Ascii,
                    imeAction = ImeAction.Next,
                ),
            )
            if (value.isNotEmpty()) {
                TextButton(onClick = { onValue("") }) { Text("Clear") }
            }
            if (field.units.size > 1) {
                UnitPicker(field, unit, onUnit)
            } else {
                val only = field.unitLabel(unit)
                if (only.isNotEmpty()) {
                    Text(only, modifier = Modifier.padding(start = 12.dp, end = 12.dp))
                }
            }
        }
        if (field.hint.isNotEmpty()) {
            Text(
                field.hint,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 4.dp),
            )
        }
    }
}

/** A text button showing the current unit; tapping it lists the alternatives. */
@Composable
private fun UnitPicker(field: PhysicsField, unit: String, onUnit: (String) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Box(Modifier.padding(start = 4.dp)) {
        TextButton(onClick = { expanded = true }) {
            Text(field.unitLabel(unit).ifEmpty { "no unit" } + " ▾")
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            for (choice in field.units) {
                DropdownMenuItem(
                    text = { Text(choice.label.ifEmpty { "no unit" }) },
                    onClick = {
                        onUnit(choice.id)
                        expanded = false
                    },
                )
            }
        }
    }
}

/**
 * What was found, shown above the tabs on the result screen: each unknown that
 * was worked out with its value and unit, then each one that could not be
 * found with what else is needed.
 */
@Composable
fun PhysicsHeadline(data: PhysicsData) {
    Card(
        modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 4.dp, bottom = 4.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = 220.dp)
                .verticalScroll(rememberScrollState())
                .padding(12.dp),
        ) {
            if (data.found.isEmpty() && data.missing.isEmpty()) {
                Text(data.headline.text, style = MaterialTheme.typography.titleMedium)
            }
            if (data.found.isNotEmpty()) {
                Text("Found", style = MaterialTheme.typography.labelLarge)
                for (item in data.found) {
                    Row(Modifier.fillMaxWidth().padding(top = 2.dp)) {
                        Text(
                            item.label,
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.weight(1f).padding(end = 8.dp),
                        )
                        Text(item.text, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                    }
                }
            }
            if (data.missing.isNotEmpty()) {
                Text(
                    "Cannot be found yet",
                    style = MaterialTheme.typography.labelLarge,
                    modifier = Modifier.padding(top = if (data.found.isEmpty()) 0.dp else 8.dp),
                )
                for (item in data.missing) {
                    Text(
                        item.label + ": " + item.reason,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(top = 2.dp),
                    )
                }
            }
        }
    }
}

/**
 * Graph tab for a physics result: the solution over time with the ideal case
 * behind it, a picture of the motion, and a time slider that drives both.
 * The slider position is kept as a fraction of the time shown.
 */
@Composable
fun PhysicsTab(model: SolverViewModel) {
    val data = model.result?.physics ?: return
    if (data.curve.isEmpty()) {
        // The motion itself is not pinned down by the values entered, so there is nothing to draw.
        Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
            Text(
                "There is no graph yet: the values entered do not fix the whole motion. " +
                    "Enter the values listed under \"Cannot be found yet\" to see it.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        return
    }
    val fraction = model.pdeTime.coerceIn(0f, 1f)
    val time = fraction * data.tMax

    // One pass through the motion takes about six seconds.
    LaunchedEffect(model.pdePlaying, model.viewResetKey) {
        while (model.pdePlaying) {
            delay(33)
            val next = model.pdeTime + 1f / 180f
            if (next < 1f) {
                model.pdeTime = next
            } else {
                model.pdeTime = 1f
                model.pdePlaying = false
            }
        }
    }

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            PhysicsGraph(data, time, Modifier.fillMaxSize())
        }
        data.animation?.let { animation ->
            PhysicsPicture(animation, time, Modifier.fillMaxWidth().height(96.dp))
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
                        if (!model.pdePlaying && model.pdeTime >= 1f) model.pdeTime = 0f
                        model.pdePlaying = !model.pdePlaying
                    },
                ) { Text(if (model.pdePlaying) "Pause" else "Play") }
                Text("t = ${"%.3g".format(time)} s", modifier = Modifier.width(112.dp))
                Slider(
                    value = fraction,
                    onValueChange = {
                        model.pdePlaying = false
                        model.pdeTime = it
                    },
                    valueRange = 0f..1f,
                    modifier = Modifier.weight(1f),
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                PhysicsLegend(MaterialTheme.colorScheme.primary, data.curveLabel)
                if (data.ideal.isNotEmpty()) {
                    Spacer(Modifier.width(16.dp))
                    PhysicsLegend(MaterialTheme.colorScheme.outline, data.idealLabel)
                }
            }
            Text(
                data.yLabel + " against " + data.xLabel +
                    (data.marker?.let { ". The dot marks: " + it.label + "." } ?: "."),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun PhysicsLegend(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).background(color, CircleShape))
        Spacer(Modifier.width(6.dp))
        Text(label, style = MaterialTheme.typography.labelMedium)
    }
}

/** The solution curve on fixed axes, with the ideal curve, the answer marker and a time cursor. */
@Composable
private fun PhysicsGraph(data: PhysicsData, time: Double, modifier: Modifier = Modifier) {
    val colors = MaterialTheme.colorScheme
    val measurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 11.sp, color = colors.onSurfaceVariant)
    val curve = remember(data) { Samples.flatten(data.curve) }
    val ideal = remember(data) { Samples.flatten(data.ideal) }
    val range = remember(data) { data.valueRange() }
    val tMax = if (data.tMax > 0) data.tMax else 1.0
    val padT = tMax * 0.06
    val padV = (range.second - range.first) * 0.15
    val vp = Viewport(-padT, tMax + padT, range.first - padV, range.second + padV)

    Canvas(modifier = modifier.clipToBounds()) {
        val w = size.width
        val h = size.height
        if (w <= 0f || h <= 0f) return@Canvas
        val hair = 1.dp.toPx()

        val xCount = (w / 80.dp.toPx()).toInt().coerceAtLeast(2)
        val yCount = (h / 70.dp.toPx()).toInt().coerceAtLeast(2)
        val xStep = Plot.tickStep(vp.xMin, vp.xMax, xCount)
        val yStep = Plot.tickStep(vp.yMin, vp.yMax, yCount)
        for (x in Plot.ticks(vp.xMin, vp.xMax, xCount)) {
            val px = vp.toPxX(x, w)
            drawLine(colors.outlineVariant, Offset(px, 0f), Offset(px, h), hair)
            val layout = measurer.measure(Plot.tickLabel(x, xStep), labelStyle)
            val left = px - layout.size.width / 2f
            if (left >= 0f && left + layout.size.width <= w) {
                drawText(layout, topLeft = Offset(left, h - layout.size.height - 2.dp.toPx()))
            }
        }
        for (y in Plot.ticks(vp.yMin, vp.yMax, yCount)) {
            val py = vp.toPxY(y, h)
            drawLine(colors.outlineVariant, Offset(0f, py), Offset(w, py), hair)
            val layout = measurer.measure(Plot.tickLabel(y, yStep), labelStyle)
            val top = py - layout.size.height / 2f
            if (top >= 0f && top + layout.size.height <= h - 16.dp.toPx()) {
                drawText(layout, topLeft = Offset(3.dp.toPx(), top))
            }
        }
        // The axes: value zero and time zero.
        val zero = vp.toPxY(0.0, h)
        if (zero in 0f..h) drawLine(colors.onSurface, Offset(0f, zero), Offset(w, zero), 1.5.dp.toPx())
        val start = vp.toPxX(0.0, w)
        drawLine(colors.onSurface, Offset(start, 0f), Offset(start, h), 1.5.dp.toPx())

        if (ideal.size >= 4) drawPhysicsLines(Plot.path(ideal, vp, w, h), colors.outline, 2.dp.toPx())
        if (curve.size >= 4) drawPhysicsLines(Plot.path(curve, vp, w, h), colors.primary, 3.dp.toPx())

        // Where the slider is: a vertical line and a dot riding the curve.
        val cursorX = vp.toPxX(time, w)
        drawLine(colors.tertiary, Offset(cursorX, 0f), Offset(cursorX, h), hair)
        if (data.curve.isNotEmpty()) {
            val value = Samples.valueAt(data.curve, time)
            drawCircle(colors.tertiary, 5.dp.toPx(), Offset(cursorX, vp.toPxY(value, h)))
        }

        val marker = data.marker
        if (marker != null) {
            val centre = Offset(vp.toPxX(marker.t, w), vp.toPxY(marker.y, h))
            drawCircle(colors.surface, 7.dp.toPx(), centre)
            drawCircle(colors.error, 5.dp.toPx(), centre)
            if (marker.label.isNotEmpty()) {
                val layout = measurer.measure(marker.label, labelStyle)
                val left = (centre.x - layout.size.width - 8.dp.toPx()).coerceAtLeast(2.dp.toPx())
                val top = (centre.y - layout.size.height - 6.dp.toPx()).coerceAtLeast(2.dp.toPx())
                drawText(layout, topLeft = Offset(left, top))
            }
        }
    }
}

private fun DrawScope.drawPhysicsLines(lines: List<FloatArray>, color: Color, width: Float) {
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

/**
 * A simple picture of the situation at the slider's time: an object above the
 * ground, a mass on a spring, or a capacitor filling with charge.
 */
@Composable
private fun PhysicsPicture(animation: PhysicsAnimation, time: Double, modifier: Modifier = Modifier) {
    val colors = MaterialTheme.colorScheme
    val measurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 11.sp, color = colors.onSurfaceVariant)
    val level = animation.fractionAt(time).toFloat()
    val caption = animation.label + ": " + "%.4g".format(animation.valueAt(time))

    Canvas(modifier = modifier.clipToBounds()) {
        val w = size.width
        val h = size.height
        if (w <= 0f || h <= 0f) return@Canvas
        val pad = 10.dp.toPx()
        val stroke = 2.dp.toPx()
        val layout = measurer.measure(caption, labelStyle)
        drawText(layout, topLeft = Offset(pad, 2.dp.toPx()))

        when (animation.kind) {
            "fall" -> {
                // The ground at the bottom; the object's height above it.
                val ground = h - pad
                val top = pad + layout.size.height
                drawLine(colors.onSurface, Offset(pad, ground), Offset(w - pad, ground), stroke)
                val y = ground - level * (ground - top) - 6.dp.toPx()
                drawLine(colors.outlineVariant, Offset(w / 2f, top), Offset(w / 2f, ground), 1.dp.toPx())
                drawCircle(colors.primary, 6.dp.toPx(), Offset(w / 2f, y.coerceIn(top, ground - 6.dp.toPx())))
            }
            "spring" -> {
                // A wall on the left, a coiled spring, and the mass sliding left and right.
                val mid = h / 2f + layout.size.height / 2f
                val wall = pad * 2
                val block = 22.dp.toPx()
                val travel = w - wall - pad * 2 - block
                val massLeft = wall + travel * (0.25f + 0.6f * level)
                drawLine(colors.onSurface, Offset(wall, mid - block), Offset(wall, mid + block), stroke)
                val coils = 12
                val path = Path()
                path.moveTo(wall, mid)
                for (i in 1..coils) {
                    val x = wall + (massLeft - wall) * i / coils
                    val dy = if (i == coils) 0f else if (i % 2 == 1) -7.dp.toPx() else 7.dp.toPx()
                    path.lineTo(x, mid + dy)
                }
                drawPath(path, colors.outline, style = Stroke(stroke, cap = StrokeCap.Round, join = StrokeJoin.Round))
                drawRect(colors.primary, Offset(massLeft, mid - block / 2), Size(block, block))
            }
            else -> {
                // A capacitor as a tank: the filled part is its voltage relative to the largest shown.
                val left = w / 2f - 40.dp.toPx()
                val top = pad + layout.size.height
                val boxW = 80.dp.toPx()
                val boxH = h - top - pad
                if (boxH > 0f) {
                    drawRect(colors.primaryContainer, Offset(left, top), Size(boxW, boxH))
                    val filled = boxH * level
                    drawRect(colors.primary, Offset(left, top + boxH - filled), Size(boxW, filled))
                    drawRect(colors.onSurface, Offset(left, top), Size(boxW, boxH), style = Stroke(stroke))
                }
            }
        }
    }
}
