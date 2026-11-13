package com.androidmath.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.androidmath.core.Expr
import com.androidmath.core.InitialCondition
import com.androidmath.core.Ode
import com.androidmath.core.Plot
import com.androidmath.core.Viewport

/**
 * Slope field for y' = f(x, y) with solution curves on top.
 *
 * Drag to pan, pinch to zoom, tap to draw the numerical solution through
 * the tapped point. [exact] is the closed-form solution (evaluated with
 * constant [constant]); when it is null and an initial condition exists,
 * the curve through it is computed numerically instead.
 */
@Composable
fun GraphView(
    rhs: Expr?,
    exact: Expr?,
    constant: Double,
    ic: InitialCondition?,
    tapped: List<Pair<Double, Double>>,
    resetKey: Int,
    onTap: (Double, Double) -> Unit,
    modifier: Modifier = Modifier,
) {
    var size by remember { mutableStateOf(IntSize.Zero) }
    var viewport by remember(resetKey) { mutableStateOf<Viewport?>(null) }
    val currentOnTap by rememberUpdatedState(onTap)

    val w = size.width.toFloat()
    val h = size.height.toFloat()
    val vp = viewport ?: if (w > 0 && h > 0) {
        Viewport.centred(ic?.x0 ?: 0.0, ic?.y0 ?: 0.0, 5.0, w, h).also { viewport = it }
    } else null

    val colors = MaterialTheme.colorScheme
    val density = LocalDensity.current
    val measurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 11.sp, color = colors.onSurfaceVariant)
    val spacingPx = with(density) { 26.dp.toPx() }
    val lengthPx = with(density) { 13.dp.toPx() }

    val field = remember(rhs, vp, size) {
        if (rhs != null && vp != null) Plot.slopeField(rhs, vp, w, h, spacingPx, lengthPx) else FloatArray(0)
    }
    val exactLines = remember(exact, constant, vp, size) {
        if (exact != null && vp != null) {
            Plot.curve({ x -> exact.eval(x, 0.0, constant) }, vp, w, h, (w / 2).toInt().coerceIn(200, 1200))
        } else emptyList()
    }
    val starts = remember(tapped.toList(), ic, exact) {
        val list = tapped.toMutableList()
        if (ic != null && exact == null) list.add(0, ic.x0 to ic.y0)
        list
    }
    val numericLines = remember(rhs, starts, vp, size) {
        if (rhs == null || vp == null) emptyList()
        else starts.flatMap { (x0, y0) ->
            // Integrate over a range that reaches the start point even if it is off-screen.
            val curve = Ode.solve(
                rhs, x0, y0,
                minOf(vp.xMin, x0), maxOf(vp.xMax, x0),
                minOf(vp.yMin, y0), maxOf(vp.yMax, y0),
            )
            Plot.path(curve, vp, w, h)
        }
    }

    Canvas(
        modifier = modifier
            .clipToBounds()
            .onSizeChanged { size = it }
            .pointerInput(resetKey) {
                detectTransformGestures { centroid, pan, zoom, _ ->
                    val current = viewport ?: return@detectTransformGestures
                    val width = this.size.width.toFloat()
                    val height = this.size.height.toFloat()
                    viewport = current
                        .zoomedBy(zoom, centroid.x, centroid.y, width, height)
                        .pannedBy(pan.x, pan.y, width, height)
                }
            }
            .pointerInput(resetKey) {
                detectTapGestures { point ->
                    val current = viewport ?: return@detectTapGestures
                    currentOnTap(
                        current.toWorldX(point.x, this.size.width.toFloat()),
                        current.toWorldY(point.y, this.size.height.toFloat()),
                    )
                }
            },
    ) {
        if (vp == null) return@Canvas
        drawGrid(vp, measurer, labelStyle, colors.outlineVariant, colors.onSurface)

        val fieldColor = colors.onSurfaceVariant.copy(alpha = 0.55f)
        val fieldStroke = 1.25.dp.toPx()
        var i = 0
        while (i + 3 < field.size) {
            drawLine(fieldColor, Offset(field[i], field[i + 1]), Offset(field[i + 2], field[i + 3]), fieldStroke, StrokeCap.Round)
            i += 4
        }

        drawPolylines(numericLines, colors.tertiary, 2.dp.toPx())
        drawPolylines(exactLines, colors.primary, 3.dp.toPx())

        for ((x0, y0) in tapped) {
            drawCircle(colors.tertiary, 4.dp.toPx(), Offset(vp.toPxX(x0, w), vp.toPxY(y0, h)))
        }
        if (ic != null) {
            val centre = Offset(vp.toPxX(ic.x0, w), vp.toPxY(ic.y0, h))
            drawCircle(colors.surface, 7.dp.toPx(), centre)
            drawCircle(colors.primary, 5.dp.toPx(), centre)
        }
    }
}

private fun DrawScope.drawPolylines(lines: List<FloatArray>, color: Color, width: Float) {
    if (lines.isEmpty()) return
    val path = Path()
    for (line in lines) {
        path.moveTo(line[0], line[1])
        var i = 2
        while (i + 1 < line.size) {
            path.lineTo(line[i], line[i + 1])
            i += 2
        }
    }
    drawPath(path, color, style = Stroke(width, cap = StrokeCap.Round, join = StrokeJoin.Round))
}

private fun DrawScope.drawGrid(
    vp: Viewport, measurer: TextMeasurer, style: TextStyle, gridColor: Color, axisColor: Color,
) {
    val w = size.width
    val h = size.height
    val xStep = Plot.tickStep(vp.xMin, vp.xMax, (w / 80.dp.toPx()).toInt().coerceAtLeast(2))
    val yStep = Plot.tickStep(vp.yMin, vp.yMax, (h / 80.dp.toPx()).toInt().coerceAtLeast(2))
    val xTicks = Plot.ticks(vp.xMin, vp.xMax, (w / 80.dp.toPx()).toInt().coerceAtLeast(2))
    val yTicks = Plot.ticks(vp.yMin, vp.yMax, (h / 80.dp.toPx()).toInt().coerceAtLeast(2))
    val hair = 1.dp.toPx()

    for (x in xTicks) {
        val px = vp.toPxX(x, w)
        drawLine(gridColor, Offset(px, 0f), Offset(px, h), hair)
    }
    for (y in yTicks) {
        val py = vp.toPxY(y, h)
        drawLine(gridColor, Offset(0f, py), Offset(w, py), hair)
    }

    // Axes, pinned to the edge when the origin is off-screen so labels stay visible.
    val axisX = vp.toPxX(0.0, w)
    val axisY = vp.toPxY(0.0, h)
    if (axisX in 0f..w) drawLine(axisColor, Offset(axisX, 0f), Offset(axisX, h), 1.5.dp.toPx())
    if (axisY in 0f..h) drawLine(axisColor, Offset(0f, axisY), Offset(w, axisY), 1.5.dp.toPx())

    val pad = 3.dp.toPx()
    for (x in xTicks) {
        if (x == 0.0) continue
        val layout = measurer.measure(Plot.tickLabel(x, xStep), style)
        val top = (axisY + pad).coerceIn(pad, h - layout.size.height - pad)
        val left = vp.toPxX(x, w) - layout.size.width / 2f
        if (left < 0f || left + layout.size.width > w) continue
        drawText(layout, topLeft = Offset(left, top))
    }
    for (y in yTicks) {
        if (y == 0.0) continue
        val layout = measurer.measure(Plot.tickLabel(y, yStep), style)
        val left = (axisX + pad).coerceIn(pad, w - layout.size.width - pad)
        val top = vp.toPxY(y, h) - layout.size.height / 2f
        if (top < 0f || top + layout.size.height > h) continue
        drawText(layout, topLeft = Offset(left, top))
    }
}
