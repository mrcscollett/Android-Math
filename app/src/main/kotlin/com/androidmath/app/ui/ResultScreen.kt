@file:OptIn(ExperimentalMaterial3Api::class)

package com.androidmath.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.androidmath.app.SolverViewModel
import com.androidmath.core.Expr

/** Matches the width cap of the steps page so everything lines up on tablets. */
private val CONTENT_WIDTH = 760.dp

@Composable
fun ResultScreen(model: SolverViewModel) {
    val result = model.result ?: return
    val methods = result.applicableMethods
    val method = model.selectedMethod
    val numeric = result.kind == "numeric"

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        if (result.title.isNotBlank()) result.title else when (result.kind) {
                            "numeric" -> "Approximation"
                            "transform" -> "Transform"
                            "fourier" -> "Fourier series"
                            "boundary" -> "Boundary value problem"
                            "eigen" -> "Eigenvalue problem"
                            "pde" -> when (result.pde) {
                                "wave" -> "Wave equation"
                                "laplace" -> "Laplace's equation"
                                "characteristics" -> "Characteristics"
                                "classify" -> "Classification"
                                "fd" -> "Finite differences"
                                "disk" -> "Laplace's equation"
                                else -> "Heat equation"
                            }
                            else -> "Solution"
                        }
                    )
                },
                navigationIcon = { TextButton(onClick = { model.back() }) { Text("Edit") } },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Column(Modifier.widthIn(max = CONTENT_WIDTH).fillMaxWidth()) {
                if (methods.size > 1) {
                    Text(
                        when (result.kind) {
                            "numeric" -> "Compare ${methods.size} methods"
                            "transform" -> "${methods.size} ways to find this"
                            "fourier" -> "${methods.size} series for this function"
                            "tool" -> "${methods.size} ways to do this"
                            else -> "${methods.size} ways to solve this"
                        },
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(start = 16.dp, top = 4.dp),
                    )
                }
                if (methods.isNotEmpty()) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState())
                            .padding(horizontal = 16.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        for (m in methods) {
                            FilterChip(
                                selected = m.id == method?.id,
                                onClick = { model.methodId = m.id },
                                label = { Text(m.name) },
                            )
                        }
                    }
                }
            }
            PrimaryTabRow(selectedTabIndex = model.tab) {
                Tab(selected = model.tab == 0, onClick = { model.tab = 0 }, text = { Text("Steps") })
                Tab(selected = model.tab == 1, onClick = { model.tab = 1 }, text = { Text("Graph") })
            }
            if (model.tab == 0) {
                MathWebView(
                    script = "showSolution(${jsString(model.resultJson)}, ${jsString(method?.id ?: "")});",
                    modifier = Modifier.fillMaxSize(),
                )
            } else if (result.kind == "system" || result.graph == "phase") {
                PhaseTab(model)
            } else if (result.kind == "pde" || result.graph == "surface") {
                PdeTab(model)
            } else {
                GraphTab(model)
            }
        }
    }
}

@Composable
private fun GraphTab(model: SolverViewModel) {
    val result = model.result ?: return
    val method = model.selectedMethod
    val numeric = result.kind == "numeric"
    val rhs = remember(result.rhsExpr) { Expr.parseOrNull(result.rhsExpr) }
    val particular = remember(method) { Expr.parseOrNull(method?.plot?.particularExpr) }
    val general = remember(method) { Expr.parseOrNull(method?.plot?.generalExpr) }
    val points = remember(method) {
        method?.plot?.pointArray()?.takeIf { it.size >= 4 }
    }
    // Prefer the solution of the initial value problem; otherwise show one
    // member of the family, chosen with the sliders.
    val exact = particular ?: general
    val generalText = method?.plot?.generalExpr.orEmpty()
    val family = particular == null && general != null
    val showC1 = family && Regex("""\bC\b""").containsMatchIn(generalText)
    val showC2 = family && generalText.contains("C2")
    val canTap = rhs != null
    // Fourier partial sums or eigenfunctions to step through with a slider.
    val partials = method?.plot?.partials.orEmpty()
    val partialIndex = model.partialIndex.coerceIn(0, (partials.size - 1).coerceAtLeast(0))
    val overlay = remember(method, partialIndex) {
        Expr.parseOrNull(partials.getOrNull(partialIndex)?.expr)
    }
    val wide = result.kind == "fourier" || result.kind == "eigen" || result.kind == "boundary" ||
        (result.kind == "tool" && result.span != null)
    val halfWidth = if (wide) ((result.span ?: 2.5) * 2).coerceIn(1.0, 100.0) else 5.0

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (rhs == null && exact == null && points == null && overlay == null) {
                Text(
                    if (result.kind == "tool") {
                        "There is nothing to graph for this result; see the Steps tab."
                    } else if (result.kind == "boundary") {
                        "This boundary value problem has no solution, so there is nothing to graph."
                    } else if (result.kind == "transform") {
                        "This function cannot be graphed (it contains an impulse or a " +
                            "function the grapher does not support)."
                    } else if (result.order == 2 || method != null) {
                        "There is no formula to graph for this method. Its solution is " +
                            "left in implicit or integral form."
                    } else {
                        "This equation uses a function the grapher does not support yet."
                    },
                    modifier = Modifier.align(Alignment.Center).padding(24.dp),
                )
            } else {
                GraphView(
                    rhs = rhs,
                    exact = exact,
                    constant = model.constant.toDouble(),
                    constant2 = model.constant2.toDouble(),
                    ic = result.ic,
                    points = points,
                    tapped = model.tappedPoints,
                    resetKey = model.viewResetKey,
                    onTap = { x, y -> if (canTap) model.tappedPoints.add(x to y) },
                    modifier = Modifier.fillMaxSize(),
                    overlay = overlay,
                    halfWidth = halfWidth,
                )
            }
        }
        Column(
            modifier = Modifier
                .align(Alignment.CenterHorizontally)
                .widthIn(max = CONTENT_WIDTH)
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            if (partials.size > 1) {
                val current = partials[partialIndex].n
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        // The engine may name each curve itself (a time, a polynomial).
                        partials[partialIndex].label
                            ?: if (result.kind == "eigen") "n = $current" else "N = $current",
                        modifier = Modifier.width(96.dp),
                    )
                    Slider(
                        value = partialIndex.toFloat(),
                        onValueChange = { model.partialIndex = (it + 0.5f).toInt() },
                        valueRange = 0f..(partials.size - 1).toFloat(),
                        steps = (partials.size - 2).coerceAtLeast(0),
                        modifier = Modifier.weight(1f),
                    )
                }
            }
            if (showC1) {
                ConstantSlider(if (showC2) "C₁" else "C", model.constant) { model.constant = it }
            }
            if (showC2) {
                ConstantSlider("C₂", model.constant2) { model.constant2 = it }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (exact != null) {
                    Legend(
                        MaterialTheme.colorScheme.primary,
                        result.legends["exact"] ?: when (result.kind) {
                            "transform" -> "Function of t"
                            "fourier" -> "f(x) (periodic)"
                            "boundary" -> "Solution"
                            "tool" -> when (result.title) {
                                "Derivative" -> "Derivative"
                                "Taylor series" -> "Function"
                                else -> "Integrand"
                            }
                            else -> "Exact solution"
                        },
                    )
                    Spacer(Modifier.width(16.dp))
                }
                if (overlay != null && partials.isNotEmpty()) {
                    Legend(
                        MaterialTheme.colorScheme.tertiary,
                        partials[partialIndex].label
                            ?: if (result.kind == "eigen") "Eigenfunction ${partials[partialIndex].n}"
                            else if (result.kind == "tool") "Taylor polynomial, degree ${partials[partialIndex].n}"
                            else "Partial sum, ${partials[partialIndex].n} terms",
                    )
                }
                if (points != null) {
                    Legend(
                        MaterialTheme.colorScheme.tertiary,
                        result.legends["points"]
                            ?: if (result.kind == "tool") "In the complex plane" else method?.name ?: "Numerical",
                    )
                } else if (canTap) {
                    Legend(MaterialTheme.colorScheme.tertiary, "Numerical (RK4)")
                }
                Spacer(Modifier.weight(1f))
                if (model.tappedPoints.isNotEmpty()) {
                    TextButton(onClick = { model.tappedPoints.clear() }) { Text("Clear") }
                }
                TextButton(onClick = { model.viewResetKey++ }) { Text("Reset view") }
            }
            Text(
                if (canTap) {
                    "Tap the graph to draw the solution through that point. Drag to move, pinch to zoom."
                } else {
                    "Drag to move, pinch to zoom."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun PhaseTab(model: SolverViewModel) {
    val result = model.result ?: return
    val method = model.selectedMethod
    val f = remember(result.field) { Expr.parseOrNull(result.field?.f) }
    val g = remember(result.field) { Expr.parseOrNull(result.field?.g) }
    val exactX = remember(method) { Expr.parseOrNull(method?.plot?.xExpr) }
    val exactY = remember(method) { Expr.parseOrNull(method?.plot?.yExpr) }

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            val autonomous = result.field != null
            val hasCurve = exactX != null && exactY != null
            // A three-equation system has no plane to draw in.
            val three = result.input3x3()
            if (autonomous && (f == null || g == null)) {
                Text(
                    "This system uses a function the grapher does not support yet.",
                    modifier = Modifier.align(Alignment.Center).padding(24.dp),
                )
            } else if (three || (!autonomous && !hasCurve)) {
                Text(
                    if (three) {
                        "A 3\u00d73 system has no phase plane; see the Steps tab."
                    } else {
                        "The forcing depends on t, so there is no fixed direction field. " +
                            "Add initial values to see the solution curve."
                    },
                    modifier = Modifier.align(Alignment.Center).padding(24.dp),
                )
            } else {
                PhaseView(
                    f = f, g = g,
                    exactX = exactX, exactY = exactY,
                    ic = result.ic,
                    equilibria = result.equilibria,
                    tapped = model.tappedPoints,
                    resetKey = model.viewResetKey,
                    onTap = { x, y -> if (autonomous) model.tappedPoints.add(x to y) },
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
        Column(
            modifier = Modifier
                .align(Alignment.CenterHorizontally)
                .widthIn(max = CONTENT_WIDTH)
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (exactX != null && exactY != null) {
                    Legend(MaterialTheme.colorScheme.primary, result.legends["exact"] ?: "Exact solution")
                    Spacer(Modifier.width(16.dp))
                }
                if (result.field != null) {
                    Legend(MaterialTheme.colorScheme.tertiary, result.legends["trajectories"] ?: "Trajectories")
                }
                if (result.equilibria.isNotEmpty()) {
                    Spacer(Modifier.width(16.dp))
                    Legend(MaterialTheme.colorScheme.error, result.legends["equilibria"] ?: "Equilibria")
                }
                Spacer(Modifier.weight(1f))
                if (model.tappedPoints.isNotEmpty()) {
                    TextButton(onClick = { model.tappedPoints.clear() }) { Text("Clear") }
                }
                TextButton(onClick = { model.viewResetKey++ }) { Text("Reset view") }
            }
            Text(
                // A math tool may describe its own picture.
                result.legends["hint"] ?: if (result.field == null) {
                    "The curve is the solution (x(t), y(t)) for the given initial values. " +
                        "Drag to move, pinch to zoom."
                } else {
                    "Tap the phase plane to draw the trajectory through that point. The axes " +
                        "are x and y; time runs along each curve in the direction of the dots."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ConstantSlider(label: String, value: Float, onChange: (Float) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("$label = ${"%.2f".format(value)}", modifier = Modifier.width(96.dp))
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = -5f..5f,
            modifier = Modifier.weight(1f),
        )
    }
}

@Composable
private fun Legend(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).background(color, CircleShape))
        Spacer(Modifier.width(6.dp))
        Text(label, style = MaterialTheme.typography.labelMedium)
    }
}

/** True for a three-equation system (its solution names z(t)). */
private fun com.androidmath.core.SolveResult.input3x3(): Boolean = inputLatex.contains("z' =")
