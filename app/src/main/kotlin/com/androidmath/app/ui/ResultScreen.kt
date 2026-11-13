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
                title = { Text(if (numeric) "Approximation" else "Solution") },
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
                        if (numeric) "Compare ${methods.size} methods" else "${methods.size} ways to solve this",
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

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (rhs == null && exact == null && points == null) {
                Text(
                    if (result.order == 2 || method != null) {
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
            if (showC1) {
                ConstantSlider(if (showC2) "C₁" else "C", model.constant) { model.constant = it }
            }
            if (showC2) {
                ConstantSlider("C₂", model.constant2) { model.constant2 = it }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (exact != null) {
                    Legend(MaterialTheme.colorScheme.primary, "Exact solution")
                    Spacer(Modifier.width(16.dp))
                }
                if (points != null) {
                    Legend(MaterialTheme.colorScheme.tertiary, method?.name ?: "Numerical")
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
