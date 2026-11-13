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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.background
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

@Composable
fun ResultScreen(model: SolverViewModel) {
    val result = model.result ?: return
    val methods = result.applicableMethods
    val method = model.selectedMethod

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Solution") },
                navigationIcon = { TextButton(onClick = { model.back() }) { Text("Edit") } },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            if (methods.size > 1) {
                Text(
                    "${methods.size} ways to solve this",
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
    val rhs = remember(result.rhsExpr) { Expr.parseOrNull(result.rhsExpr) }
    val particular = remember(method) { Expr.parseOrNull(method?.plot?.particularExpr) }
    val general = remember(method) { Expr.parseOrNull(method?.plot?.generalExpr) }
    // Prefer the solution of the initial value problem; otherwise show one
    // member of the family, chosen with the slider.
    val exact = particular ?: general
    val showSlider = particular == null && general != null &&
        (method?.plot?.generalExpr?.contains("C") == true)

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (rhs == null) {
                Text(
                    "This equation uses a function the grapher does not support yet.",
                    modifier = Modifier.align(Alignment.Center).padding(24.dp),
                )
            } else {
                GraphView(
                    rhs = rhs,
                    exact = exact,
                    constant = model.constant.toDouble(),
                    ic = result.ic,
                    tapped = model.tappedPoints,
                    resetKey = model.viewResetKey,
                    onTap = { x, y -> model.tappedPoints.add(x to y) },
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
        Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
            if (showSlider) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("C = ${"%.2f".format(model.constant)}", modifier = Modifier.width(88.dp))
                    Slider(
                        value = model.constant,
                        onValueChange = { model.constant = it },
                        valueRange = -5f..5f,
                        modifier = Modifier.weight(1f),
                    )
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (exact != null) {
                    Legend(MaterialTheme.colorScheme.primary, "Exact solution")
                    Spacer(Modifier.width(16.dp))
                }
                Legend(MaterialTheme.colorScheme.tertiary, "Numerical (RK4)")
                Spacer(Modifier.weight(1f))
                if (model.tappedPoints.isNotEmpty()) {
                    TextButton(onClick = { model.tappedPoints.clear() }) { Text("Clear") }
                }
                TextButton(onClick = { model.viewResetKey++ }) { Text("Reset view") }
            }
            Text(
                "Tap the graph to draw the solution through that point. Drag to move, pinch to zoom.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
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
