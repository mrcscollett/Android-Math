@file:OptIn(ExperimentalMaterial3Api::class)

package com.androidmath.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.unit.dp
import com.androidmath.app.Mode
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.text.style.TextOverflow
import com.androidmath.app.SolverViewModel
import com.androidmath.core.History
import com.androidmath.core.HistoryEntry

private data class Section(val title: String, val detail: String, val mode: Mode?)

private val SECTIONS = listOf(
    Section(
        "First-order equations",
        "Separable, linear, exact, Bernoulli and homogeneous equations, with slope fields",
        Mode.FirstOrder,
    ),
    Section(
        "Second-order equations",
        "Constant coefficients, undetermined coefficients, variation of parameters, " +
            "Cauchy\u2013Euler, series solutions, and Laplace transforms with step and impulse forcing",
        Mode.SecondOrder,
    ),
    Section(
        "Numerical methods",
        "Euler, improved Euler and Runge\u2013Kutta with step tables and error",
        Mode.Numerical,
    ),
    Section(
        "Laplace transforms",
        "Transforms and inverse transforms: partial fractions, shifting theorems, " +
            "steps and impulses, convolution",
        Mode.Laplace,
    ),
    Section(
        "Systems",
        "Eigenvalue method and matrix exponential for linear systems, linearization " +
            "of nonlinear ones, and phase portraits",
        Mode.Systems,
    ),
    Section(
        "Fourier series",
        "Full, sine and cosine series with the coefficient integrals and converging partial sums",
        Mode.Fourier,
    ),
    Section(
        "Boundary value problems",
        "Two-point boundary conditions, and the eigenvalue problem behind separation of variables",
        Mode.Boundary,
    ),
    Section("Partial differential equations", "Heat, wave and Laplace equations", null),
    Section("Physics 1", "Motion, drag, springs, pendulums, cooling", null),
    Section("Physics 2", "RC, RL and RLC circuits, charged particles, decay", null),
)

@Composable
fun HomeScreen(model: SolverViewModel) {
    var confirmClear by remember { mutableStateOf(false) }
    val favorites = History.favorites(model.history)
    val recent = History.recent(model.history)

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear recent problems?") },
            text = { Text("This removes every recent problem. Favorites are kept.") },
            confirmButton = {
                TextButton(onClick = {
                    model.clearRecent()
                    confirmClear = false
                }) { Text("Clear") }
            },
            dismissButton = {
                TextButton(onClick = { confirmClear = false }) { Text("Cancel") }
            },
        )
    }

    Scaffold(topBar = { TopAppBar(title = { Text("DiffEq Solver") }) }) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            horizontalAlignment = Alignment.CenterHorizontally,
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 24.dp),
        ) {
            items(SECTIONS) { section ->
                Card(
                    modifier = Modifier
                        .widthIn(max = 640.dp)
                        .fillMaxWidth()
                        .padding(vertical = 6.dp)
                        .alpha(if (section.mode != null) 1f else 0.55f)
                        .then(
                            if (section.mode != null) Modifier.clickable { model.open(section.mode) }
                            else Modifier
                        ),
                    colors = if (section.mode != null) {
                        CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                    } else {
                        CardDefaults.cardColors()
                    },
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Text(section.title, style = MaterialTheme.typography.titleMedium)
                        Text(section.detail, style = MaterialTheme.typography.bodyMedium)
                        if (section.mode == null) {
                            Text(
                                "Coming soon",
                                style = MaterialTheme.typography.labelMedium,
                                modifier = Modifier.padding(top = 4.dp),
                            )
                        }
                    }
                }
            }
            if (favorites.isNotEmpty()) {
                item {
                    HistoryHeader("Favorites", null)
                }
                items(favorites, key = { "f" + it.key }) { entry ->
                    HistoryRow(entry, model)
                }
            }
            if (recent.isNotEmpty()) {
                item {
                    HistoryHeader("Recent") { confirmClear = true }
                }
                items(recent, key = { "r" + it.key }) { entry ->
                    HistoryRow(entry, model)
                }
            }
        }
    }
}

@Composable
private fun HistoryHeader(title: String, onClear: (() -> Unit)?) {
    Row(
        modifier = Modifier.widthIn(max = 640.dp).fillMaxWidth().padding(top = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(title, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
        if (onClear != null) {
            TextButton(onClick = onClear) { Text("Clear recent") }
        }
    }
}

/** One saved problem: tap to reopen and solve it, star to keep it. */
@Composable
private fun HistoryRow(entry: HistoryEntry, model: SolverViewModel) {
    Row(
        modifier = Modifier
            .widthIn(max = 640.dp)
            .fillMaxWidth()
            .clickable { model.restore(entry.mode, entry.fields) }
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(
                entry.section,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                entry.label,
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        TextButton(onClick = { model.toggleFavorite(entry) }) {
            Text(
                if (entry.favorite) "\u2605" else "\u2606",
                style = MaterialTheme.typography.titleLarge,
            )
        }
    }
}
