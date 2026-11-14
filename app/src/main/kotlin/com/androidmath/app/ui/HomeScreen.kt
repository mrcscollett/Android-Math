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

private val EQUATION_SECTIONS = listOf(
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
    Section(
        "Partial differential equations",
        "Heat, wave and Laplace equations with animated solutions, characteristics, " +
            "classification and finite differences",
        Mode.Pde,
    ),
)

private val TOOL_SECTIONS = listOf(
    Section(
        "Linear algebra",
        "Row reduction, systems, determinants, inverses, eigenvalues, null space, the Wronskian",
        Mode.LinearAlgebra,
    ),
    Section(
        "Calculus",
        "Integrals with the technique shown, improper integrals, partial fractions, derivatives",
        Mode.Calculus,
    ),
    Section(
        "Series",
        "Taylor series, convergence tests, radius and interval of convergence",
        Mode.Series,
    ),
    Section(
        "Complex numbers",
        "Arithmetic, polar and exponential form, De Moivre's theorem, roots",
        Mode.Complex,
    ),
    Section(
        "Special functions",
        "Bessel functions, Legendre and other orthogonal polynomials, Sturm\u2013Liouville form",
        Mode.Special,
    ),
    Section(
        "Fourier transforms",
        "Transforms and inverses, sine and cosine transforms, heat flow on the whole line",
        Mode.FourierTransform,
    ),
    Section(
        "Difference equations",
        "Linear recurrences such as Fibonacci: closed formulas and first terms",
        Mode.Recurrence,
    ),
    Section(
        "Stability and dynamics",
        "Lyapunov functions, conservative systems, limit cycles, bifurcations",
        Mode.Dynamics,
    ),
    Section(
        "Vector calculus",
        "Gradient, divergence, curl, line and double integrals, Green's, Stokes' and the divergence theorem",
        Mode.Vector,
    ),
)

private val GROUPS = listOf(
    "Differential equations" to EQUATION_SECTIONS,
    "Math tools" to TOOL_SECTIONS,
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
            for ((heading, sections) in GROUPS) {
                item(key = "group-$heading") {
                    Text(
                        heading,
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.widthIn(max = 640.dp).fillMaxWidth().padding(top = 12.dp, bottom = 2.dp),
                    )
                }
                items(sections, key = { "section-" + it.title }) { section ->
                    SectionCard(section, model)
                }
            }
            // Physics scenarios come from the engine's catalog, so new ones appear by themselves.
            item(key = "group-physics") {
                Text(
                    "Physics",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.widthIn(max = 640.dp).fillMaxWidth().padding(top = 12.dp, bottom = 2.dp),
                )
            }
            val catalog = model.physicsCatalog
            if (catalog == null) {
                item(key = "physics-loading") {
                    PhysicsCard("Physics scenarios", "Starting the solver\u2026", false) {}
                }
            } else {
                items(catalog.scenarios, key = { "scenario-" + it.id }) { scenario ->
                    PhysicsCard(scenario.title, scenario.description, true) { model.openScenario(scenario.id) }
                }
                items(catalog.upcoming, key = { "upcoming-$it" }) { name ->
                    PhysicsCard(name, "Coming soon", false) {}
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
private fun SectionCard(section: Section, model: SolverViewModel) {
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

/** A physics scenario on the home screen; [ready] ones open their form. */
@Composable
private fun PhysicsCard(title: String, detail: String, ready: Boolean, onOpen: () -> Unit) {
    Card(
        modifier = Modifier
            .widthIn(max = 640.dp)
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .alpha(if (ready) 1f else 0.55f)
            .then(if (ready) Modifier.clickable(onClick = onOpen) else Modifier),
        colors = if (ready) {
            CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        } else {
            CardDefaults.cardColors()
        },
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(detail, style = MaterialTheme.typography.bodyMedium)
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
