@file:OptIn(ExperimentalMaterial3Api::class)

package com.androidmath.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.unit.dp

private data class Section(val title: String, val detail: String, val ready: Boolean)

private val SECTIONS = listOf(
    Section("First-order equations", "Separable and linear equations, initial value problems, slope fields", true),
    Section("More first-order types", "Exact, Bernoulli, homogeneous, autonomous equations", false),
    Section("Second order and higher", "Constant coefficients, undetermined coefficients, variation of parameters", false),
    Section("Laplace transforms", "Transforms, inverse transforms, step and impulse forcing", false),
    Section("Systems", "Eigenvalue method, phase portraits, linearization", false),
    Section("Numerical methods", "Euler, improved Euler, Runge-Kutta, error tables", false),
    Section("Series and PDEs", "Power series, Fourier series, heat and wave equations", false),
    Section("Physics 1", "Motion, drag, springs, pendulums, cooling", false),
    Section("Physics 2", "RC, RL and RLC circuits, charged particles, decay", false),
)

@Composable
fun HomeScreen(onOpenFirstOrder: () -> Unit) {
    Scaffold(topBar = { TopAppBar(title = { Text("DiffEq Solver") }) }) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 24.dp),
        ) {
            items(SECTIONS) { section ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 6.dp)
                        .alpha(if (section.ready) 1f else 0.55f)
                        .then(if (section.ready) Modifier.clickable(onClick = onOpenFirstOrder) else Modifier),
                    colors = if (section.ready) {
                        CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                    } else {
                        CardDefaults.cardColors()
                    },
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Text(section.title, style = MaterialTheme.typography.titleMedium)
                        Text(section.detail, style = MaterialTheme.typography.bodyMedium)
                        if (!section.ready) {
                            Text(
                                "Coming soon",
                                style = MaterialTheme.typography.labelMedium,
                                modifier = Modifier.padding(top = 4.dp),
                            )
                        }
                    }
                }
            }
        }
    }
}
