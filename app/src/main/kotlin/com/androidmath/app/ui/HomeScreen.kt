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
            "Cauchy\u2013Euler, and Laplace transforms for initial value problems",
        Mode.SecondOrder,
    ),
    Section(
        "Numerical methods",
        "Euler, improved Euler and Runge\u2013Kutta with step tables and error",
        Mode.Numerical,
    ),
    Section("Laplace transform tools", "Transform tables, step and impulse forcing, convolution", null),
    Section("Systems", "Eigenvalue method, phase portraits, linearization", null),
    Section("Series and PDEs", "Power series, Fourier series, heat and wave equations", null),
    Section("Physics 1", "Motion, drag, springs, pendulums, cooling", null),
    Section("Physics 2", "RC, RL and RLC circuits, charged particles, decay", null),
)

@Composable
fun HomeScreen(onOpen: (Mode) -> Unit) {
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
                            if (section.mode != null) Modifier.clickable { onOpen(section.mode) }
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
        }
    }
}
