package com.androidmath.app

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import com.androidmath.app.ui.AppTheme
import com.androidmath.app.ui.HomeScreen
import com.androidmath.app.ui.InputScreen
import com.androidmath.app.ui.ResultScreen

class MainActivity : ComponentActivity() {
    private val model: SolverViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (savedInstanceState == null) openFromIntent(intent)
        setContent {
            AppTheme {
                BackHandler(enabled = model.screen != Screen.Home) { model.back() }
                when (model.screen) {
                    Screen.Home -> HomeScreen(model)
                    Screen.Input -> InputScreen(model)
                    Screen.Result -> ResultScreen(model)
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        openFromIntent(intent)
    }

    /**
     * Lets another app or a test open a problem directly. Extras: "equation"
     * (required), "mode" (first, second, numeric, laplace, inverse, system,
     * fourier, boundary, eigen, heat, wave, laplace), "tab" (0 steps, 1 graph), and the input
     * boxes by name: equation2, equation3, x0, y0, yp0, z0, h, n, and for the
     * boundary section a, va, ka, b, vb, kb ("dy" for a slope condition), bc.
     * For the PDE modes: equation is f, equation2 is g (wave), x0 and y0 are
     * L and k, L and c, or a and b; bc is the condition code or the side; t1, t2.
     */
    private fun openFromIntent(intent: Intent?) {
        val source = intent ?: return
        if (source.getStringExtra("equation") == null) return
        val fields = HashMap<String, String>()
        for (name in FIELD_NAMES) {
            source.getStringExtra(name)?.let { fields[name] = it }
        }
        model.restore(source.getStringExtra("mode") ?: "first", fields, source.getIntExtra("tab", 0))
    }

    private companion object {
        val FIELD_NAMES = listOf(
            "equation", "equation2", "equation3", "x0", "y0", "yp0", "z0", "h", "n",
            "a", "va", "ka", "b", "vb", "kb", "bc", "t1", "t2",
        )
    }
}
