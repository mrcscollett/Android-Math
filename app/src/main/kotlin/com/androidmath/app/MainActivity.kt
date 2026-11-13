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
                    Screen.Home -> HomeScreen(onOpen = { model.open(it) })
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
     * Lets another app or a test open a problem directly. Extras: "equation",
     * optional "mode" (first, second, numeric, laplace, inverse), "x0", "y0", "yp0", "h", "n"
     * and "tab" (0 steps, 1 graph).
     */
    private fun openFromIntent(intent: Intent?) {
        val equation = intent?.getStringExtra("equation") ?: return
        model.open(
            when (intent.getStringExtra("mode")) {
                "second" -> Mode.SecondOrder
                "numeric" -> Mode.Numerical
                "laplace", "inverse" -> Mode.Laplace
                else -> Mode.FirstOrder
            }
        )
        model.chooseInverse(intent.getStringExtra("mode") == "inverse")
        model.setEquationText(equation)
        model.x0 = intent.getStringExtra("x0") ?: ""
        model.y0 = intent.getStringExtra("y0") ?: ""
        model.yPrime0 = intent.getStringExtra("yp0") ?: ""
        intent.getStringExtra("h")?.let { model.stepSize = it }
        intent.getStringExtra("n")?.let { model.stepCount = it }
        model.solve(openTab = intent.getIntExtra("tab", 0))
    }
}
