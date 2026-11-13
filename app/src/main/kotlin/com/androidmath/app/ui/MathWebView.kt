package com.androidmath.app.ui

import android.annotation.SuppressLint
import android.graphics.Color as AndroidColor
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

private class PageState {
    var loaded = false
    var pending = ""
    var applied = ""
}

/**
 * Shows the bundled KaTeX page and runs [script] in it whenever it changes.
 * The page exposes showPreview(...), showSolution(...) and showMethod(...).
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun MathWebView(script: String, modifier: Modifier = Modifier) {
    val colors = MaterialTheme.colorScheme
    val ok = okColor()
    val theme = "setTheme({" +
        "'text':'${colors.onSurface.toCss()}'," +
        "'muted':'${colors.onSurfaceVariant.toCss()}'," +
        "'card':'${colors.surfaceVariant.toCss()}'," +
        "'outline':'${colors.outlineVariant.toCss()}'," +
        "'accent':'${colors.primary.toCss()}'," +
        "'on-accent':'${colors.onPrimary.toCss()}'," +
        "'accent-soft':'${colors.primaryContainer.toCss()}'," +
        "'ok':'${ok.toCss()}'," +
        "'bad':'${colors.error.toCss()}'});"
    val full = theme + script
    val state = remember { PageState() }

    AndroidView(
        modifier = modifier,
        factory = { context ->
            WebView(context).apply {
                settings.javaScriptEnabled = true
                // Only the bundled page is ever shown; nothing is loaded from the network.
                settings.blockNetworkLoads = true
                setBackgroundColor(AndroidColor.TRANSPARENT)
                overScrollMode = android.view.View.OVER_SCROLL_NEVER
                webViewClient = object : WebViewClient() {
                    // Without this, a crashed renderer process takes the whole app down.
                    override fun onRenderProcessGone(
                        view: WebView, detail: android.webkit.RenderProcessGoneDetail,
                    ): Boolean {
                        android.util.Log.e("MathWebView", "renderer gone, crashed=${detail.didCrash()}")
                        state.loaded = false
                        return true
                    }

                    override fun onPageFinished(view: WebView, url: String?) {
                        state.loaded = true
                        state.applied = state.pending
                        view.evaluateJavascript(state.pending, null)
                    }
                }
                loadUrl("file:///android_asset/math/steps.html")
            }
        },
        onRelease = { view -> view.destroy() },
        update = { view ->
            state.pending = full
            if (state.loaded && state.applied != full) {
                state.applied = full
                view.evaluateJavascript(full, null)
            }
        },
    )
}

/** A JavaScript string literal for [text]. */
fun jsString(text: String): String = org.json.JSONObject.quote(text)
