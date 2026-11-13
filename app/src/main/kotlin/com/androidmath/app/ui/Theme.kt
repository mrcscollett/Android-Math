package com.androidmath.app.ui

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext

private val Light = lightColorScheme(
    primary = Color(0xFF3F51B5),
    primaryContainer = Color(0xFFE0E3FF),
    onPrimaryContainer = Color(0xFF00105C),
    tertiary = Color(0xFFB85C00),
)

private val Dark = darkColorScheme(
    primary = Color(0xFFBAC3FF),
    onPrimary = Color(0xFF08218A),
    primaryContainer = Color(0xFF293CA0),
    onPrimaryContainer = Color(0xFFE0E3FF),
    tertiary = Color(0xFFFFB870),
)

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val scheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(LocalContext.current) else dynamicLightColorScheme(LocalContext.current)
        dark -> Dark
        else -> Light
    }
    MaterialTheme(colorScheme = scheme, content = content)
}

/** CSS colour (#rrggbb) for passing theme colours to the steps page. */
fun Color.toCss(): String = "#%06x".format(toArgb() and 0xFFFFFF)

/** Good (verified) and bad (failed check) colours that read on both themes. */
@Composable
fun okColor(): Color = if (isSystemInDarkTheme()) Color(0xFF7DDB9C) else Color(0xFF1B6E3C)
