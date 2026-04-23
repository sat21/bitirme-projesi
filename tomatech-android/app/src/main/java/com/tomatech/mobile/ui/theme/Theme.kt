package com.tomatech.mobile.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.dp

private val LightColors = lightColorScheme(
    primary = LeafDeep,
    onPrimary = PureWhite,
    primaryContainer = SuccessContainer,
    onPrimaryContainer = LeafDeep,
    secondary = LeafFresh,
    onSecondary = PureWhite,
    secondaryContainer = FieldMist,
    onSecondaryContainer = InkDark,
    tertiary = SunAmber,
    onTertiary = InkDark,
    tertiaryContainer = WarningContainer,
    onTertiaryContainer = InkDark,
    background = DawnPaper,
    onBackground = InkDark,
    surface = CreamCard,
    onSurface = InkDark,
    surfaceVariant = FieldMist,
    onSurfaceVariant = InkMuted,
    error = AlertRed,
    onError = PureWhite,
    errorContainer = AlertContainer,
    onErrorContainer = AlertRed,
    outline = SoilRich
)

private val AppShapes = Shapes(
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(20.dp),
    large = RoundedCornerShape(28.dp)
)

@Composable
fun TomaTechTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColors,
        typography = AppTypography,
        shapes = AppShapes,
        content = content
    )
}
