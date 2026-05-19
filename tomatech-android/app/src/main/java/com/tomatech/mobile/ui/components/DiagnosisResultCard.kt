package com.tomatech.mobile.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.rounded.Insights
import androidx.compose.material.icons.rounded.Biotech
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tomatech.mobile.DiagnosisDecision
import com.tomatech.mobile.ml.InferenceResult
import com.tomatech.mobile.ui.theme.*
import java.util.Locale

@Composable
fun DiagnosisResultCard(result: InferenceResult, decision: DiagnosisDecision?, modifier: Modifier = Modifier) {
    val isHealthy = result.top1.label.lowercase().contains("healthy")
    val isBackground = result.top1.label.lowercase().contains("background")

    val statusColor = when {
        isHealthy -> SuccessGreen
        isBackground -> Color.Gray
        else -> ErrorRed
    }

    val displayIcon = when {
        isHealthy -> Icons.Default.CheckCircle
        isBackground -> Icons.Default.Info
        else -> Icons.Default.Warning
    }
    
    // Animating the card layout appearance
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(result) {
        visible = false
        visible = true
    }

    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(tween(500)) + slideInVertically(initialOffsetY = { 50 }, animationSpec = tween(500, easing = FastOutSlowInEasing))
    ) {
        Card(
            modifier = modifier
                .fillMaxWidth()
                .padding(vertical = 12.dp)
                .shadow(
                    elevation = 24.dp,
                    shape = RoundedCornerShape(32.dp),
                    spotColor = statusColor.copy(alpha = 0.4f)
                ),
            shape = RoundedCornerShape(32.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Brush.linearGradient(
                            colors = listOf(
                                Color.White,
                                statusColor.copy(alpha = 0.03f)
                            )
                        )
                    )
            ) {
                Column(modifier = Modifier.padding(28.dp)) {
                    // Header Section
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                        // Glowing Icon Wrapper (Glass effect)
                        Box(
                            modifier = Modifier
                                .size(64.dp)
                                .clip(RoundedCornerShape(20.dp))
                                .background(
                                    Brush.linearGradient(
                                        colors = listOf(
                                            statusColor.copy(alpha = 0.15f),
                                            statusColor.copy(alpha = 0.05f)
                                        )
                                    )
                                )
                                .border(1.dp, statusColor.copy(alpha = 0.3f), RoundedCornerShape(20.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(displayIcon, contentDescription = null, tint = statusColor, modifier = Modifier.size(36.dp))
                        }

                        Spacer(modifier = Modifier.width(20.dp))

                        Column {
                            Text("YZ Analiz Sonucu", style = MaterialTheme.typography.labelLarge, color = statusColor, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                            Text(
                                text = decision?.title ?: "Belirlenemedi",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Black,
                                color = TextPrimary
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(28.dp))
                    HorizontalDivider(color = Color.Gray.copy(alpha = 0.15f))
                    Spacer(modifier = Modifier.height(28.dp))

                    if (!isBackground) {
                        // Technical Title
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Rounded.Biotech, contentDescription = null, tint = statusColor, modifier = Modifier.size(20.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "TEKNİK DETAYLAR",
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.ExtraBold,
                                color = TextSecondary,
                                letterSpacing = 1.5.sp
                            )
                        }

                        Spacer(modifier = Modifier.height(20.dp))

                        // Glassmorphic Info Boxes
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            DetailBox(
                                label = "Model Yüzdesi",
                                value = String.format(Locale.getDefault(), "%%%d", (result.top1.confidence * 100).toInt()),
                                icon = Icons.Rounded.Insights,
                                color = statusColor,
                                modifier = Modifier.weight(1f)
                            )
                            Spacer(modifier = Modifier.width(16.dp))
                            DetailBox(
                                label = "Teşhis Sınıfı",
                                value = result.top1.label.removePrefix("Tomato___").replace("_", " "),
                                icon = null,
                                color = TextPrimary,
                                modifier = Modifier.weight(1f)
                            )
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        // Animated Progress Bar
                        val targetProgress = result.top1.confidence
                        val animatedProgress by animateFloatAsState(
                            targetValue = if (visible) targetProgress else 0f,
                            animationSpec = tween(1000, delayMillis = 300, easing = FastOutSlowInEasing),
                            label = "ConfidenceProgress"
                        )

                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(16.dp))
                                .background(statusColor.copy(alpha = 0.05f))
                                .padding(16.dp)
                        ) {
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Güven Aralığı", style = MaterialTheme.typography.bodySmall, color = TextSecondary, fontWeight = FontWeight.Bold)
                                Text(
                                    text = if (targetProgress > 0.8) "Yüksek Hassasiyet" else "Orta Hassasiyet",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.Black,
                                    color = statusColor
                                )
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                            LinearProgressIndicator(
                                progress = { animatedProgress },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(8.dp)
                                    .clip(CircleShape),
                                color = statusColor,
                                trackColor = statusColor.copy(alpha = 0.15f)
                            )
                        }
                    } else {
                        // Background case (Premium)
                        Surface(
                            color = Color.Gray.copy(alpha = 0.05f),
                            shape = RoundedCornerShape(16.dp),
                            border = BorderStroke(1.dp, Color.Gray.copy(alpha = 0.1f)),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.Info, contentDescription = null, tint = TextSecondary, modifier = Modifier.size(24.dp))
                                Spacer(modifier = Modifier.width(12.dp))
                                Text(
                                    text = "Görüntüde yaprak tespit edilemedi. Lütfen daha yakından ve net bir fotoğraf çekiniz.",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = TextSecondary
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DetailBox(label: String, value: String, icon: androidx.compose.ui.graphics.vector.ImageVector?, color: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFFF7F9FC))
            .border(1.dp, Color.White, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, tint = color.copy(alpha = 0.7f), modifier = Modifier.size(18.dp))
            Spacer(modifier = Modifier.height(4.dp))
        }
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = TextSecondary)
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = value, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Black, color = color, maxLines = 2)
    }
}
