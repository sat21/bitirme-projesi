package com.tomatech.mobile.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.tomatech.mobile.DecisionThresholds
import com.tomatech.mobile.DiagnosisDecision
import com.tomatech.mobile.DiagnosisStatus
import com.tomatech.mobile.ml.InferenceResult
import com.tomatech.mobile.ml.Prediction
import com.tomatech.mobile.ml.TomatoClasses
import java.util.Locale

@Composable
fun DiagnosisResultCard(
    result: InferenceResult,
    decision: DiagnosisDecision?,
    modifier: Modifier = Modifier,
) {
    val status = decision?.status
    val computedMargin = decision?.margin
        ?: (result.top1.confidence - (result.top3.getOrNull(1)?.confidence ?: 0f))
    val shouldHideClassPredictions = status == DiagnosisStatus.INVALID_IMAGE ||
        (decision?.title?.contains("Gecersiz", ignoreCase = true) == true) ||
        (decision == null &&
            result.top1.confidence < DecisionThresholds.INVALID_IMAGE_CONFIDENCE_THRESHOLD)

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "Teshis Ozeti",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )

            if (decision != null) {
                DecisionBanner(decision = decision)
                ActionGuidanceCard(decision = decision)
            }

            if (shouldHideClassPredictions) {
                InvalidImageSuppressionCard()
            } else {
                TopPredictionRow(
                    prediction = result.top1,
                    status = status
                )

                HorizontalDivider()

                Text(
                    text = if (status == DiagnosisStatus.UNCERTAIN) {
                        "Alternatif Olasiliklar"
                    } else {
                        "Alternatif Tahminler"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold
                )

                result.top3.forEachIndexed { index, prediction ->
                    Text(
                        text = "${index + 1}. ${TomatoClasses.displayName(prediction.label)} - ${prediction.confidence.toPercentText()}",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }

            HorizontalDivider()

            ConfidenceInterpretation(
                status = status,
                topConfidence = decision?.topConfidence ?: result.top1.confidence,
                margin = computedMargin,
                showConfidenceNumbers = !shouldHideClassPredictions
            )

            Text(
                text = "Cikarim suresi: ${result.latencyMs.toMsText()}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun ActionGuidanceCard(decision: DiagnosisDecision) {
    val (containerColor, contentColor, actions) = when (decision.status) {
        DiagnosisStatus.DIAGNOSIS -> Triple(
            MaterialTheme.colorScheme.secondaryContainer,
            MaterialTheme.colorScheme.onSecondaryContainer,
            listOf(
                "Belirti ilerlemesini 24-48 saat aralikla tekrar kontrol edin.",
                "Ayni yapragi farkli aci ve isikta tekrar tarayip sonucu karsilastirin.",
                "Kesin tani icin ziraat uzmani degerlendirmesi alin."
            )
        )

        DiagnosisStatus.UNCERTAIN -> Triple(
            MaterialTheme.colorScheme.tertiaryContainer,
            MaterialTheme.colorScheme.onTertiaryContainer,
            listOf(
                "Tek bir yapragi duz arka planda yeniden cekin.",
                "Isigi artirin ve bulaniklik olusmamasina dikkat edin.",
                "En az 2 farkli kareyle tekrar analiz yapin."
            )
        )

        DiagnosisStatus.INVALID_IMAGE -> Triple(
            MaterialTheme.colorScheme.errorContainer,
            MaterialTheme.colorScheme.onErrorContainer,
            listOf(
                "Kadraja sadece domates yapragi alin.",
                "Kamerayi sabitleyip net bir goruntu cekin.",
                "Gerekirse galeriden daha net bir fotograf secin."
            )
        )
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = "Sonraki Adimlar",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = contentColor
            )

            actions.forEachIndexed { index, action ->
                Text(
                    text = "${index + 1}) $action",
                    style = MaterialTheme.typography.bodySmall,
                    color = contentColor
                )
            }
        }
    }
}

@Composable
private fun InvalidImageSuppressionCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Text(
            text = "Bu durumda sinif tahminleri gizlenir. Once daha net bir yaprak goruntusu alin.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(12.dp)
        )
    }
}

@Composable
private fun ConfidenceInterpretation(
    status: DiagnosisStatus?,
    topConfidence: Float,
    margin: Float,
    showConfidenceNumbers: Boolean,
) {
    val message = when (status) {
        DiagnosisStatus.DIAGNOSIS -> "Bu sonuc on teshistir; karar vermeden once saha gozlemi ve uzman gorusu ile dogrulayin."
        DiagnosisStatus.UNCERTAIN -> "Model kararsiz sinyal verdi. Guvenli karar icin yeniden cekim yapilmasi onerilir."
        DiagnosisStatus.INVALID_IMAGE -> "Goruntu kalite/kapsam acisindan yetersiz bulundu. Sinif sonucu uretmek yerine tekrar cekim yapin."
        null -> "Sonuc yorumu olusturulamadi."
    }

    if (showConfidenceNumbers) {
        Text(
            text = "Model guveni: ${topConfidence.toPercentText()} | Ayrim gucu: ${margin.toPercentText()}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }

    Text(
        text = message,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}

@Composable
private fun DecisionBanner(decision: DiagnosisDecision) {
    val containerColor = when (decision.status) {
        DiagnosisStatus.DIAGNOSIS -> MaterialTheme.colorScheme.primaryContainer
        DiagnosisStatus.UNCERTAIN -> MaterialTheme.colorScheme.tertiaryContainer
        DiagnosisStatus.INVALID_IMAGE -> MaterialTheme.colorScheme.errorContainer
    }

    val contentColor = when (decision.status) {
        DiagnosisStatus.DIAGNOSIS -> MaterialTheme.colorScheme.onPrimaryContainer
        DiagnosisStatus.UNCERTAIN -> MaterialTheme.colorScheme.onTertiaryContainer
        DiagnosisStatus.INVALID_IMAGE -> MaterialTheme.colorScheme.onErrorContainer
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = decision.title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = contentColor
            )

            Text(
                text = decision.message,
                style = MaterialTheme.typography.bodySmall,
                color = contentColor
            )

            if (decision.status != DiagnosisStatus.INVALID_IMAGE) {
                Text(
                    text = "Model guveni: ${decision.topConfidence.toPercentText()} | Ayrim gucu: ${decision.margin.toPercentText()}",
                    style = MaterialTheme.typography.bodySmall,
                    color = contentColor
                )
            }
        }
    }
}

@Composable
private fun TopPredictionRow(
    prediction: Prediction,
    status: DiagnosisStatus?,
) {
    val title = when (status) {
        DiagnosisStatus.UNCERTAIN -> "Olasi Birinci Tahmin"
        else -> "Birinci Tahmin"
    }

    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Text(
                    text = TomatoClasses.displayName(prediction.label),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    fontWeight = FontWeight.Bold
                )
            }

            Text(
                text = prediction.confidence.toPercentText(),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

private fun Float.toPercentText(): String {
    val rawPercent = this * 100f
    val displayPercent = if (this in 0f..<1f) {
        rawPercent.coerceAtMost(99.99f)
    } else {
        rawPercent
    }
    return String.format(Locale.US, "%.2f%%", displayPercent)
}

private fun Float.toMsText(): String {
    return String.format(Locale.US, "%.2f ms", this)
}
