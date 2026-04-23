package com.tomatech.mobile

import com.tomatech.mobile.ml.InferenceResult

data class ImageValiditySignals(
    val greenPixelRatio: Float,
    val leafLikePixelRatio: Float,
    val skinPixelRatio: Float
)

object TomatoDecisionEngine {

    fun buildDecision(
        result: InferenceResult,
        imageSignals: ImageValiditySignals,
        enableVisualInvalidGuard: Boolean = DecisionThresholds.ENABLE_VISUAL_INVALID_GUARD
    ): DiagnosisDecision {
        val topConfidence = result.top1.confidence
        val secondConfidence = result.top3.getOrNull(1)?.confidence ?: 0f
        val margin = topConfidence - secondConfidence

        val promoteToInvalidByVisualSignal = shouldPromoteToInvalidByVisualSignal(
            result = result,
            imageSignals = imageSignals,
            topConfidence = topConfidence,
            enableVisualInvalidGuard = enableVisualInvalidGuard
        )

        return when {
            topConfidence < DecisionThresholds.INVALID_IMAGE_CONFIDENCE_THRESHOLD ||
                promoteToInvalidByVisualSignal -> {
                DiagnosisDecision(
                    status = DiagnosisStatus.INVALID_IMAGE,
                    title = "Gecersiz Goruntu",
                    message = "Domates yapragi net secilemedi. Kadraja tek yaprak alin, isigi artirin ve yeniden cekin.",
                    topConfidence = topConfidence,
                    margin = margin
                )
            }

            topConfidence < DecisionThresholds.CONFIDENT_DIAGNOSIS_THRESHOLD ||
                margin < DecisionThresholds.MIN_MARGIN_THRESHOLD -> {
                DiagnosisDecision(
                    status = DiagnosisStatus.UNCERTAIN,
                    title = "On Teshis Belirsiz",
                    message = "Model kararsiz kaldi. Tek yapragi duz arka planda, yakin plan ve net olarak yeniden cekin.",
                    topConfidence = topConfidence,
                    margin = margin
                )
            }

            else -> {
                DiagnosisDecision(
                    status = DiagnosisStatus.DIAGNOSIS,
                    title = "On Teshis Uretildi",
                    message = "Model yeterli guvenle on teshis uretti. Kesin tani icin uzman degerlendirmesi onerilir.",
                    topConfidence = topConfidence,
                    margin = margin
                )
            }
        }
    }

    private fun shouldPromoteToInvalidByVisualSignal(
        result: InferenceResult,
        imageSignals: ImageValiditySignals,
        topConfidence: Float,
        enableVisualInvalidGuard: Boolean
    ): Boolean {
        if (!enableVisualInvalidGuard) {
            return false
        }

        val isHealthyPrediction = result.top1.label.endsWith("healthy", ignoreCase = true)
        val lowLeafHueSignal =
            imageSignals.leafLikePixelRatio < DecisionThresholds.MIN_LEAF_LIKE_PIXEL_RATIO
        val lowDominantGreenSignal =
            imageSignals.greenPixelRatio < DecisionThresholds.MIN_GREEN_PIXEL_RATIO_FOR_LEAF
        val hardLowLeafSignal =
            imageSignals.greenPixelRatio < DecisionThresholds.HARD_INVALID_GREEN_PIXEL_RATIO
        val dualLowLeafSignal = lowLeafHueSignal && lowDominantGreenSignal
        val healthyWithoutLeafEvidence =
            isHealthyPrediction && (lowLeafHueSignal || lowDominantGreenSignal)
        val likelyHumanOrBackground =
            lowLeafHueSignal &&
                imageSignals.skinPixelRatio > DecisionThresholds.MAX_SKIN_PIXEL_RATIO_WITH_LOW_GREEN

        return hardLowLeafSignal ||
            dualLowLeafSignal ||
            likelyHumanOrBackground ||
            healthyWithoutLeafEvidence ||
            (lowLeafHueSignal &&
                topConfidence < DecisionThresholds.CONFIDENT_DIAGNOSIS_THRESHOLD)
    }
}