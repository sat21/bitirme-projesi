package com.tomatech.mobile

import com.tomatech.mobile.ml.InferenceResult
import com.tomatech.mobile.ml.Prediction
import org.junit.Assert.assertEquals
import org.junit.Test

class TomatoDecisionEngineTest {

    @Test
    fun lowConfidence_isInvalidEvenWhenGuardOff() {
        val result = inferenceResult(
            topLabel = "Tomato___healthy",
            topConfidence = 0.40f,
            secondConfidence = 0.30f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.30f,
            leafLikePixelRatio = 0.30f,
            skinPixelRatio = 0.01f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = false
        )

        assertEquals(DiagnosisStatus.INVALID_IMAGE, decision.status)
    }

    @Test
    fun highConfidenceAndStrongLeafSignal_isDiagnosis() {
        val result = inferenceResult(
            topLabel = "Tomato___healthy",
            topConfidence = 0.96f,
            secondConfidence = 0.03f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.42f,
            leafLikePixelRatio = 0.36f,
            skinPixelRatio = 0.01f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = true
        )

        assertEquals(DiagnosisStatus.DIAGNOSIS, decision.status)
    }

    @Test
    fun lowMargin_isUncertain() {
        val result = inferenceResult(
            topLabel = "Tomato___healthy",
            topConfidence = 0.95f,
            secondConfidence = 0.90f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.30f,
            leafLikePixelRatio = 0.30f,
            skinPixelRatio = 0.00f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = true
        )

        assertEquals(DiagnosisStatus.UNCERTAIN, decision.status)
    }

    @Test
    fun selfieLikeHealthy_withGuardOn_isInvalid() {
        val result = inferenceResult(
            topLabel = "Tomato___healthy",
            topConfidence = 0.96f,
            secondConfidence = 0.03f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.01f,
            leafLikePixelRatio = 0.02f,
            skinPixelRatio = 0.22f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = true
        )

        assertEquals(DiagnosisStatus.INVALID_IMAGE, decision.status)
    }

    @Test
    fun selfieLikeHealthy_withGuardOff_isDiagnosis() {
        val result = inferenceResult(
            topLabel = "Tomato___healthy",
            topConfidence = 0.96f,
            secondConfidence = 0.03f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.01f,
            leafLikePixelRatio = 0.02f,
            skinPixelRatio = 0.22f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = false
        )

        assertEquals(DiagnosisStatus.DIAGNOSIS, decision.status)
    }

    @Test
    fun highConfidenceDiseaseWithLowLeafEvidence_withGuardOn_isInvalid() {
        val result = inferenceResult(
            topLabel = "Tomato___Bacterial_spot",
            topConfidence = 0.96f,
            secondConfidence = 0.03f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.07f,
            leafLikePixelRatio = 0.08f,
            skinPixelRatio = 0.01f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = true
        )

        assertEquals(DiagnosisStatus.INVALID_IMAGE, decision.status)
    }

    @Test
    fun highConfidenceDiseaseWithLowLeafEvidence_withGuardOff_isDiagnosis() {
        val result = inferenceResult(
            topLabel = "Tomato___Bacterial_spot",
            topConfidence = 0.96f,
            secondConfidence = 0.03f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.07f,
            leafLikePixelRatio = 0.08f,
            skinPixelRatio = 0.01f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = false
        )

        assertEquals(DiagnosisStatus.DIAGNOSIS, decision.status)
    }

    @Test
    fun uncertainByConfidence_canBePromotedInvalidWhenHumanBackgroundSignalExists() {
        val result = inferenceResult(
            topLabel = "Tomato___Bacterial_spot",
            topConfidence = 0.78f,
            secondConfidence = 0.21f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.05f,
            leafLikePixelRatio = 0.04f,
            skinPixelRatio = 0.15f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = true
        )

        assertEquals(DiagnosisStatus.INVALID_IMAGE, decision.status)
    }

    @Test
    fun uncertainByConfidence_staysUncertainWhenGuardOff() {
        val result = inferenceResult(
            topLabel = "Tomato___Bacterial_spot",
            topConfidence = 0.78f,
            secondConfidence = 0.21f
        )
        val signals = ImageValiditySignals(
            greenPixelRatio = 0.05f,
            leafLikePixelRatio = 0.04f,
            skinPixelRatio = 0.15f
        )

        val decision = TomatoDecisionEngine.buildDecision(
            result = result,
            imageSignals = signals,
            enableVisualInvalidGuard = false
        )

        assertEquals(DiagnosisStatus.UNCERTAIN, decision.status)
    }

    private fun inferenceResult(
        topLabel: String,
        topConfidence: Float,
        secondConfidence: Float,
        thirdConfidence: Float = 0.01f
    ): InferenceResult {
        return InferenceResult(
            top1 = Prediction(topLabel, topConfidence),
            top3 = listOf(
                Prediction(topLabel, topConfidence),
                Prediction("Tomato___Early_blight", secondConfidence),
                Prediction("Tomato___Late_blight", thirdConfidence)
            ),
            latencyMs = 12.3f
        )
    }
}
