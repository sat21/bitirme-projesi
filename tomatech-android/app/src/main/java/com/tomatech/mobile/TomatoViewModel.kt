package com.tomatech.mobile

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.tomatech.mobile.ml.InferenceResult
import com.tomatech.mobile.ml.TomatoClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs

private const val DEFAULT_MODEL_ASSET = "checkpoints_tomato_1_5x_baseline_best_model_int8.tflite"

enum class DiagnosisStatus {
    DIAGNOSIS,
    UNCERTAIN,
    INVALID_IMAGE
}

data class DiagnosisDecision(
    val status: DiagnosisStatus,
    val title: String,
    val message: String,
    val topConfidence: Float,
    val margin: Float
)

data class TomatoUiState(
    val selectedBitmap: Bitmap? = null,
    val result: InferenceResult? = null,
    val decision: DiagnosisDecision? = null,
    val isRunning: Boolean = false,
    val errorMessage: String? = null
)

private data class DiagnosisComputation(
    val result: InferenceResult,
    val imageSignals: ImageValiditySignals
)

class TomatoViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(TomatoUiState())
    val uiState: StateFlow<TomatoUiState> = _uiState.asStateFlow()

    private val classifier: TomatoClassifier? = runCatching {
        TomatoClassifier(
            context = application,
            modelAssetName = DEFAULT_MODEL_ASSET,
            numThreads = 4
        )
    }.onFailure { throwable ->
        _uiState.update {
            it.copy(errorMessage = "Model yuklenemedi: ${throwable.message}")
        }
    }.getOrNull()

    fun onImageSelected(bitmap: Bitmap) {
        _uiState.update {
            it.copy(
                selectedBitmap = bitmap,
                result = null,
                decision = null,
                errorMessage = null
            )
        }
    }

    fun setError(message: String) {
        _uiState.update { it.copy(errorMessage = message) }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    fun runDiagnosis() {
        val snapshot = _uiState.value

        if (snapshot.isRunning) {
            return
        }

        val image = snapshot.selectedBitmap
        if (image == null) {
            setError("Lutfen once tek bir yaprak fotografi secin.")
            return
        }

        val activeClassifier = classifier
        if (activeClassifier == null) {
            setError("Model baslatilamadi. Uygulamayi yeniden acin.")
            return
        }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isRunning = true,
                    result = null,
                    decision = null,
                    errorMessage = null
                )
            }

            val inferenceResult = runCatching {
                withContext(Dispatchers.Default) {
                    val imageSignals = analyzeImageSignals(image)
                    val inference = activeClassifier.classify(image)
                    DiagnosisComputation(
                        result = inference,
                        imageSignals = imageSignals
                    )
                }
            }

            inferenceResult
                .onSuccess { computation ->
                    val decision = TomatoDecisionEngine.buildDecision(
                        result = computation.result,
                        imageSignals = computation.imageSignals
                    )

                    if (DecisionThresholds.ENABLE_DECISION_DEBUG_LOGS) {
                        Log.d(
                            "TomatoDecision",
                            "status=${decision.status} top=${"%.3f".format(decision.topConfidence)} " +
                                "margin=${"%.3f".format(decision.margin)} label=${computation.result.top1.label} " +
                                "green=${"%.3f".format(computation.imageSignals.greenPixelRatio)} " +
                                "leafLike=${"%.3f".format(computation.imageSignals.leafLikePixelRatio)} " +
                                "skin=${"%.3f".format(computation.imageSignals.skinPixelRatio)}"
                        )
                    }

                    _uiState.update {
                        it.copy(
                            isRunning = false,
                            result = computation.result,
                            decision = decision
                        )
                    }
                }
                .onFailure { throwable ->
                    _uiState.update {
                        it.copy(
                            isRunning = false,
                            errorMessage = "Teshis sirasinda hata olustu: ${throwable.message}"
                        )
                    }
                }
        }
    }

    override fun onCleared() {
        classifier?.close()
        super.onCleared()
    }

    private fun analyzeImageSignals(bitmap: Bitmap): ImageValiditySignals {
        val sampledBitmap = downscaleForSignalCheck(bitmap, maxEdge = 160)
        val width = sampledBitmap.width
        val height = sampledBitmap.height
        val pixels = IntArray(width * height)
        sampledBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var greenLikePixelCount = 0
        var leafLikePixelCount = 0
        var skinLikePixelCount = 0
        val hsv = FloatArray(3)
        pixels.forEach { pixel ->
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            val blue = Color.blue(pixel)

            val maxChannel = maxOf(red, green, blue)
            val minChannel = minOf(red, green, blue)
            val saturation = if (maxChannel == 0) {
                0f
            } else {
                (maxChannel - minChannel).toFloat() / maxChannel.toFloat()
            }

            Color.RGBToHSV(red, green, blue, hsv)
            val hue = hsv[0]

            val isGreenLike =
                green >= 45 &&
                    green >= red * 1.08f &&
                    green >= blue * 1.05f &&
                    (green - maxOf(red, blue)) >= 8 &&
                    saturation >= 0.16f

            if (isGreenLike) {
                greenLikePixelCount += 1
            }

            val isLeafLike =
                saturation >= 0.14f &&
                    hue in 35f..170f

            if (isLeafLike) {
                leafLikePixelCount += 1
            }

            val isSkinLike =
                red > 95 &&
                    green > 40 &&
                    blue > 20 &&
                    (maxChannel - minChannel) > 15 &&
                    abs(red - green) > 15 &&
                    red > green &&
                    red > blue

            if (isSkinLike) {
                skinLikePixelCount += 1
            }
        }

        val totalPixelCount = pixels.size.coerceAtLeast(1)
        return ImageValiditySignals(
            greenPixelRatio = greenLikePixelCount.toFloat() / totalPixelCount.toFloat(),
            leafLikePixelRatio = leafLikePixelCount.toFloat() / totalPixelCount.toFloat(),
            skinPixelRatio = skinLikePixelCount.toFloat() / totalPixelCount.toFloat()
        )
    }

    private fun downscaleForSignalCheck(bitmap: Bitmap, maxEdge: Int): Bitmap {
        val sourceWidth = bitmap.width
        val sourceHeight = bitmap.height
        val longestEdge = maxOf(sourceWidth, sourceHeight)

        if (longestEdge <= maxEdge) {
            return bitmap
        }

        val scale = maxEdge.toFloat() / longestEdge.toFloat()
        val targetWidth = (sourceWidth * scale).toInt().coerceAtLeast(1)
        val targetHeight = (sourceHeight * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }
}
