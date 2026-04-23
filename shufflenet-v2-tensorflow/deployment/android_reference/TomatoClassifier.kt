package com.tomatech.ai

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.roundToInt

class TomatoClassifier(
    context: Context,
    modelAssetName: String,
    private val numThreads: Int = 4
) : Closeable {

    private val interpreter: Interpreter
    private val inputDataType: DataType
    private val outputDataType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val outputScale: Float
    private val outputZeroPoint: Int
    private val inputSize: Int
    private val numClasses: Int

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            setUseXNNPACK(true)
        }

        interpreter = Interpreter(loadModelFile(context, modelAssetName), options)

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)

        inputDataType = inputTensor.dataType()
        outputDataType = outputTensor.dataType()

        inputScale = inputTensor.quantizationParams().scale
        inputZeroPoint = inputTensor.quantizationParams().zeroPoint
        outputScale = outputTensor.quantizationParams().scale
        outputZeroPoint = outputTensor.quantizationParams().zeroPoint

        val inputShape = inputTensor.shape()
        val outputShape = outputTensor.shape()

        inputSize = inputShape[1]
        numClasses = outputShape[1]

        require(numClasses == TomatoClasses.labels.size) {
            "Model class count ($numClasses) does not match labels (${TomatoClasses.labels.size})"
        }
    }

    fun classify(bitmap: Bitmap): InferenceResult {
        val floatInput = ImagePreprocessor.bitmapToNormalizedFloatArray(bitmap, inputSize)
        val inputBuffer = toInputBuffer(floatInput)

        val startNs = System.nanoTime()
        val probabilities = runInference(inputBuffer)
        val latencyMs = (System.nanoTime() - startNs) / 1_000_000f

        val ranked = probabilities
            .mapIndexed { index, score -> Prediction(TomatoClasses.labels[index], score) }
            .sortedByDescending { it.confidence }

        return InferenceResult(
            top1 = ranked.first(),
            top3 = ranked.take(3),
            latencyMs = latencyMs
        )
    }

    private fun runInference(inputBuffer: ByteBuffer): FloatArray {
        return when (outputDataType) {
            DataType.FLOAT32 -> {
                val output = Array(1) { FloatArray(numClasses) }
                interpreter.run(inputBuffer, output)
                softmax(output[0])
            }
            DataType.INT8 -> {
                val output = Array(1) { ByteArray(numClasses) }
                interpreter.run(inputBuffer, output)
                val dequantized = FloatArray(numClasses)
                for (i in 0 until numClasses) {
                    dequantized[i] = (output[0][i].toInt() - outputZeroPoint) * outputScale
                }
                softmax(dequantized)
            }
            DataType.UINT8 -> {
                val output = Array(1) { ByteArray(numClasses) }
                interpreter.run(inputBuffer, output)
                val dequantized = FloatArray(numClasses)
                for (i in 0 until numClasses) {
                    val raw = output[0][i].toInt() and 0xFF
                    dequantized[i] = (raw - outputZeroPoint) * outputScale
                }
                softmax(dequantized)
            }
            else -> error("Unsupported output dtype: $outputDataType")
        }
    }

    private fun toInputBuffer(floatInput: FloatArray): ByteBuffer {
        return when (inputDataType) {
            DataType.FLOAT32 -> {
                val buffer = ByteBuffer.allocateDirect(floatInput.size * 4).order(ByteOrder.nativeOrder())
                for (value in floatInput) {
                    buffer.putFloat(value)
                }
                buffer.rewind()
                buffer
            }
            DataType.INT8 -> {
                check(inputScale > 0f) { "INT8 input scale must be > 0" }
                val buffer = ByteBuffer.allocateDirect(floatInput.size).order(ByteOrder.nativeOrder())
                for (value in floatInput) {
                    val quantized = (value / inputScale + inputZeroPoint).roundToInt().coerceIn(-128, 127)
                    buffer.put(quantized.toByte())
                }
                buffer.rewind()
                buffer
            }
            DataType.UINT8 -> {
                check(inputScale > 0f) { "UINT8 input scale must be > 0" }
                val buffer = ByteBuffer.allocateDirect(floatInput.size).order(ByteOrder.nativeOrder())
                for (value in floatInput) {
                    val quantized = (value / inputScale + inputZeroPoint).roundToInt().coerceIn(0, 255)
                    buffer.put(quantized.toByte())
                }
                buffer.rewind()
                buffer
            }
            else -> error("Unsupported input dtype: $inputDataType")
        }
    }

    private fun softmax(values: FloatArray): FloatArray {
        val maxValue = values.maxOrNull() ?: 0f
        val exps = FloatArray(values.size)
        var sum = 0f

        for (i in values.indices) {
            val e = exp(values[i] - maxValue)
            exps[i] = e
            sum += e
        }

        if (sum == 0f) {
            return FloatArray(values.size)
        }

        for (i in exps.indices) {
            exps[i] /= sum
        }

        return exps
    }

    private fun loadModelFile(context: Context, modelAssetName: String): ByteBuffer {
        context.assets.openFd(modelAssetName).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
                val fileChannel = inputStream.channel
                return fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                )
            }
        }
    }

    override fun close() {
        interpreter.close()
    }
}
