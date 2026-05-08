package com.tomatech.mobile.ml

import android.graphics.Bitmap

object ImagePreprocessor {
    fun bitmapToNormalizedFloatArray(
        bitmap: Bitmap,
        inputSize: Int = 224
    ): FloatArray {
        val sourceBitmap = if (bitmap.config == Bitmap.Config.HARDWARE) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false) ?: bitmap
        } else {
            bitmap
        }

        // Center Crop
        val width = sourceBitmap.width
        val height = sourceBitmap.height
        val minDim = Math.min(width, height)
        val startX = (width - minDim) / 2
        val startY = (height - minDim) / 2

        val croppedBitmap = Bitmap.createBitmap(sourceBitmap, startX, startY, minDim, minDim)
        
        // Yeniden boyutlandırma (Squashing/ezilme olmadan temiz kare)
        val resized = Bitmap.createScaledBitmap(croppedBitmap, inputSize, inputSize, true)
        val pixelCount = inputSize * inputSize
        val pixels = IntArray(pixelCount)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val values = FloatArray(pixelCount * 3)
        var outIndex = 0

        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            values[outIndex++] = r / 127.5f - 1.0f
            values[outIndex++] = g / 127.5f - 1.0f
            values[outIndex++] = b / 127.5f - 1.0f
        }

        if (resized !== croppedBitmap) {
            resized.recycle()
        }
        if (croppedBitmap !== sourceBitmap) {
            croppedBitmap.recycle()
        }
        if (sourceBitmap !== bitmap) {
            sourceBitmap.recycle()
        }

        return values
    }
}
