package com.tomatech.mobile.ml

object ModelCalibration {
    // Model zaten etiket yumuşatma (label_smoothing=0.1) ile eğitildiğinden ekstra bir yumuşatmaya (temperature scaling) ihtiyacımız yok.
    // Orijinal logitleri 1.0f ile bölerek direkt softmax uyguluyoruz.
    const val TEMPERATURE_SCALING_FACTOR = 1.0f
}
