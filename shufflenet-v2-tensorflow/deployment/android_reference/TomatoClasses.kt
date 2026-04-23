package com.tomatech.ai

data class Prediction(
    val label: String,
    val confidence: Float
)

data class InferenceResult(
    val top1: Prediction,
    val top3: List<Prediction>,
    val latencyMs: Float
)

object TomatoClasses {
    val labels: List<String> = listOf(
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    )
}
