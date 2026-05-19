package com.tomatech.mobile

object DecisionThresholds {
    // 11. Sınıfı (Background) eğittiğimiz için artık renk tabanlı manuel filtreye (Visual Guard) ihtiyacımız yok. Hastalıklı (sarı/kahverengi) yaprakları yanlışlıkla elememesi için kapattık.
    const val ENABLE_VISUAL_INVALID_GUARD = false
    const val ENABLE_DECISION_DEBUG_LOGS = true

    // Çok sınırda kalan net bitkileri de kabul etmesi için alt limiti daha da düşürdük
    const val INVALID_IMAGE_CONFIDENCE_THRESHOLD = 0.45f
    const val CONFIDENT_DIAGNOSIS_THRESHOLD = 0.75f
    
    // Modelin en yüksek 2 tahmini arasındaki fark (Kararsızlık payı)
    const val MIN_MARGIN_THRESHOLD = 0.05f
    const val MIN_TOP3_MASS_FOR_DIAGNOSIS = 0.80f
    const val MAX_NORMALIZED_ENTROPY_FOR_DIAGNOSIS = 0.60f
    const val MIN_GREEN_PIXEL_RATIO_FOR_LEAF = 0.08f
    const val MIN_LEAF_LIKE_PIXEL_RATIO = 0.12f
    const val HARD_INVALID_GREEN_PIXEL_RATIO = 0.03f
    const val MAX_SKIN_PIXEL_RATIO_WITH_LOW_GREEN = 0.08f
}
