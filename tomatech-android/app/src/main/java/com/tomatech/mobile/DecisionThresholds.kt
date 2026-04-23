package com.tomatech.mobile

object DecisionThresholds {
    // Quick rollback: set to false to return to confidence/margin-only decision behavior.
    const val ENABLE_VISUAL_INVALID_GUARD = true
    const val ENABLE_DECISION_DEBUG_LOGS = false

    const val INVALID_IMAGE_CONFIDENCE_THRESHOLD = 0.70f
    const val CONFIDENT_DIAGNOSIS_THRESHOLD = 0.90f
    const val MIN_MARGIN_THRESHOLD = 0.10f
    const val MIN_GREEN_PIXEL_RATIO_FOR_LEAF = 0.08f
    const val MIN_LEAF_LIKE_PIXEL_RATIO = 0.12f
    const val HARD_INVALID_GREEN_PIXEL_RATIO = 0.03f
    const val MAX_SKIN_PIXEL_RATIO_WITH_LOW_GREEN = 0.08f
}
