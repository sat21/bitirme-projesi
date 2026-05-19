package com.tomatech.mobile.data

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream
import java.util.*
import org.json.JSONArray
import org.json.JSONObject

data class HistoryItem(
    val id: String,
    val label: String,
    val confidence: Float,
    val timestamp: Long,
    val imagePath: String
)

class HistoryRepository(private val context: Context) {
    private val prefs = context.getSharedPreferences("history_prefs", Context.MODE_PRIVATE)
    private val imagesDir = File(context.filesDir, "diagnosis_images").apply { if (!exists()) mkdirs() }

    fun saveHistoryItem(label: String, confidence: Float, bitmap: Bitmap) {
        val id = UUID.randomUUID().toString()
        val imageFile = File(imagesDir, "$id.jpg")
        
        FileOutputStream(imageFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out)
        }

        val newItem = HistoryItem(
            id = id,
            label = label,
            confidence = confidence,
            timestamp = System.currentTimeMillis(),
            imagePath = imageFile.absolutePath
        )

        val items = getAllHistory().toMutableList()
        items.add(0, newItem) // Add to top
        
        saveItems(items)
    }

    fun getAllHistory(): List<HistoryItem> {
        val json = prefs.getString("items", null) ?: return emptyList()
        val array = JSONArray(json)
        val items = mutableListOf<HistoryItem>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            items.add(
                HistoryItem(
                    id = obj.getString("id"),
                    label = obj.getString("label"),
                    confidence = obj.getDouble("confidence").toFloat(),
                    timestamp = obj.getLong("timestamp"),
                    imagePath = obj.getString("imagePath")
                )
            )
        }
        return items
    }

    private fun saveItems(items: List<HistoryItem>) {
        val array = JSONArray()
        items.forEach { item ->
            val obj = JSONObject().apply {
                put("id", item.id)
                put("label", item.label)
                put("confidence", item.confidence)
                put("timestamp", item.timestamp)
                put("imagePath", item.imagePath)
            }
            array.put(obj)
        }
        prefs.edit().putString("items", array.toString()).apply()
    }

    fun clearHistory() {
        imagesDir.deleteRecursively()
        imagesDir.mkdirs()
        prefs.edit().remove("items").apply()
    }
}
