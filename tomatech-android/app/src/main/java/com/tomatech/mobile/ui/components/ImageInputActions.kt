package com.tomatech.mobile.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ImageInputActions(
    onPickFromGallery: () -> Unit,
    onCaptureWithCamera: () -> Unit,
    enabled: Boolean = true,
    galleryButtonLabel: String = "Galeriden Sec",
    cameraButtonLabel: String = "Kamera",
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        OutlinedButton(
            onClick = onPickFromGallery,
            enabled = enabled,
            modifier = Modifier
                .weight(1f)
                .height(52.dp)
        ) {
            Text(galleryButtonLabel)
        }

        Button(
            onClick = onCaptureWithCamera,
            enabled = enabled,
            modifier = Modifier
                .weight(1f)
                .height(52.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary
            )
        ) {
            Text(cameraButtonLabel)
        }
    }
}
