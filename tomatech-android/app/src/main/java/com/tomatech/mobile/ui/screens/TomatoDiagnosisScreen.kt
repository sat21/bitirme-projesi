package com.tomatech.mobile.ui.screens

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.tomatech.mobile.TomatoViewModel
import com.tomatech.mobile.ui.components.CameraPreviewCard
import com.tomatech.mobile.ui.components.DiagnosisActionButton
import com.tomatech.mobile.ui.components.DiagnosisResultCard
import com.tomatech.mobile.ui.components.ErrorMessageCard
import com.tomatech.mobile.ui.components.ImageInputActions
import com.tomatech.mobile.ui.components.SelectedImagePreview

private data class CameraUxHint(
    val title: String,
    val message: String,
    val showSettingsAction: Boolean = false
)

@Composable
fun TomatoDiagnosisScreen(viewModel: TomatoViewModel) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()
    val bitmap = uiState.selectedBitmap

    var isCameraPreviewOpen by remember { mutableStateOf(false) }
    var isCapturingWithCamera by remember { mutableStateOf(false) }
    var cameraUxHint by remember { mutableStateOf<CameraUxHint?>(null) }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri == null) {
            return@rememberLauncherForActivityResult
        }

        decodeBitmap(context, uri)
            .onSuccess {
                isCameraPreviewOpen = false
                isCapturingWithCamera = false
                cameraUxHint = null
                viewModel.onImageSelected(it)
            }
            .onFailure { viewModel.setError("Gorsel okunamadi: ${it.message}") }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (!granted) {
            isCameraPreviewOpen = false
            isCapturingWithCamera = false
            val canShowRationale = context.findActivity()?.let {
                ActivityCompat.shouldShowRequestPermissionRationale(it, Manifest.permission.CAMERA)
            } ?: false

            cameraUxHint = if (canShowRationale) {
                CameraUxHint(
                    title = "Kamera izni gerekli",
                    message = "Kamera ile cekim icin izin vermeniz gerekiyor. Kamera butonuna tekrar dokunup izni onaylayin."
                )
            } else {
                CameraUxHint(
                    title = "Kamera izni kapali",
                    message = "Kamera izni kalici olarak kapali olabilir. Ayarlar ekranindan izinleri acip tekrar deneyin.",
                    showSettingsAction = true
                )
            }
            return@rememberLauncherForActivityResult
        }

        cameraUxHint = null
        isCameraPreviewOpen = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.background,
                        MaterialTheme.colorScheme.surfaceVariant
                    )
                )
            )
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.25f))
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "OFFLINE AI",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.secondary,
                    fontWeight = FontWeight.SemiBold
                )

                Text(
                    text = "TomaTech - Saha Tarama Modu",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Text(
            text = "Domates Hastalik Teshisi",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Kamera ile cekin veya galeriden bir yaprak fotografi secin.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        ImageInputActions(
            onPickFromGallery = {
                cameraUxHint = null
                isCameraPreviewOpen = false
                isCapturingWithCamera = false
                viewModel.clearError()
                galleryLauncher.launch("image/*")
            },
            onCaptureWithCamera = {
                cameraUxHint = null
                viewModel.clearError()

                if (isCameraPreviewOpen) {
                    isCameraPreviewOpen = false
                    isCapturingWithCamera = false
                    return@ImageInputActions
                }

                if (hasCameraPermission(context)) {
                    isCameraPreviewOpen = true
                } else {
                    cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                }
            },
            enabled = !uiState.isRunning && !isCapturingWithCamera,
            cameraButtonLabel = when {
                isCameraPreviewOpen -> "Kamerayi Kapat"
                bitmap == null -> "Kamera"
                else -> "Yeniden Cek"
            }
        )

        if (isCameraPreviewOpen) {
            CameraPreviewCard(
                isCapturing = isCapturingWithCamera,
                onCaptureStart = { isCapturingWithCamera = true },
                onCaptureEnd = { isCapturingWithCamera = false },
                onPhotoCaptured = { capturedUri ->
                    decodeBitmap(context, capturedUri)
                        .onSuccess {
                            isCameraPreviewOpen = false
                            cameraUxHint = CameraUxHint(
                                title = "Fotograf hazir",
                                message = "Goruntu secildi. Simdi Analizi Baslat ile teshisi calistirabilirsiniz."
                            )
                            viewModel.onImageSelected(it)
                        }
                        .onFailure {
                            viewModel.setError("Kamera gorseli okunamadi: ${it.message}")
                        }
                },
                onError = { error ->
                    isCapturingWithCamera = false
                    viewModel.setError(error)
                },
                onClose = {
                    isCameraPreviewOpen = false
                    isCapturingWithCamera = false
                    cameraUxHint = CameraUxHint(
                        title = "Kamera cekimi iptal edildi",
                        message = "Fotograf kaydedilmedi. Yeniden Cek ile tekrar deneyin."
                    )
                }
            )
        }

        cameraUxHint?.let { hint ->
            CameraUxHintCard(
                hint = hint,
                onOpenSettings = { openAppSettings(context) },
                onDismiss = { cameraUxHint = null }
            )
        }

        if (bitmap != null) {
            SelectedImagePreview(bitmap = bitmap)
        }

        DiagnosisActionButton(
            enabled = bitmap != null && !uiState.isRunning && !isCapturingWithCamera && !isCameraPreviewOpen,
            isRunning = uiState.isRunning,
            onClick = viewModel::runDiagnosis
        )

        uiState.errorMessage?.let { error ->
            ErrorMessageCard(message = error)
        }

        uiState.result?.let { result ->
            DiagnosisResultCard(
                result = result,
                decision = uiState.decision
            )
        }
    }
}

@Composable
private fun CameraUxHintCard(
    hint: CameraUxHint,
    onOpenSettings: () -> Unit,
    onDismiss: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = hint.title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onTertiaryContainer
            )

            Text(
                text = hint.message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer
            )

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (hint.showSettingsAction) {
                    OutlinedButton(onClick = onOpenSettings) {
                        Text("Ayarlari Ac")
                    }
                }

                OutlinedButton(onClick = onDismiss) {
                    Text("Kapat")
                }
            }
        }
    }
}

private fun hasCameraPermission(context: Context): Boolean {
    return ContextCompat.checkSelfPermission(
        context,
        Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED
}

private fun openAppSettings(context: Context) {
    val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
        data = Uri.fromParts("package", context.packageName, null)
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
    context.startActivity(intent)
}

private tailrec fun Context.findActivity(): Activity? {
    return when (this) {
        is Activity -> this
        is ContextWrapper -> baseContext.findActivity()
        else -> null
    }
}

private fun decodeBitmap(context: Context, uri: Uri): Result<Bitmap> {
    return runCatching {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
        }
    }
}
