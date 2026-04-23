package com.tomatech.mobile.ui.components

import android.annotation.SuppressLint
import android.net.Uri
import android.view.MotionEvent
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.io.File
import java.util.concurrent.TimeUnit

@SuppressLint("ClickableViewAccessibility")
@Composable
fun CameraPreviewCard(
    isCapturing: Boolean,
    onCaptureStart: () -> Unit,
    onCaptureEnd: () -> Unit,
    onPhotoCaptured: (Uri) -> Unit,
    onError: (String) -> Unit,
    onClose: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var boundCamera by remember { mutableStateOf<Camera?>(null) }
    var hasFlashUnit by remember { mutableStateOf(false) }
    var isFlashEnabled by remember { mutableStateOf(false) }
    var lensFacing by remember { mutableStateOf(CameraSelector.LENS_FACING_BACK) }

    DisposableEffect(lifecycleOwner, previewView, lensFacing) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val bindCamera = Runnable {
            runCatching {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val capture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                capture.flashMode = ImageCapture.FLASH_MODE_OFF

                var activeLensFacing = lensFacing
                var selector = CameraSelector.Builder()
                    .requireLensFacing(activeLensFacing)
                    .build()

                if (!cameraProvider.hasCamera(selector)) {
                    val fallbackLensFacing = if (activeLensFacing == CameraSelector.LENS_FACING_BACK) {
                        CameraSelector.LENS_FACING_FRONT
                    } else {
                        CameraSelector.LENS_FACING_BACK
                    }

                    val fallbackSelector = CameraSelector.Builder()
                        .requireLensFacing(fallbackLensFacing)
                        .build()

                    if (!cameraProvider.hasCamera(fallbackSelector)) {
                        error("Bu cihazda kullanilabilir kamera bulunamadi.")
                    }

                    activeLensFacing = fallbackLensFacing
                    selector = fallbackSelector
                    lensFacing = fallbackLensFacing
                    onError("Secilen kamera bulunamadi. Mevcut kameraya gecis yapildi.")
                }

                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    selector,
                    preview,
                    capture
                )

                boundCamera = camera
                hasFlashUnit = camera.cameraInfo.hasFlashUnit()
                if (!hasFlashUnit) {
                    isFlashEnabled = false
                } else {
                    camera.cameraControl.enableTorch(isFlashEnabled)
                }

                imageCapture = capture
            }.onFailure {
                onError("Kamera baslatilamadi: ${it.message}")
            }
        }

        cameraProviderFuture.addListener(bindCamera, mainExecutor)

        onDispose {
            runCatching {
                boundCamera?.cameraControl?.enableTorch(false)
            }
            boundCamera = null
            runCatching {
                cameraProviderFuture.get().unbindAll()
            }
        }
    }

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "Canli Kamera Onizleme",
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = "Odaklamak icin onizlemeye dokunun.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            AndroidView(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp),
                factory = { previewView },
                update = { view ->
                    view.setOnTouchListener { _, event ->
                        if (event.action == MotionEvent.ACTION_UP) {
                            val activeCamera = boundCamera
                            if (activeCamera == null) {
                                onError("Kamera odagi henuz hazir degil.")
                                return@setOnTouchListener true
                            }

                            val point = view.meteringPointFactory.createPoint(event.x, event.y)
                            val action = FocusMeteringAction.Builder(
                                point,
                                FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
                            )
                                .setAutoCancelDuration(3, TimeUnit.SECONDS)
                                .build()

                            runCatching {
                                activeCamera.cameraControl.startFocusAndMetering(action)
                            }.onFailure {
                                onError("Odaklama baslatilamadi: ${it.message}")
                            }
                        }
                        true
                    }
                }
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = onClose,
                    enabled = !isCapturing
                ) {
                    Text("Kapat")
                }

                OutlinedButton(
                    onClick = {
                        runCatching {
                            boundCamera?.cameraControl?.enableTorch(false)
                        }
                        isFlashEnabled = false
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                            CameraSelector.LENS_FACING_FRONT
                        } else {
                            CameraSelector.LENS_FACING_BACK
                        }
                    },
                    enabled = !isCapturing
                ) {
                    val lensLabel = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                        "On Kameraya Gec"
                    } else {
                        "Arka Kameraya Gec"
                    }
                    Text(lensLabel)
                }
            }

            OutlinedButton(
                onClick = {
                    val activeCamera = boundCamera
                    if (activeCamera == null) {
                        onError("Kamera henuz hazir degil. Flash degistirilemedi.")
                        return@OutlinedButton
                    }

                    val newValue = !isFlashEnabled
                    runCatching {
                        activeCamera.cameraControl.enableTorch(newValue)
                        isFlashEnabled = newValue
                    }.onFailure {
                        onError("Flash degistirilemedi: ${it.message}")
                    }
                },
                enabled = !isCapturing && hasFlashUnit
            ) {
                val flashLabel = when {
                    !hasFlashUnit -> "Flash Yok"
                    isFlashEnabled -> "Fener Acik"
                    else -> "Fener Kapali"
                }
                Text(flashLabel)
            }

            Button(
                onClick = {
                    val activeCapture = imageCapture
                    if (activeCapture == null) {
                        onError("Kamera henuz hazir degil. Biraz sonra tekrar deneyin.")
                        return@Button
                    }

                    activeCapture.flashMode = ImageCapture.FLASH_MODE_OFF

                    val photoFile = runCatching {
                        File.createTempFile("tomatech_cx_", ".jpg", context.cacheDir)
                    }.getOrElse {
                        onError("Gecici fotograf dosyasi olusturulamadi: ${it.message}")
                        return@Button
                    }

                    onCaptureStart()
                    val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
                    activeCapture.takePicture(
                        outputOptions,
                        mainExecutor,
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                                onCaptureEnd()
                                onPhotoCaptured(Uri.fromFile(photoFile))
                            }

                            override fun onError(exception: ImageCaptureException) {
                                onCaptureEnd()
                                onError("Kamera cekimi basarisiz: ${exception.message}")
                            }
                        }
                    )
                },
                enabled = !isCapturing,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (isCapturing) "Kaydediliyor..." else "Cek")
            }
        }
    }
}
