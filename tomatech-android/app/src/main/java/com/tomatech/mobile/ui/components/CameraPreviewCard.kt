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
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
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
    captureTrigger: Boolean = false,
    onCaptureTriggerConsumed: () -> Unit = {},
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
                val selector = CameraSelector.Builder()
                    .requireLensFacing(activeLensFacing)
                    .build()

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
            runCatching { boundCamera?.cameraControl?.enableTorch(false) }
            boundCamera = null
            runCatching { cameraProviderFuture.get().unbindAll() }
        }
    }

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { previewView },
            update = { view ->
                view.setOnTouchListener { _, event ->
                    if (event.action == MotionEvent.ACTION_UP) {
                        val activeCamera = boundCamera
                        if (activeCamera != null) {
                            val point = view.meteringPointFactory.createPoint(event.x, event.y)
                            val action = FocusMeteringAction.Builder(
                                point,
                                FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
                            ).setAutoCancelDuration(3, TimeUnit.SECONDS).build()

                            runCatching { activeCamera.cameraControl.startFocusAndMetering(action) }
                        }
                    }
                    true
                }
            }
        )

        Column(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 100.dp, end = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            if (hasFlashUnit) {
                IconButton(
                    onClick = {
                        val activeCamera = boundCamera
                        if (activeCamera != null) {
                            val newValue = !isFlashEnabled
                            runCatching {
                                activeCamera.cameraControl.enableTorch(newValue)
                                isFlashEnabled = newValue
                            }
                        }
                    },
                    modifier = Modifier.background(Color.Black.copy(alpha = 0.4f), CircleShape)
                ) {
                    Icon(
                        imageVector = if (isFlashEnabled) Icons.Default.FlashOn else Icons.Default.FlashOff,
                        contentDescription = "Flaş",
                        tint = Color.White
                    )
                }
            }

            IconButton(
                onClick = {
                    runCatching { boundCamera?.cameraControl?.enableTorch(false) }
                    isFlashEnabled = false
                    lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                        CameraSelector.LENS_FACING_FRONT
                    } else {
                        CameraSelector.LENS_FACING_BACK
                    }
                },
                modifier = Modifier.background(Color.Black.copy(alpha = 0.4f), CircleShape)
            ) {
                Icon(
                    imageVector = Icons.Default.Cameraswitch,
                    contentDescription = "Kamera Çevir",
                    tint = Color.White
                )
            }
        }

        LaunchedEffect(captureTrigger) {
            if (captureTrigger) {
                onCaptureTriggerConsumed()
                if (!isCapturing && imageCapture != null) {
                    val activeCapture = imageCapture!!
                    activeCapture.flashMode = ImageCapture.FLASH_MODE_OFF

                    val photoFile = runCatching {
                        File.createTempFile("tomatech_cx_", ".jpg", context.cacheDir)
                    }.getOrElse { return@LaunchedEffect }

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
                                onError("Çekim başarısız")
                            }
                        }
                    )
                }
            }
        }
    }
}
