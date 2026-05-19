package com.tomatech.mobile.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Eco
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.tomatech.mobile.TomatoViewModel
import com.tomatech.mobile.ui.components.CameraPreviewCard
import com.tomatech.mobile.ui.components.DiagnosisActionButton
import com.tomatech.mobile.ui.components.DiagnosisResultCard
import com.tomatech.mobile.ui.components.ErrorMessageCard
import com.tomatech.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TomatoDiagnosisScreen(viewModel: TomatoViewModel, onBack: () -> Unit = {}) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()
    val bitmap = uiState.selectedBitmap

    var isCameraPreviewOpen by remember { mutableStateOf(true) }
    var isCapturingWithCamera by remember { mutableStateOf(false) }
    var capturePhotoTrigger by remember { mutableStateOf(false) }
    
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    var showResultSheet by remember { mutableStateOf(false) }

    LaunchedEffect(uiState.result) {
        if (uiState.result != null) {
            showResultSheet = true
        }
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            decodeBitmap(context, uri)
                .onSuccess {
                    isCameraPreviewOpen = false
                    viewModel.onImageSelected(it)
                }
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            isCameraPreviewOpen = true
        }
    }
    
    LaunchedEffect(Unit) {
        if (!hasCameraPermission(context)) {
            isCameraPreviewOpen = false
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        // Core Viewport (Camera or Image)
        if (isCameraPreviewOpen && hasCameraPermission(context)) {
            Box(Modifier.fillMaxSize()) {
                CameraPreviewCard(
                    modifier = Modifier.fillMaxSize(),
                    isCapturing = isCapturingWithCamera,
                    onCaptureStart = { isCapturingWithCamera = true },
                    onCaptureEnd = { isCapturingWithCamera = false },
                    onPhotoCaptured = { capturedUri ->
                        decodeBitmap(context, capturedUri)
                            .onSuccess {
                                isCameraPreviewOpen = false
                                viewModel.onImageSelected(it)
                            }
                    },
                    onError = { viewModel.setError(it) },
                    onClose = { isCameraPreviewOpen = false },
                    captureTrigger = capturePhotoTrigger,
                    onCaptureTriggerConsumed = { capturePhotoTrigger = false }
                )
                
                // Futuristic AR Scanner Overlay
                TargetingOverlay()
            }
        } else if (bitmap != null) {
            Box(Modifier.fillMaxSize()) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Seçilen Fotoğraf",
                    contentScale = ContentScale.Crop, // Crop instead of Fit for full-bleed immersion
                    modifier = Modifier.fillMaxSize()
                )
                
                // Scanning Animation over the frozen image
                if (uiState.isRunning) {
                    AiScanningAnimationOverlay()
                } else {
                    Box(modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.4f))) // Focus dims
                }
            }
        }

        // Top Navigation Bar (Glassmorphic)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color.Black.copy(alpha = 0.8f), Color.Transparent)
                    )
                )
                .statusBarsPadding()
                .padding(horizontal = 16.dp, vertical = 16.dp)
        ) {
            IconButton(onClick = onBack, modifier = Modifier.align(Alignment.CenterStart)) {
                Icon(Icons.Default.ArrowBack, contentDescription = "Geri", tint = Color.White)
            }
            
            Column(
                modifier = Modifier.align(Alignment.Center), 
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Surface(
                    color = Color.White.copy(alpha = 0.15f),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, Color.White.copy(alpha = 0.2f))
                ) {
                    Text(
                        text = if (isCameraPreviewOpen) "KAMERA AKTİF" else "ANALİZ BEKLENİYOR",
                        color = Color.White,
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 2.sp,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
                    )
                }
            }

            if (bitmap != null && !isCameraPreviewOpen && !uiState.isRunning) {
                IconButton(
                    onClick = { viewModel.clearImage() },
                    modifier = Modifier.align(Alignment.CenterEnd)
                ) {
                    Icon(Icons.Outlined.Close, contentDescription = "İptal Et", tint = Color.White)
                }
            }
        }
        
        // Dynamic Error Display
        AnimatedVisibility(
            visible = uiState.errorMessage != null,
            enter = slideInVertically(initialOffsetY = { -it }) + fadeIn(),
            exit = slideOutVertically() + fadeOut(),
            modifier = Modifier.align(Alignment.TopCenter).padding(top = 100.dp)
        ) {
            uiState.errorMessage?.let { error ->
                ErrorMessageCard(message = error)
            }
        }

        // Bottom Dashboard Controls
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.95f))
                    )
                )
                .navigationBarsPadding()
                .padding(bottom = 32.dp, top = 64.dp, start = 32.dp, end = 32.dp)
        ) {
            AnimatedContent(
                targetState = Pair(isCameraPreviewOpen, bitmap != null),
                transitionSpec = { fadeIn(tween(300)) togetherWith fadeOut(tween(300)) }
            ) { state ->
                val (isCam, hasImage) = state
                
                if (hasImage && !isCam) {
                    // Start Diagnosis UI
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        AnimatedVisibility(!uiState.isRunning) {
                            Text(
                                "Fotoğraf Hazır. Analizi Başlatın.",
                                color = Color.White.copy(alpha = 0.8f),
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.padding(bottom = 16.dp)
                            )
                        }
                        
                        DiagnosisActionButton(
                            enabled = !uiState.isRunning,
                            isRunning = uiState.isRunning,
                            onClick = { viewModel.runDiagnosis() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(64.dp)
                                .shadow(16.dp, RoundedCornerShape(20.dp), spotColor = TomatoPrimary)
                        )
                    }
                } else {
                    // Camera / Gallery Options
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        ControlSmallButton(
                            icon = Icons.Default.PhotoLibrary,
                            label = "Galeri",
                            onClick = { galleryLauncher.launch("image/*") }
                        )
                        
                        // Unified Capture UI
                        Surface(
                            onClick = { 
                                if (!isCapturingWithCamera) {
                                    capturePhotoTrigger = true 
                                }
                            },
                            shape = CircleShape,
                            color = Color.Transparent,
                            border = BorderStroke(4.dp, Color.White),
                            modifier = Modifier.size(80.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(6.dp)
                                    .background(if (isCapturingWithCamera) Color.Gray else Color.White, CircleShape)
                            )
                        }
                        
                        ControlSmallButton(
                            icon = Icons.Default.Camera,
                            label = "Yenile",
                            onClick = { 
                                if(hasCameraPermission(context)) isCameraPreviewOpen = true
                                else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                            }
                        )
                    }
                }
            }
        }
    }

    // Modal Results 
    if (showResultSheet && uiState.result != null) {
        ModalBottomSheet(
            onDismissRequest = { showResultSheet = false },
            sheetState = sheetState,
            containerColor = SurfaceLight,
            dragHandle = { BottomSheetDefaults.DragHandle(color = Color.Gray.copy(alpha = 0.3f)) },
            shape = RoundedCornerShape(topStart = 40.dp, topEnd = 40.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 24.dp)
            ) {
                // Animated Entrance for Results
                DiagnosisResultCard(
                    result = uiState.result!!,
                    decision = uiState.decision
                )
                
                Spacer(modifier = Modifier.height(32.dp))
                
                Button(
                    onClick = { showResultSheet = false },
                    modifier = Modifier.fillMaxWidth().height(60.dp),
                    shape = RoundedCornerShape(20.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = TomatoPrimary)
                ) {
                    Text("Kaydet ve Kapat", fontWeight = FontWeight.Bold, fontSize = 16.sp)
                }
                Spacer(modifier = Modifier.height(48.dp))
            }
        }
    }
}

// ----------------------------------------------------
// UI COMPONENTS
// ----------------------------------------------------

@Composable
fun TargetingOverlay() {
    val infiniteTransition = rememberInfiniteTransition(label = "leaf_pulse")
    val alphaAnim by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.8f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ), label = "leaf_alpha"
    )
    val scaleAnim by infiniteTransition.animateFloat(
        initialValue = 0.95f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ), label = "leaf_scale"
    )

    // Draws an AR-like targeting box
    Box(
        modifier = Modifier
            .fillMaxSize()
            .drawBehind {
                val boxWidth = size.width * 0.7f
                val boxHeight = boxWidth * 1.2f
                val topLeft = Offset((size.width - boxWidth) / 2, (size.height - boxHeight) / 2)
                val cornerRadius = 40f
                val strokeWidth = 8f

                // Draw corners
                val targetColor = Color.White.copy(alpha = 0.5f)

                drawRoundRect(
                    color = targetColor,
                    topLeft = topLeft,
                    size = Size(boxWidth, boxHeight),
                    cornerRadius = CornerRadius(cornerRadius),
                    style = Stroke(width = strokeWidth, pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(40f, 40f)))
                )
            }
    ) {
        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Elegant pulsing transparent leaf icon in the center
            Icon(
                imageVector = Icons.Default.Eco,
                contentDescription = null,
                tint = Color.White.copy(alpha = alphaAnim),
                modifier = Modifier
                    .size(64.dp)
                    .scale(scaleAnim)
            )
            
            Spacer(modifier = Modifier.height(140.dp))
            Text("Yaprağı Çerçeveye Hizalayın", color = Color.White.copy(alpha = 0.8f), fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun AiScanningAnimationOverlay() {
    val infiniteTransition = rememberInfiniteTransition()
    val scanLineY by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        )
    )

    Box(modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.5f))) {
        // Scanning laser
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .align(Alignment.TopCenter)
                .offset(y = (200 + (scanLineY * 400)).dp) // Hardcoded range for visual effect
                .shadow(12.dp, spotColor = TomatoPrimary)
                .background(TomatoPrimary)
        )
        
        Text(
            text = "YAPAY ZEKA ANALİZ EDİYOR...",
            color = TomatoPrimary,
            fontWeight = FontWeight.Black,
            letterSpacing = 2.sp,
            modifier = Modifier.align(Alignment.Center).background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(8.dp)).padding(12.dp)
        )
    }
}

@Composable
fun ControlSmallButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Surface(
            onClick = onClick,
            color = Color.White.copy(alpha = 0.15f),
            shape = CircleShape,
            modifier = Modifier.size(56.dp),
            border = BorderStroke(1.dp, Color.White.copy(alpha = 0.2f))
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(icon, contentDescription = null, tint = Color.White)
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Text(label, color = Color.White, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Medium)
    }
}

// ----------------------------------------------------
// UTILS
// ----------------------------------------------------

private fun hasCameraPermission(context: Context): Boolean {
    return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
}

private fun decodeBitmap(context: Context, uri: Uri): Result<Bitmap> {
    return runCatching {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            android.provider.MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }
    }
}
