package com.tomatech.mobile.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowForward
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material.icons.rounded.*
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.automirrored.rounded.ArrowForward
import androidx.compose.material.icons.automirrored.outlined.Logout
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tomatech.mobile.ui.theme.*
import com.tomatech.mobile.ui.viewmodels.AuthViewModel
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

@Composable
fun MainScreen(
    authViewModel: AuthViewModel,
    tomatoViewModel: com.tomatech.mobile.TomatoViewModel,
    onNavigateToDiagnosis: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onLogout: () -> Unit
) {
    var selectedTab by rememberSaveable { mutableIntStateOf(0) }

    Scaffold(
        bottomBar = {
            Surface(
                modifier = Modifier
                    .padding(horizontal = 32.dp, vertical = 24.dp)
                    .shadow(elevation = 32.dp, shape = RoundedCornerShape(40.dp), spotColor = TomatoPrimary.copy(alpha = 0.6f)),
                shape = RoundedCornerShape(40.dp),
                color = Color.White.copy(alpha = 0.95f),
            ) {
                NavigationBar(
                    containerColor = Color.Transparent,
                    tonalElevation = 0.dp,
                    modifier = Modifier.height(72.dp)
                ) {
                    NavigationBarItem(
                        selected = selectedTab == 0,
                        onClick = { selectedTab = 0 },
                        alwaysShowLabel = false,
                        icon = { 
                            Icon(
                                if (selectedTab == 0) Icons.Rounded.Home else Icons.Outlined.Home, 
                                contentDescription = "Ana Sayfa", 
                                modifier = Modifier.size(26.dp)
                            ) 
                        },
                        label = { Text("Ana Sayfa", fontWeight = FontWeight.Bold, fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = TomatoPrimary,
                            indicatorColor = TomatoPrimary.copy(alpha = 0.12f),
                            unselectedIconColor = Color.Gray.copy(alpha = 0.5f),
                            unselectedTextColor = TomatoPrimary
                        )
                    )

                    NavigationBarItem(
                        selected = selectedTab == 1,
                        onClick = { selectedTab = 1 },
                        alwaysShowLabel = false,
                        icon = { 
                            Icon(
                                if (selectedTab == 1) Icons.Rounded.History else Icons.Outlined.History, 
                                contentDescription = "Geçmiş", 
                                modifier = Modifier.size(26.dp)
                            ) 
                        },
                        label = { Text("Geçmiş", fontWeight = FontWeight.Bold, fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = TomatoPrimary,
                            indicatorColor = TomatoPrimary.copy(alpha = 0.12f),
                            unselectedIconColor = Color.Gray.copy(alpha = 0.5f),
                            unselectedTextColor = TomatoPrimary
                        )
                    )
                    
                    // Central FAB Gap
                    Box(modifier = Modifier.weight(0.4f))
                    
                    NavigationBarItem(
                        selected = selectedTab == 2,
                        onClick = { selectedTab = 2 },
                        alwaysShowLabel = false,
                        icon = { 
                            Icon(
                                if (selectedTab == 2) Icons.Rounded.Analytics else Icons.Outlined.Analytics, 
                                contentDescription = "Analiz", 
                                modifier = Modifier.size(26.dp)
                            ) 
                        },
                        label = { Text("Analiz", fontWeight = FontWeight.Bold, fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = TomatoPrimary,
                            indicatorColor = TomatoPrimary.copy(alpha = 0.12f),
                            unselectedIconColor = Color.Gray.copy(alpha = 0.5f),
                            unselectedTextColor = TomatoPrimary
                        )
                    )

                    NavigationBarItem(
                        selected = selectedTab == 3,
                        onClick = { selectedTab = 3 },
                        alwaysShowLabel = false,
                        icon = { 
                            Icon(
                                if (selectedTab == 3) Icons.Rounded.Person else Icons.Outlined.Person, 
                                contentDescription = "Profil", 
                                modifier = Modifier.size(26.dp)
                            ) 
                        },
                        label = { Text("Profil", fontWeight = FontWeight.Bold, fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = TomatoPrimary,
                            indicatorColor = TomatoPrimary.copy(alpha = 0.12f),
                            unselectedIconColor = Color.Gray.copy(alpha = 0.5f),
                            unselectedTextColor = TomatoPrimary
                        )
                    )
                }
            }
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = onNavigateToDiagnosis,
                containerColor = Color.Transparent,
                elevation = FloatingActionButtonDefaults.elevation(0.dp),
                modifier = Modifier
                    .size(76.dp)
                    .offset(y = 52.dp)
                    .shadow(24.dp, CircleShape, spotColor = TomatoPrimary.copy(alpha = 0.8f))
                    .background(
                        Brush.linearGradient(
                            colors = listOf(TomatoPrimary, Color(0xFFFF6B6B))
                        ),
                        CircleShape
                    )
            ) {
                val infiniteTransition = rememberInfiniteTransition(label = "pulse")
                val scale by infiniteTransition.animateFloat(
                    initialValue = 1f,
                    targetValue = 1.15f,
                    animationSpec = infiniteRepeatable(
                        animation = tween(1200, easing = FastOutSlowInEasing),
                        repeatMode = RepeatMode.Reverse
                    ), label = "pulse_scale"
                )

                Box(contentAlignment = Alignment.Center) {
                    // Scanning Frame Brackets
                    Icon(
                        Icons.Rounded.CenterFocusWeak,
                        contentDescription = null,
                        modifier = Modifier.size(42.dp).scale(scale),
                        tint = Color.White.copy(alpha = 0.5f)
                    )
                    
                    // The Leaf
                    Icon(
                        Icons.Rounded.Eco,
                        contentDescription = "Tara", 
                        modifier = Modifier.size(28.dp).scale(scale), 
                        tint = Color.White
                    )
                }
            }
        },
        floatingActionButtonPosition = FabPosition.Center
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFFF7F9FC)) // Modern Soft Background
                .padding(innerPadding)
        ) {
            AnimatedContent(
                targetState = selectedTab,
                transitionSpec = {
                    (fadeIn(animationSpec = tween(300, delayMillis = 90)) +
                            slideInVertically(initialOffsetY = { 50 }))
                        .togetherWith(fadeOut(animationSpec = tween(90)))
                }, 
                label = "tab_transition"
            ) { targetTab ->
                when (targetTab) {
                    0 -> HomeDashboard(tomatoViewModel, onNavigateToHistory = { selectedTab = 1 })
                    1 -> HistoryDashboard(tomatoViewModel)
                    2 -> AnalyticsDashboard()
                    3 -> ProfileDashboard(authViewModel, onNavigateToSettings, onLogout)
                }
            }
        }
    }
}

@Composable
fun HistoryDashboard(viewModel: com.tomatech.mobile.TomatoViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val history = uiState.history

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp)
    ) {
        Spacer(modifier = Modifier.height(48.dp))
        Text(
            text = "Geçmiş",
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.Black,
            color = TextPrimary
        )
        Text(
            text = "Yaptığınız tüm analizler",
            style = MaterialTheme.typography.titleMedium,
            color = TextSecondary
        )
        
        Spacer(modifier = Modifier.height(24.dp))

        if (history.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Henüz geçmiş analiz yok.", color = TextSecondary)
            }
        } else {
            androidx.compose.foundation.lazy.LazyColumn(
                contentPadding = PaddingValues(bottom = 120.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                items(history.size) { index ->
                    val item = history[index]
                    CompactHistoryCard(item)
                }
            }
        }
    }
}

@Composable
fun AnalyticsDashboard() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Rounded.Analytics, 
            contentDescription = null, 
            modifier = Modifier.size(80.dp), 
            tint = TomatoPrimary.copy(alpha = 0.4f)
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "Analiz Verileri",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = TextPrimary
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Mahsul sağlığı ve istatistikleri çok yakında burada olacak.",
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyLarge,
            color = TextSecondary
        )
    }
}

@Composable
fun CompactHistoryCard(item: com.tomatech.mobile.data.HistoryItem) {
    val date = remember(item.timestamp) {
        SimpleDateFormat("dd MMM yyyy, HH:mm", Locale.getDefault()).format(Date(item.timestamp))
    }
    
    val labelLower = item.label.lowercase()
    val isHealthy = labelLower.contains("healthy")
    val isInvalid = labelLower.contains("geçersiz")
    
    val statusColor = when {
        isHealthy -> SuccessGreen
        isInvalid -> WarningYellow
        else -> ErrorRed
    }

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(12.dp, RoundedCornerShape(24.dp), spotColor = statusColor.copy(alpha = 0.1f)),
        shape = RoundedCornerShape(24.dp),
        color = Color.White
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(statusColor.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    if (isHealthy) Icons.Rounded.Eco else Icons.Rounded.Analytics,
                    contentDescription = null,
                    tint = statusColor,
                    modifier = Modifier.size(28.dp)
                )
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = item.label.replace("_", " "),
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary,
                    maxLines = 1
                )
                Text(
                    text = date,
                    style = MaterialTheme.typography.labelMedium,
                    color = TextSecondary
                )
            }
            
            Text(
                text = "%${(item.confidence * 100).toInt()}",
                fontWeight = FontWeight.Black,
                color = statusColor,
                fontSize = 18.sp
            )
        }
    }
}

@Composable
fun HomeDashboard(
    viewModel: com.tomatech.mobile.TomatoViewModel,
    onNavigateToHistory: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val scrollState = rememberScrollState()
    
    val currentHour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
    val greeting = when (currentHour) {
        in 5..11 -> "Günaydın ☀️"
        in 12..17 -> "İyi Günler 🌤️"
        in 18..21 -> "İyi Akşamlar 🌇"
        else -> "İyi Geceler 🌙"
    }

    // Health Score Animation
    var animationPlayed by remember { mutableStateOf(false) }
    val currentHealthScore by animateFloatAsState(
        targetValue = if (animationPlayed) 210f else 0f,
        animationSpec = tween(durationMillis = 1500, delayMillis = 300, easing = FastOutSlowInEasing),
        label = "health_gauge"
    )

    LaunchedEffect(Unit) {
        animationPlayed = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .background(Brush.verticalGradient(listOf(Color(0xFFF7F9FC), Color.White)))
    ) {
        // --- SECTION 1: GLASSMORPHIC HEADER ---
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 32.dp)
        ) {
            Column {
                Text(
                    text = greeting,
                    style = MaterialTheme.typography.labelLarge,
                    color = TomatoPrimary.copy(alpha = 0.8f),
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Bahçen Nasıl?",
                    style = MaterialTheme.typography.displaySmall.copy(
                        fontWeight = FontWeight.Black,
                        letterSpacing = (-1).sp,
                        lineHeight = 40.sp
                    ),
                    color = TextPrimary
                )
            }
            
            // Notification Badge
            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .size(48.dp)
                    .shadow(12.dp, CircleShape, spotColor = Color.Black.copy(alpha = 0.1f)),
                shape = CircleShape,
                color = Color.White
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(Icons.Outlined.Notifications, contentDescription = null, modifier = Modifier.size(24.dp), tint = TextPrimary)
                    Box(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(12.dp)
                            .size(10.dp)
                            .background(TomatoPrimary, CircleShape)
                            .border(1.5.dp, Color.White, CircleShape)
                    )
                }
            }
        }

        // --- SECTION 2: GARDEN HEALTH MONITOR (GAUGE) ---
        Surface(
            modifier = Modifier
                .padding(horizontal = 24.dp)
                .fillMaxWidth()
                .shadow(32.dp, RoundedCornerShape(32.dp), spotColor = TomatoPrimary.copy(alpha = 0.2f)),
            shape = RoundedCornerShape(32.dp),
            color = Color.White
        ) {
            Row(
                modifier = Modifier.padding(24.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Circular Progress Gauge
                Box(contentAlignment = Alignment.Center, modifier = Modifier.size(100.dp)) {
                    Canvas(modifier = Modifier.size(100.dp)) {
                        drawArc(
                            color = Color(0xFFF1F2F6),
                            startAngle = 140f,
                            sweepAngle = 260f,
                            useCenter = false,
                            style = Stroke(width = 12.dp.toPx(), cap = StrokeCap.Round)
                        )
                        drawArc(
                            brush = Brush.linearGradient(listOf(SuccessGreen, Color(0xFF55EFC4))),
                            startAngle = 140f,
                            sweepAngle = currentHealthScore, // Animated Health
                            useCenter = false,
                            style = Stroke(width = 12.dp.toPx(), cap = StrokeCap.Round)
                        )
                    }
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("%${(currentHealthScore / 260f * 100).toInt()}", fontWeight = FontWeight.Black, fontSize = 24.sp, color = TextPrimary)
                        Text("Sağlık", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                    }
                }

                Spacer(modifier = Modifier.width(24.dp))

                Column {
                    Text("Genel Durum: Harika", fontWeight = FontWeight.ExtraBold, fontSize = 18.sp, color = TextPrimary)
                    Text(
                        "Son 24 saat içinde herhangi bir hastalık belirtisi saptanmadı.",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                        lineHeight = 18.sp
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Rounded.TrendingUp, contentDescription = null, tint = SuccessGreen, modifier = Modifier.size(16.dp))
                        Text(" Geçen haftaya göre +%5", color = SuccessGreen, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // --- NEW SECTION: RECENT SCANS SUMMARY ---
        if (uiState.history.isNotEmpty()) {
            val latestItem = uiState.history.first()
            val labelLower = latestItem.label.lowercase()
            val isHealthy = labelLower.contains("healthy")
            val statusColor = if (isHealthy) SuccessGreen else if (labelLower.contains("geçersiz")) WarningYellow else ErrorRed

            Column(modifier = Modifier.padding(horizontal = 24.dp)) {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("Son Tarama", fontWeight = FontWeight.Black, fontSize = 18.sp, color = TextPrimary)
                    TextButton(onClick = onNavigateToHistory) {
                        Text("Tümünü Gör", color = TomatoPrimary, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    modifier = Modifier.fillMaxWidth().shadow(16.dp, RoundedCornerShape(24.dp), spotColor = statusColor.copy(alpha = 0.15f)),
                    shape = RoundedCornerShape(24.dp),
                    color = Color.White
                ) {
                    Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier.size(48.dp).background(statusColor.copy(alpha = 0.1f), RoundedCornerShape(14.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Rounded.History, contentDescription = null, tint = statusColor)
                        }
                        Spacer(modifier = Modifier.width(16.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(latestItem.label.replace("_", " "), fontWeight = FontWeight.Bold, color = TextPrimary, maxLines = 1)
                            Text("Güven: %${(latestItem.confidence * 100).toInt()}", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                        }
                        Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = Color.LightGray)
                    }
                }
            }
            Spacer(modifier = Modifier.height(32.dp))
        }

        // --- SECTION 3: AI SCANNER CTA ---
        Surface(
            modifier = Modifier
                .padding(horizontal = 24.dp)
                .fillMaxWidth()
                .shadow(24.dp, RoundedCornerShape(32.dp), spotColor = TomatoPrimary.copy(alpha = 0.5f)),
            shape = RoundedCornerShape(32.dp),
            color = TomatoPrimary
        ) {
            Box(modifier = Modifier.background(Brush.horizontalGradient(listOf(TomatoPrimary, Color(0xFFEE5253))))) {
                Icon(
                    Icons.Default.Eco,
                    contentDescription = null,
                    tint = Color.White.copy(alpha = 0.07f),
                    modifier = Modifier.size(200.dp).offset(x = 220.dp, y = (-40).dp)
                )

                Row(
                    modifier = Modifier.padding(28.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("AI Tarama Başlat", color = Color.White, fontWeight = FontWeight.Black, fontSize = 22.sp)
                        Text(
                            "Hastalıkları saniyeler içinde tespit etmek için kamerayı kullanın.",
                            color = Color.White.copy(alpha = 0.8f),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                    
                    Surface(
                        modifier = Modifier.size(56.dp),
                        shape = CircleShape,
                        color = Color.White.copy(alpha = 0.2f),
                        onClick = { }
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(Icons.AutoMirrored.Rounded.ArrowForward, contentDescription = null, tint = Color.White)
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // --- NEW SECTION 4: AI INSIGHT OF THE DAY ---
        Surface(
            modifier = Modifier
                .padding(horizontal = 24.dp)
                .fillMaxWidth()
                .shadow(16.dp, RoundedCornerShape(24.dp), spotColor = Color(0xFF6C5CE7).copy(alpha = 0.3f)),
            shape = RoundedCornerShape(24.dp),
            color = Color(0xFFF0EFFF)
        ) {
            Row(modifier = Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier.size(52.dp).background(Color(0xFF6C5CE7), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Rounded.Lightbulb, contentDescription = null, tint = Color.White, modifier = Modifier.size(24.dp))
                }
                Spacer(modifier = Modifier.width(16.dp))
                Column {
                    Text("Günün AI Tavsiyesi", fontWeight = FontWeight.Black, fontSize = 15.sp, color = Color(0xFF6C5CE7))
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        "Domatesleriniz bu hafta yüksek UV'den stres yapabilir. Sulamayı %10 artırmanızı öneririz.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.DarkGray,
                        lineHeight = 16.sp
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // --- SECTION 5: ENVIRONMENTAL MONITORING (IoT / Weather) ---
        Column(modifier = Modifier.padding(horizontal = 24.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Bottom
            ) {
                Column {
                    Text("Çevre İzleme", fontWeight = FontWeight.Black, fontSize = 20.sp, color = TextPrimary)
                    Text("Canlı Sensör Verileri", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                }
                Icon(Icons.Rounded.Sensors, contentDescription = "Live Sensors", tint = SuccessGreen)
            }
            Spacer(modifier = Modifier.height(20.dp))
            
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                MetricCard(
                    title = "Ortam Nemi",
                    value = "%64",
                    status = "Optimal",
                    icon = Icons.Default.WaterDrop,
                    color = InfoBlue,
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Sıcaklık",
                    value = "24°C",
                    status = "Normal",
                    icon = Icons.Default.Thermostat,
                    color = WarningYellow,
                    modifier = Modifier.weight(1f)
                )
            }
            Spacer(modifier = Modifier.height(16.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                MetricCard(
                    title = "UV Endeksi",
                    value = "4.2",
                    status = "Düşük",
                    icon = Icons.Default.WbSunny,
                    color = Color(0xFFFFA502),
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Toprak pH",
                    value = "6.5",
                    status = "Zengin",
                    icon = Icons.Default.Landscape,
                    color = Color(0xFF70A1FF),
                    modifier = Modifier.weight(1f)
                )
            }
        }

        Spacer(modifier = Modifier.height(120.dp))
    }
}

@Composable
fun MetricCard(
    title: String,
    value: String,
    status: String,
    icon: ImageVector,
    color: Color,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.shadow(8.dp, RoundedCornerShape(24.dp), spotColor = Color.Black.copy(alpha = 0.05f)),
        shape = RoundedCornerShape(24.dp),
        color = Color.White
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .background(color.copy(alpha = 0.1f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
                }
                Text(
                    text = status,
                    style = MaterialTheme.typography.labelSmall,
                    color = color,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(modifier = Modifier.height(16.dp))
            Text(value, fontWeight = FontWeight.Black, fontSize = 20.sp, color = TextPrimary)
            Text(title, style = MaterialTheme.typography.labelMedium, color = TextSecondary)
        }
    }
}

@Composable
fun DashboardActionCard(
    icon: ImageVector,
    title: String,
    subtitle: String,
    color: Color,
    modifier: Modifier = Modifier,
    iconSize: androidx.compose.ui.unit.Dp = 24.dp,
    onClick: () -> Unit = {}
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(targetValue = if (isPressed) 0.95f else 1f, label = "scale")

    Surface(
        onClick = onClick,
        interactionSource = interactionSource,
        modifier = modifier
            .scale(scale)
            .shadow(if (isPressed) 2.dp else 12.dp, RoundedCornerShape(28.dp), spotColor = color.copy(alpha = 0.2f)),
        shape = RoundedCornerShape(28.dp),
        color = Color.White,
        border = BorderStroke(1.dp, Color.Black.copy(alpha = 0.03f))
    ) {
        Column(modifier = Modifier.padding(24.dp)) {
            Box(
                modifier = Modifier
                    .size(52.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(color.copy(alpha = 0.12f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(iconSize))
            }
            Spacer(modifier = Modifier.height(20.dp))
            Text(title, fontWeight = FontWeight.ExtraBold, color = TextPrimary, fontSize = 17.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text(subtitle, style = MaterialTheme.typography.labelMedium, color = TextSecondary, lineHeight = 16.sp)
        }
    }
}

@Composable
fun ProfileDashboard(authViewModel: AuthViewModel, onNavigateToSettings: () -> Unit, onLogout: () -> Unit) {
    val scrollState = rememberScrollState()
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(48.dp))
        
        // Profile Avatar
        Box(
            modifier = Modifier
                .size(130.dp)
                .shadow(16.dp, CircleShape, spotColor = TomatoPrimary.copy(alpha = 0.5f))
                .background(Brush.linearGradient(listOf(TomatoPrimary, Color(0xFFFF6B6B))), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Text("Ü", color = Color.White, style = MaterialTheme.typography.displayMedium, fontWeight = FontWeight.Bold)
        }
        
        Spacer(modifier = Modifier.height(24.dp))
        
        Text(
            text = "Üretici Hesabı",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Black,
            color = TextPrimary
        )
        Text(
            text = "Tarımsal Veri Merkezi",
            style = MaterialTheme.typography.bodyMedium,
            color = TextSecondary
        )
        
        Spacer(modifier = Modifier.height(48.dp))
        
        Surface(
            onClick = onNavigateToSettings,
            modifier = Modifier.fillMaxWidth().shadow(8.dp, RoundedCornerShape(24.dp), spotColor = TomatoPrimary.copy(alpha = 0.2f)),
            shape = RoundedCornerShape(24.dp),
            color = Color.White
        ) {
            Row(
                modifier = Modifier.padding(24.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .background(Color(0xFFF7F9FC), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Rounded.Settings, contentDescription = "Ayarlar", tint = TextPrimary, modifier = Modifier.size(20.dp))
                }
                Spacer(modifier = Modifier.width(16.dp))
                Text(
                    text = "Hesap Ayarları",
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Surface(
            onClick = onLogout,
            modifier = Modifier.fillMaxWidth().shadow(8.dp, RoundedCornerShape(24.dp), spotColor = Color(0xFFFF7675).copy(alpha = 0.2f)),
            shape = RoundedCornerShape(24.dp),
            color = Color(0xFFFFF0F0)
        ) {
            Row(
                modifier = Modifier.padding(24.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .background(Color.White, CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Outlined.Logout, contentDescription = "Log Out", tint = Color(0xFFE53935), modifier = Modifier.size(20.dp))
                }
                Spacer(modifier = Modifier.width(16.dp))
                Text(
                    text = "Güvenli Çıkış Yap",
                    color = Color(0xFFE53935),
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                )
            }
        }
        
        Spacer(modifier = Modifier.height(120.dp)) // padding for bottom bar
    }
}

@Composable
fun SettingsRow(icon: ImageVector, title: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { }
            .padding(24.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(40.dp)
                .background(Color.Black.copy(alpha = 0.04f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(icon, contentDescription = null, tint = TextPrimary, modifier = Modifier.size(20.dp))
        }
        Spacer(modifier = Modifier.width(16.dp))
        Text(text = title, color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 16.sp, modifier = Modifier.weight(1f))
        Text(text = "〉", color = TextSecondary, fontSize = 16.sp, fontWeight = FontWeight.Bold) // Simple arrow
    }
}
