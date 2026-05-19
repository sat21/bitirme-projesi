package com.tomatech.mobile.ui.screens

import android.widget.Toast
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.HelpOutline
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tomatech.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val scrollState = rememberScrollState()
    val context = LocalContext.current

    var showAnimations by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        showAnimations = true
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Ayarlar", fontWeight = FontWeight.Black, fontSize = 24.sp) },
                navigationIcon = {
                    IconButton(onClick = onBack, modifier = Modifier.padding(start = 8.dp)) {
                        Surface(
                            shape = CircleShape,
                            color = Color.White,
                            shadowElevation = 4.dp,
                            modifier = Modifier.size(40.dp)
                        ) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Geri", modifier = Modifier.padding(8.dp), tint = TextPrimary)
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFFF7F9FC),
                    titleContentColor = TextPrimary
                )
            )
        },
        containerColor = Color(0xFFF7F9FC)
    ) { padding ->
        AnimatedVisibility(
            visible = showAnimations,
            enter = fadeIn(tween(400)) + slideInVertically(initialOffsetY = { 50 }, animationSpec = tween(400, easing = FastOutSlowInEasing))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(padding)
                    .padding(horizontal = 24.dp)
            ) {
                Spacer(modifier = Modifier.height(16.dp))

                // Kullanıcı Profili Mini Bilgi
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .shadow(16.dp, RoundedCornerShape(24.dp), spotColor = TomatoPrimary.copy(alpha = 0.2f)),
                    shape = RoundedCornerShape(24.dp),
                    color = Color.White
                ) {
                    Row(
                        modifier = Modifier.padding(20.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(64.dp)
                                .clip(CircleShape)
                                .background(Brush.linearGradient(listOf(TomatoPrimary, Color(0xFFFF6B6B)))),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Rounded.Person, contentDescription = null, tint = Color.White, modifier = Modifier.size(32.dp))
                        }
                        Spacer(modifier = Modifier.width(16.dp))
                        Column {
                            Text("Premium Hesap", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Black, color = TextPrimary)
                            Text("Tüm YZ servislerine erişiminiz var", style = MaterialTheme.typography.labelMedium, color = TextSecondary)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(32.dp))

                SettingsSectionTitle("GENEL")
                SettingsGroup {
                    var darkThemeEnabled by remember { mutableStateOf(false) }
                    var notificationsEnabled by remember { mutableStateOf(true) }

                    SettingsToggleCard(
                        icon = Icons.Rounded.DarkMode,
                        title = "Karanlık Mod",
                        subtitle = "Sistem teması kullanılacaktır",
                        isChecked = darkThemeEnabled,
                        onCheckedChange = { 
                            darkThemeEnabled = it 
                            Toast.makeText(context, if(it) "Karanlık Mod aktif edildi" else "Aydınlık Moda geçildi", Toast.LENGTH_SHORT).show()
                        }
                    )
                    HorizontalDivider(color = Color.Gray.copy(alpha = 0.1f), modifier = Modifier.padding(start = 64.dp))
                    SettingsToggleCard(
                        icon = Icons.Rounded.NotificationsActive,
                        title = "Bildirimler",
                        subtitle = "Hastalık alarmı ve ipuçları",
                        iconBackground = Color(0xFF4CAF50).copy(alpha = 0.15f),
                        iconTint = Color(0xFF4CAF50),
                        isChecked = notificationsEnabled,
                        onCheckedChange = { 
                            notificationsEnabled = it 
                            Toast.makeText(context, if(it) "Bildirimler açıldı" else "Bildirimler kapatıldı", Toast.LENGTH_SHORT).show()
                        }
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                SettingsSectionTitle("SİSTEM TERCİHLERİ")
                SettingsGroup {
                    SettingsNavigationCard(
                        icon = Icons.Rounded.Language,
                        title = "Dil",
                        subtitle = "Türkçe (TR)",
                        iconBackground = Color(0xFF2196F3).copy(alpha = 0.15f),
                        iconTint = Color(0xFF2196F3),
                        onClick = { Toast.makeText(context, "Mevcut dil Türkçe olarak kilitlidir.", Toast.LENGTH_SHORT).show() }
                    )
                    HorizontalDivider(color = Color.Gray.copy(alpha = 0.1f), modifier = Modifier.padding(start = 64.dp))
                    SettingsNavigationCard(
                        icon = Icons.Rounded.Storage,
                        title = "Önbelleği Temizle",
                        subtitle = "Uygulamanın geçici dosyalarını siler",
                        iconBackground = Color(0xFFFF9800).copy(alpha = 0.15f),
                        iconTint = Color(0xFFFF9800),
                        onClick = {
                            try {
                                context.cacheDir.deleteRecursively()
                                Toast.makeText(context, "Sistem önbelleği başarıyla temizlendi!", Toast.LENGTH_SHORT).show()
                            } catch (e: Exception) {
                                Toast.makeText(context, "Temizleme hatası oluştu.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                SettingsSectionTitle("DESTEK & HAKKINDA")
                SettingsGroup {
                    SettingsNavigationCard(
                        icon = Icons.AutoMirrored.Rounded.HelpOutline,
                        title = "Yardım Merkezi",
                        subtitle = "Sık sorulan sorular ve destek",
                        iconBackground = Color(0xFF9C27B0).copy(alpha = 0.15f),
                        iconTint = Color(0xFF9C27B0),
                        onClick = { Toast.makeText(context, "Yardım merkezine bağlanılıyor...", Toast.LENGTH_SHORT).show() }
                    )
                    HorizontalDivider(color = Color.Gray.copy(alpha = 0.1f), modifier = Modifier.padding(start = 64.dp))
                    SettingsNavigationCard(
                        icon = Icons.Rounded.Info,
                        title = "Uygulama Hakkında",
                        subtitle = "Versiyon 1.0.0 (BETA)",
                        iconBackground = Color(0xFF607D8B).copy(alpha = 0.15f),
                        iconTint = Color(0xFF607D8B),
                        showArrow = false,
                        onClick = { Toast.makeText(context, "TomaTech AI Labs © 2026", Toast.LENGTH_SHORT).show() }
                    )
                }

                Spacer(modifier = Modifier.height(48.dp))
                
                // Footer
                Text(
                    text = "TomaTech AI Labs",
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                    style = MaterialTheme.typography.labelMedium,
                    color = Color.Gray.copy(alpha = 0.5f),
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp
                )
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
}

@Composable
fun SettingsSectionTitle(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.ExtraBold,
        color = TomatoPrimary,
        letterSpacing = 1.5.sp,
        modifier = Modifier.padding(start = 16.dp, bottom = 12.dp)
    )
}

@Composable
fun SettingsGroup(content: @Composable ColumnScope.() -> Unit) {
    Surface(
        modifier = Modifier.fillMaxWidth().shadow(12.dp, RoundedCornerShape(24.dp), spotColor = Color.Gray.copy(alpha = 0.1f)),
        shape = RoundedCornerShape(24.dp),
        color = Color.White
    ) {
        Column {
            content()
        }
    }
}

@Composable
fun SettingsToggleCard(
    icon: ImageVector,
    title: String,
    subtitle: String,
    isChecked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    iconBackground: Color = TomatoPrimary.copy(alpha = 0.15f),
    iconTint: Color = TomatoPrimary
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onCheckedChange(!isChecked) }
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier.size(48.dp).clip(CircleShape).background(iconBackground),
            contentAlignment = Alignment.Center
        ) {
            Icon(icon, contentDescription = null, tint = iconTint)
        }
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = TextPrimary)
            Text(text = subtitle, style = MaterialTheme.typography.labelMedium, color = TextSecondary)
        }
        Switch(
            checked = isChecked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = TomatoPrimary, uncheckedThumbColor = Color.White, uncheckedTrackColor = Color.LightGray)
        )
    }
}

@Composable
fun SettingsNavigationCard(
    icon: ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit = {},
    iconBackground: Color = TomatoPrimary.copy(alpha = 0.15f),
    iconTint: Color = TomatoPrimary,
    showArrow: Boolean = true
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier.size(48.dp).clip(CircleShape).background(iconBackground),
            contentAlignment = Alignment.Center
        ) {
            Icon(icon, contentDescription = null, tint = iconTint)
        }
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = TextPrimary)
            Text(text = subtitle, style = MaterialTheme.typography.labelMedium, color = TextSecondary)
        }
        if (showArrow) {
            Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = Color.LightGray)
        }
    }
}
