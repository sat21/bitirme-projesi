package com.tomatech.mobile

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.tomatech.mobile.ui.screens.AuthScreen
import com.tomatech.mobile.ui.screens.HistoryScreen
import com.tomatech.mobile.ui.screens.MainScreen
import com.tomatech.mobile.ui.screens.SettingsScreen
import com.tomatech.mobile.ui.screens.SplashScreen
import com.tomatech.mobile.ui.screens.TomatoDiagnosisScreen
import com.tomatech.mobile.ui.theme.TomatechMobileTheme
import com.tomatech.mobile.ui.viewmodels.AuthViewModel

class MainActivity : ComponentActivity() {

    private val authViewModel: AuthViewModel by viewModels()
    private val diagnosisViewModel: TomatoViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        
        setContent {
            TomatechMobileTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    val navController = rememberNavController()
                    val isLoggedIn by authViewModel.isLoggedIn.collectAsState()

                    NavHost(
                        navController = navController,
                        startDestination = "splash"
                    ) {
                        composable("splash") {
                            SplashScreen(
                                isLoggedIn = isLoggedIn
                            ) { destination ->
                                navController.navigate(destination) {
                                    popUpTo("splash") { inclusive = true }
                                }
                            }
                        }

                        composable("auth") {
                            AuthScreen(
                                viewModel = authViewModel
                            ) {
                                navController.navigate("main") {
                                    popUpTo("auth") { inclusive = true }
                                }
                            }
                        }

                        composable("main") {
                            MainScreen(
                                authViewModel = authViewModel,
                                tomatoViewModel = diagnosisViewModel,
                                onNavigateToDiagnosis = {
                                    navController.navigate("diagnosis")
                                },
                                onNavigateToSettings = {
                                    navController.navigate("settings")
                                },
                                onLogout = {
                                    navController.navigate("auth") {
                                        popUpTo("main") { inclusive = true }
                                    }
                                }
                            )
                        }

                        composable("diagnosis") {
                            TomatoDiagnosisScreen(
                                viewModel = diagnosisViewModel,
                                onBack = {
                                    navController.popBackStack()
                                }
                            )
                        }

                        composable("history") {
                            HistoryScreen(
                                viewModel = diagnosisViewModel,
                                onBack = {
                                    navController.popBackStack()
                                }
                            )
                        }

                        composable("settings") {
                            SettingsScreen(
                                onBack = {
                                    navController.popBackStack()
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}
