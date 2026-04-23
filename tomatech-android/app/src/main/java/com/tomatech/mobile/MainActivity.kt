package com.tomatech.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.tomatech.mobile.ui.screens.TomatoDiagnosisScreen
import com.tomatech.mobile.ui.theme.TomaTechTheme

class MainActivity : ComponentActivity() {

    private val viewModel: TomatoViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            TomaTechTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    TomatoDiagnosisScreen(viewModel = viewModel)
                }
            }
        }
    }
}
