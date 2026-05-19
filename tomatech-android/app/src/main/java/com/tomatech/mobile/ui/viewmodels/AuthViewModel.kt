package com.tomatech.mobile.ui.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.tomatech.mobile.data.AuthRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class AuthViewModel(application: Application) : AndroidViewModel(application) {
    private val repository = AuthRepository(application)
    
    private val _isLoggedIn = MutableStateFlow(repository.isUserLoggedIn())
    val isLoggedIn: StateFlow<Boolean> = _isLoggedIn

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading

    fun login(email: String, pass: String) {
        if (email.isBlank() || pass.isBlank()) {
            _errorMessage.value = "E-posta veya şifre boş olamaz."
            return
        }
        if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            _errorMessage.value = "Geçerli bir e-posta adresi giriniz."
            return
        }
        
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            kotlinx.coroutines.delay(800) // Simulate network delay for UX
            
            val success = repository.loginUser(email, pass)
            if (success) {
                _isLoggedIn.value = true
            } else {
                _errorMessage.value = "Hatalı e-posta veya şifre."
            }
            _isLoading.value = false
        }
    }

    fun register(name: String, email: String, pass: String) {
        if (name.isBlank() || email.isBlank() || pass.isBlank()) {
            _errorMessage.value = "Tüm alanları doldurmalısınız."
            return
        }
        if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            _errorMessage.value = "Geçerli bir e-posta adresi giriniz."
            return
        }
        if (pass.length < 6) {
            _errorMessage.value = "Şifre en az 6 karakter olmalıdır."
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            kotlinx.coroutines.delay(1000) // Simulate network delay for UX
            
            val success = repository.registerUser(email, name, pass)
            if (success) {
                // login directly
                val loginSuccess = repository.loginUser(email, pass)
                if(loginSuccess) _isLoggedIn.value = true
            } else {
                _errorMessage.value = "Bu e-posta adresiyle zaten kayıtlı bir hesap var."
            }
            _isLoading.value = false
        }
    }

    fun logout() {
        repository.logout()
        _isLoggedIn.value = false
    }

    fun clearError() {
        _errorMessage.value = null
    }
}
