package com.tomatech.mobile.data

import android.content.Context
import android.content.SharedPreferences

class AuthRepository(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("auth_prefs", Context.MODE_PRIVATE)

    fun isUserLoggedIn(): Boolean {
        return prefs.getBoolean("is_logged_in", false)
    }

    fun setLoggedIn(status: Boolean) {
        prefs.edit().putBoolean("is_logged_in", status).apply()
    }

    fun registerUser(email: String, name: String, pass: String): Boolean {
        if (prefs.contains("user_$email")) return false // Zaten var
        prefs.edit().putString("user_$email", pass).putString("name_$email", name).apply()
        return true
    }

    fun loginUser(email: String, pass: String): Boolean {
        val savedPass = prefs.getString("user_$email", null)
        val success = savedPass != null && savedPass == pass
        if (success) {
            setLoggedIn(true)
            prefs.edit().putString("current_user", email).apply()
        }
        return success
    }

    fun logout() {
        setLoggedIn(false)
        prefs.edit().remove("current_user").apply()
    }
}
