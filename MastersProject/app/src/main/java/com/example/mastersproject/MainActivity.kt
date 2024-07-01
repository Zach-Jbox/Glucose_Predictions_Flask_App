package com.example.mastersproject

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import org.json.JSONObject
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var currentGlucoseTextView: TextView
    private lateinit var statusTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        currentGlucoseTextView = findViewById(R.id.current_glucose)
        statusTextView = findViewById(R.id.status)

        fetchCurrentGlucose()

        findViewById<Button>(R.id.random_forest_button).setOnClickListener {
            startActivity(Intent(this, RandomForestActivity::class.java))
        }

        findViewById<Button>(R.id.xgboost_button).setOnClickListener {
            startActivity(Intent(this, XGBoostActivity::class.java))
        }

        findViewById<Button>(R.id.lstm_button).setOnClickListener {
            startActivity(Intent(this, LSTMActivity::class.java))
        }
    }

    private fun fetchCurrentGlucose() {
        val url = "https://glucose.dynv6.net/current_glucose"

        val request = Request.Builder().url(url).build()
        val client = OkHttpClient()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val responseData = response.body?.string()
                    if (responseData != null) {
                        val json = JSONObject(responseData)
                        val glucoseLevel = json.getDouble("glucose_level").toInt()
                        val status = json.getString("status")

                        runOnUiThread {
                            currentGlucoseTextView.text = "Current Glucose: $glucoseLevel"
                            statusTextView.text = "Status: $status"

                            when (status) {
                                "High" -> statusTextView.setTextColor(Color.parseColor("#FFA500"))
                                "Normal" -> statusTextView.setTextColor(Color.parseColor("#008000"))
                                "Low" -> statusTextView.setTextColor(Color.parseColor("#FF0000"))
                            }
                        }
                    }
                }
            }
        })
    }
}