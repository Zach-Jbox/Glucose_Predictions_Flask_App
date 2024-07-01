package com.example.mastersproject

import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import java.io.IOException
import org.json.JSONObject

class XGBoostActivity : AppCompatActivity() {

    private lateinit var predictionTextView: TextView
    private lateinit var graphImageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_xgboost)

        predictionTextView = findViewById(R.id.prediction_text_view)
        graphImageView = findViewById(R.id.graph_image_view)

        fetchPrediction()
        fetchGraph()

        findViewById<Button>(R.id.back_button).setOnClickListener {
            finish()
        }
    }

    private fun fetchPrediction() {
        val url = "https://glucose.dynv6.net/predict_xgboost"

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
                        val prediction = json.getDouble("predicted_glucose").toInt()
                        runOnUiThread {
                            predictionTextView.text = "Predicted Blood Glucose Level: $prediction"
                        }
                    }
                }
            }
        })
    }

    private fun fetchGraph() {
        val url = "https://glucose.dynv6.net/graph/xgb"

        val request = Request.Builder().url(url).build()
        val client = OkHttpClient()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val inputStream = response.body?.byteStream()
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                    runOnUiThread {
                        graphImageView.setImageBitmap(bitmap)
                    }
                }
            }
        })
    }
}