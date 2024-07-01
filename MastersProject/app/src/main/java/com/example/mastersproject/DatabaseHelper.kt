package com.example.mastersproject

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class DatabaseHelper(context: Context) : SQLiteOpenHelper(context, "glucose.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE IF NOT EXISTS GLUCOSE_READINGS(" +
                "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                "hour INTEGER NOT NULL," +
                "minute INTEGER NOT NULL," +
                "glucose_level REAL NOT NULL);")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Handle database schema changes if needed
    }
}