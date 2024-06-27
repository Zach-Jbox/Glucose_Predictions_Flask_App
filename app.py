from flask import Flask, jsonify
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import xgboost as xgb
from datetime import datetime
import time
import threading
import numpy as np
from pydexcom import Dexcom

app = Flask(__name__)

# Connect to the database and create the GLUCOSE_READINGS table
conn = sqlite3.connect('glucose.db')
cursor = conn.cursor()

glucose_readings = """CREATE TABLE IF NOT EXISTS GLUCOSE_READINGS(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    glucose_level REAL NOT NULL
);"""

cursor.execute(glucose_readings)

conn.commit()
conn.close()

def add_glucose_reading(glucose_value):
    conn = sqlite3.connect('glucose.db')
    cursor = conn.cursor()

    now = datetime.now()
    hour = now.hour
    minute = now.minute

    cursor.execute("INSERT INTO GLUCOSE_READINGS (hour, minute, glucose_level) VALUES (?, ?, ?)",
                   (hour, minute, glucose_value))
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM GLUCOSE_READINGS")
    total_entries = cursor.fetchone()[0]

    if total_entries > 288:
        cursor.execute("DELETE FROM GLUCOSE_READINGS WHERE id IN (SELECT id FROM GLUCOSE_READINGS ORDER BY id LIMIT ?)", 
                       (total_entries - 288,))
        conn.commit()

    conn.close()

def update_glucose_readings():
    dexcom = Dexcom("Username", "Password")
    while True:
        glucose_reading = dexcom.get_current_glucose_reading()
        add_glucose_reading(glucose_reading.value)
        print(f"Reading added: {glucose_reading.value} at {datetime.now()}")
        time.sleep(300)

# Start the Dexcom update in a separate thread
update_thread = threading.Thread(target=update_glucose_readings)
update_thread.start()

# Define the route for Random Forest prediction
@app.route('/predict_random_forest', methods=['GET'])
def predict_random_forest():
    conn = sqlite3.connect('glucose.db')
    query = "SELECT * FROM GLUCOSE_READINGS"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['hour'] = df['hour'].astype(int)
    df['minute'] = df['minute'].astype(int)
    df['glucose_level'] = df['glucose_level'].astype(float)

    next_hour = df['hour'].iloc[-1]
    next_minute = df['minute'].iloc[-1] + 30

    if next_minute >= 60:
        next_hour += 1
        next_minute -= 60

    next_data_point = pd.DataFrame([[next_hour, next_minute]], columns=['hour', 'minute'])

    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(df[['hour', 'minute']], df['glucose_level'])

    next_prediction = rf_model.predict(next_data_point)
    predicted_glucose = int(round(next_prediction[0]))

    return jsonify({'predicted_glucose': predicted_glucose})

# Define the route for XGBoost prediction
@app.route('/predict_xgboost', methods=['GET'])
def predict_xgboost():
    conn = sqlite3.connect('glucose.db')
    query = "SELECT * FROM GLUCOSE_READINGS"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['hour'] = df['hour'].astype(int)
    df['minute'] = df['minute'].astype(int)
    df['glucose_level'] = df['glucose_level'].astype(float)

    X = df[['hour', 'minute']]
    y = df['glucose_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    next_hour = df['hour'].iloc[-1]
    next_minute = df['minute'].iloc[-1] + 30

    if next_minute >= 60:
        next_hour += 1
        next_minute -= 60

    next_data_point = pd.DataFrame([[next_hour, next_minute]], columns=['hour', 'minute'])

    next_prediction = model.predict(next_data_point)
    predicted_glucose = int(round(next_prediction[0]))

    return jsonify({'predicted_glucose': predicted_glucose})

# Define the route for LSTM prediction
@app.route('/predict_lstm', methods=['GET'])
def predict_lstm():
    # Connect to the database
    conn = sqlite3.connect('glucose.db')
    query = "SELECT * FROM GLUCOSE_READINGS"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Data preprocessing
    df['hour'] = df['hour'].astype(int)
    df['minute'] = df['minute'].astype(int)
    df['glucose_level'] = df['glucose_level'].astype(float)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['glucose_level']])

    # Define the create_dataset function
    def create_dataset(X, y, time_steps=1, future_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps - future_steps + 1):
            v = X[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps + future_steps - 1])
        return np.array(Xs), np.array(ys)

    # Define time steps and future steps
    time_steps = 10
    future_steps_30min = 6
    future_steps_2hours = 24

    # Create datasets for 30 minutes and 2 hours
    X_30min, y_30min = create_dataset(pd.DataFrame(data_scaled), pd.DataFrame(data_scaled), time_steps, future_steps_30min)
    X_2hours, y_2hours = create_dataset(pd.DataFrame(data_scaled), pd.DataFrame(data_scaled), time_steps, future_steps_2hours)

    # Split data into training and testing sets
    X_train_30min, X_test_30min, y_train_30min, y_test_30min = train_test_split(X_30min, y_30min, test_size=0.2, shuffle=False)
    X_train_2hours, X_test_2hours, y_train_2hours, y_test_2hours = train_test_split(X_2hours, y_2hours, test_size=0.2, shuffle=False)

    # Create and train LSTM models for 30 minutes and 2 hours
    model_30min = Sequential()
    model_30min.add(LSTM(units=64, input_shape=(X_train_30min.shape[1], X_train_30min.shape[2])))
    model_30min.add(Dense(units=1))
    model_30min.compile(optimizer='adam', loss='mean_squared_error')
    model_30min.fit(X_train_30min, y_train_30min, epochs=10, batch_size=32, validation_split=0.2)

    model_2hours = Sequential()
    model_2hours.add(LSTM(units=64, input_shape=(X_train_2hours.shape[1], X_train_2hours.shape[2])))
    model_2hours.add(Dense(units=1))
    model_2hours.compile(optimizer='adam', loss='mean_squared_error')
    model_2hours.fit(X_train_2hours, y_train_2hours, epochs=100, batch_size=32, validation_split=0.2)

    # Make predictions for 30 minutes and 2 hours
    X_pred_30min = np.array([data_scaled[-time_steps:]])
    y_pred_30min = model_30min.predict(X_pred_30min)
    y_pred_rescaled_30min = scaler.inverse_transform(y_pred_30min.reshape(-1, 1))
    y_pred_30min_rounded = int(round(y_pred_rescaled_30min[0][0]))

    X_pred_2hours = np.array([data_scaled[-time_steps:]])
    y_pred_2hours = model_2hours.predict(X_pred_2hours)
    y_pred_rescaled_2hours = scaler.inverse_transform(y_pred_2hours.reshape(-1, 1))
    y_pred_2hours_rounded = int(round(y_pred_rescaled_2hours[0][0]))

    # Return the predictions as JSON
    return jsonify({
        'predicted_glucose_30min': y_pred_30min_rounded,
        'predicted_glucose_2hours': y_pred_2hours_rounded
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)