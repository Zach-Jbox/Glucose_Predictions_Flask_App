from flask import Flask, jsonify, send_file
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import xgboost as xgb
from datetime import datetime, timedelta
import time
import threading
import numpy as np
import os
import joblib
from pydexcom import Dexcom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Database initialization and management functions
def init_db():
    conn = sqlite3.connect('glucose.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS GLUCOSE_READINGS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            glucose_level REAL NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RF_PREDICTIONS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            prediction REAL NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS XGB_PREDICTIONS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            prediction REAL NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS LSTM_PREDICTIONS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            prediction REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()

def trim_table(table_name, max_rows=288):
    conn = sqlite3.connect('glucose.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    if count > max_rows:
        cursor.execute(f"DELETE FROM {table_name} WHERE id IN (SELECT id FROM {table_name} ORDER BY id ASC LIMIT ?)", (count - max_rows,))
    conn.commit()
    conn.close()

# Function to add a glucose reading to the database
def add_glucose_reading(glucose_value):
    conn = sqlite3.connect('glucose.db')
    cursor = conn.cursor()
    now = datetime.now()
    cursor.execute("INSERT INTO GLUCOSE_READINGS (hour, minute, glucose_level) VALUES (?, ?, ?)",
                   (now.hour, now.minute, glucose_value))
    conn.commit()
    conn.close()
    print(f"Added glucose reading: hour={now.hour}, minute={now.minute}, glucose_level={glucose_value}")
    trim_table('GLUCOSE_READINGS')

# Background thread function for Dexcom updates
def update_glucose_readings():
    dexcom = Dexcom("Username", "Password")
    while True:
        glucose_reading = dexcom.get_current_glucose_reading()
        add_glucose_reading(glucose_reading.value)
        print(f"Reading added: {glucose_reading.value} at {datetime.now()}")
        time.sleep(300)  # Update every 5 minutes

# Function to save predictions
def save_prediction(table_name, hour, minute, prediction):
    conn = sqlite3.connect('glucose.db')
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO {table_name} (hour, minute, prediction) VALUES (?, ?, ?)",
                   (int(hour), int(minute), float(prediction)))
    conn.commit()
    conn.close()
    print(f"Saved prediction: table={table_name}, hour={hour}, minute={minute}, prediction={prediction}")
    trim_table(table_name)

# Create dataset function for LSTM
def create_dataset(X, y, time_steps=1, future_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - future_steps + 1):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps + future_steps - 1])
    return np.array(Xs), np.array(ys)

# Function to update Random Forest predictions
def update_rf_predictions():
    while True:
        conn = sqlite3.connect('glucose.db')
        df = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS", conn)
        conn.close()

        if df.empty:
            print("No data available for Random Forest prediction")
            time.sleep(300)
            continue

        df['hour'] = df['hour'].astype(int)
        df['minute'] = df['minute'].astype(int)

        # Log data for debugging
        print("Training data for Random Forest:")
        print(df.tail())

        model_path = 'random_forest_model.pkl'
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_model.fit(df[['hour', 'minute']], df['glucose_level'])
        joblib.dump(rf_model, model_path)

        next_hour = (df['hour'].iloc[-1] + (df['minute'].iloc[-1] + 30) // 60) % 24
        next_minute = (df['minute'].iloc[-1] + 30) % 60
        next_data_point = pd.DataFrame([[next_hour, next_minute]], columns=['hour', 'minute'])
        prediction = rf_model.predict(next_data_point)
        rounded_prediction = int(round(prediction[0]))
        save_prediction("RF_PREDICTIONS", next_hour, next_minute, rounded_prediction)
        print(f"RF prediction: {rounded_prediction} for time {next_hour}:{next_minute}")
        time.sleep(300)  # Update every 5 minutes

# Function to update XGBoost predictions
def update_xgb_predictions():
    while True:
        conn = sqlite3.connect('glucose.db')
        df = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS", conn)
        conn.close()

        if df.empty:
            print("No data available for XGBoost prediction")
            time.sleep(300)
            continue

        df['hour'] = df['hour'].astype(int)
        df['minute'] = df['minute'].astype(int)

        # Log data for debugging
        print("Training data for XGBoost:")
        print(df.tail())

        model_path = 'xgboost_model.pkl'
        X_train, X_test, y_train, y_test = train_test_split(df[['hour', 'minute']], df['glucose_level'], test_size=0.2, random_state=42)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

        next_hour = (df['hour'].iloc[-1] + (df['minute'].iloc[-1] + 30) // 60) % 24
        next_minute = (df['minute'].iloc[-1] + 30) % 60
        next_data_point = pd.DataFrame([[next_hour, next_minute]], columns=['hour', 'minute'])
        prediction = model.predict(next_data_point)
        rounded_prediction = int(round(prediction[0]))
        save_prediction("XGB_PREDICTIONS", next_hour, next_minute, rounded_prediction)
        print(f"XGB prediction: {rounded_prediction} for time {next_hour}:{next_minute}")
        time.sleep(300)  # Update every 5 minutes

# Function to update LSTM predictions
def update_lstm_predictions():
    while True:
        conn = sqlite3.connect('glucose.db')
        df = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS", conn)
        conn.close()

        if df.empty:
            print("No data available for LSTM prediction")
            time.sleep(300)
            continue

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[['glucose_level']])
        time_steps = 10
        future_steps_2hours = 24

        X_2hours, y_2hours = create_dataset(data_scaled, data_scaled, time_steps, future_steps_2hours)
        if X_2hours.size == 0 or y_2hours.size == 0:
            print("Not enough data to create the LSTM dataset")
            time.sleep(300)
            continue

        model_path = 'lstm_model_2hours.h5'
        if not os.path.exists(model_path):
            X_train_2hours, X_test_2hours, y_train_2hours, y_test_2hours = train_test_split(X_2hours, y_2hours, test_size=0.2, shuffle=False)
            model_2hours = Sequential()
            model_2hours.add(LSTM(units=64, input_shape=(X_train_2hours.shape[1], X_train_2hours.shape[2])))
            model_2hours.add(Dense(units=1))
            model_2hours.compile(optimizer='adam', loss='mean_squared_error')
            model_2hours.fit(X_train_2hours, y_train_2hours, epochs=100, batch_size=32, validation_split=0.2)
            model_2hours.save(model_path)
        else:
            model_2hours = load_model(model_path)

        X_pred_2hours = np.array([data_scaled[-time_steps:]])
        y_pred_2hours = model_2hours.predict(X_pred_2hours)
        y_pred_rescaled_2hours = scaler.inverse_transform(y_pred_2hours.reshape(-1, 1))
        rounded_prediction = int(round(y_pred_rescaled_2hours[0][0]))

        last_hour = df['hour'].iloc[-1]
        last_minute = df['minute'].iloc[-1]
        next_hour = (last_hour + (last_minute + 120) // 60) % 24
        next_minute = (last_minute + 120) % 60

        save_prediction("LSTM_PREDICTIONS", next_hour, next_minute, rounded_prediction)
        time.sleep(300)  # Update every 5 minutes

# Fetch data from SQLite database
def fetch_data():
    conn = sqlite3.connect('glucose.db')
    glucose_readings = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS ORDER BY id ASC", conn)
    rf_predictions = pd.read_sql_query("SELECT * FROM RF_PREDICTIONS ORDER BY id ASC", conn)
    xgb_predictions = pd.read_sql_query("SELECT * FROM XGB_PREDICTIONS ORDER BY id ASC", conn)
    lstm_predictions = pd.read_sql_query("SELECT * FROM LSTM_PREDICTIONS ORDER BY id ASC", conn)
    conn.close()
    return glucose_readings, rf_predictions, xgb_predictions, lstm_predictions

# Align predictions with actual values
def align_predictions(glucose_readings, predictions):
    aligned_data = pd.DataFrame()
    aligned_data['actual'] = glucose_readings['glucose_level'].shift(-1)  # Shift to align actuals with predictions
    aligned_data['predicted'] = predictions['prediction']
    aligned_data.dropna(inplace=True)
    return aligned_data

# Generate and save line graphs
def generate_and_save_graphs():
    while True:
        glucose_readings, rf_predictions, xgb_predictions, lstm_predictions = fetch_data()
        
        rf_data = align_predictions(glucose_readings, rf_predictions)
        xgb_data = align_predictions(glucose_readings, xgb_predictions)
        lstm_data = align_predictions(glucose_readings, lstm_predictions)

        for model, data in zip(['rf', 'xgb', 'lstm'], [rf_data, xgb_data, lstm_data]):
            plt.figure()
            plt.plot(data.index, data['actual'], label='Actual')
            plt.plot(data.index, data['predicted'], label='Predicted')
            plt.title(f'{model.upper()} Predictions vs Actual')
            plt.xlabel('Time')
            plt.ylabel('Glucose Level')
            plt.legend()
            plt.savefig(f'{model}_predictions_vs_actual.png')
            plt.close()
        
        time.sleep(300)  # Update every 5 minutes

# Start background threads for prediction updates and graph generation
rf_thread = threading.Thread(target=update_rf_predictions)
xgb_thread = threading.Thread(target=update_xgb_predictions)
lstm_thread = threading.Thread(target=update_lstm_predictions)
update_thread = threading.Thread(target=update_glucose_readings)
graph_thread = threading.Thread(target=generate_and_save_graphs)

rf_thread.daemon = True
xgb_thread.daemon = True
lstm_thread.daemon = True
update_thread.daemon = True
graph_thread.daemon = True

rf_thread.start()
xgb_thread.start()
lstm_thread.start()
update_thread.start()
graph_thread.start()

# Flask routes for retrieving the latest predictions
@app.route('/predict_random_forest', methods=['GET'])
def predict_random_forest():
    conn = sqlite3.connect('glucose.db')
    df = pd.read_sql_query("SELECT * FROM RF_PREDICTIONS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No predictions available'})

    latest_prediction = df.iloc[0]
    return jsonify({'predicted_glucose': latest_prediction['prediction']})

@app.route('/predict_xgboost', methods=['GET'])
def predict_xgboost():
    conn = sqlite3.connect('glucose.db')
    df = pd.read_sql_query("SELECT * FROM XGB_PREDICTIONS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No predictions available'})

    latest_prediction = df.iloc[0]
    return jsonify({'predicted_glucose': latest_prediction['prediction']})

@app.route('/predict_lstm', methods=['GET'])
def predict_lstm():
    conn = sqlite3.connect('glucose.db')
    df = pd.read_sql_query("SELECT * FROM LSTM_PREDICTIONS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No predictions available'})

    latest_prediction = df.iloc[0]
    return jsonify({'predicted_glucose_2hours': latest_prediction['prediction']})

# Flask route for serving graphs
@app.route('/graph/<model>', methods=['GET'])
def get_graph(model):
    valid_models = ['rf', 'xgb', 'lstm']
    if model in valid_models:
        file_path = f'{model}_predictions_vs_actual.png'
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Graph not found'}), 404
    else:
        return jsonify({'error': 'Invalid model'}), 400

# Flask route for fetching current glucose level and status
@app.route('/current_glucose', methods=['GET'])
def current_glucose():
    conn = sqlite3.connect('glucose.db')
    df = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No glucose readings available'})

    latest_reading = df.iloc[0]
    glucose_level = latest_reading['glucose_level']
    status = 'Normal'

    if glucose_level < 70:
        status = 'Low'
    elif glucose_level > 180:
        status = 'High'

    return jsonify({'glucose_level': glucose_level, 'status': status})

if __name__ == "__main__":
    app.run(debug=False, port=5000)