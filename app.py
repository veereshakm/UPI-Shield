from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.secret_key = 'super_secret_key'

# Ensure data directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model and Scaler
MODEL_PATH = 'model/fraud_model.h5'
SCALER_PATH = 'model/scaler.pkl'

def load_artifacts():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return None, None

model, scaler = load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin':
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'transactions.csv'))
            return render_template('upload.html', message="File uploaded successfully!")
    return render_template('upload.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        # Trigger training script
        # In a real app, this should be async (Celery/RQ)
        # For demo, we'll just run the script or import the function
        # Using os.system for simplicity to run in separate process context if needed, 
        # or better, import the function. Let's import.
        from model.train_model import train_model
        # We need to change CWD or adjust paths in train_model if importing
        # But train_model.py assumes relative path '../data'. 
        # Let's just run it via command line to be safe with paths
        import subprocess
        subprocess.run(["python", "model/train_model.py"], check=True)
        
        # Reload model
        global model, scaler
        model, scaler = load_artifacts()
        
        return render_template('upload.html', message="Training completed successfully!")
    except Exception as e:
        return render_template('upload.html', message=f"Training failed: {e}")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Extract features
            amount = float(request.form['amount'])
            category = int(request.form['category'])
            hour = int(request.form['hour'])
            zip_code = int(request.form['zip_code'])
            dob = request.form['dob']
            
            # Calculate Age
            from datetime import datetime
            dob_date = datetime.strptime(dob, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
            
            # Preprocess
            # Feature order must match training: trans_amount, category, trans_hour, zip, age
            features = np.array([[amount, category, hour, zip_code, age]])
            
            if scaler:
                features_scaled = scaler.transform(features)
                features_reshaped = features_scaled.reshape(1, 5, 1)
                
                # Predict
                prob = model.predict(features_reshaped)[0][0]
                prediction_text = "Fraud" if prob > 0.5 else "Valid"
            else:
                prediction_text = "Model not loaded"
                
        except Exception as e:
            print(e)
            prediction_text = f"Error: {e}"
            
    return render_template('predict.html', prediction=prediction_text)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
