import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def train_model():
    # Load data
    # Check for uploaded file first, then fallback to default dataset
    data_path = '../data/transactions.csv'
    if not os.path.exists(data_path):
        # Fallback to the user provided dataset in root if not in data folder yet
        # But app.py saves uploads to data/transactions.csv. 
        # The user provided file is at root/upi_fraud_dataset.csv.
        # Let's check root first for the initial training
        if os.path.exists('../upi_fraud_dataset.csv'):
            data_path = '../upi_fraud_dataset.csv'
        else:
            print("Data file not found!")
            return

    df = pd.read_csv(data_path)
    
    # Preprocessing
    # Columns: Id,trans_hour,trans_day,trans_month,trans_year,category,upi_number,age,trans_amount,state,zip,fraud_risk
    # Features: trans_amount, category, trans_hour, zip, age
    feature_cols = ['trans_amount', 'category', 'trans_hour', 'zip', 'age']
    
    # Ensure columns exist (handle case where uploaded CSV might have different names if it was the old format)
    # For now assuming the new format as per user request
    try:
        X = df[feature_cols].copy()
        y = df['fraud_risk']
    except KeyError:
        # Fallback for old dummy data format if needed, but user wants new dataset
        print("Columns not found. Ensure dataset matches schema: trans_amount, category, trans_hour, zip, age")
        return

    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for CNN (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build CNN Model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Save artifacts
    model.save('fraud_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_model()
