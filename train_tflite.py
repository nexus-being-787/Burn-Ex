#!/usr/bin/env python3
"""
Burn-Ex -- TFLite Model Trainer
----------------------------------
Trains a Keras neural network for calorie regression and exports it to TFLite.
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from met_calculator import ACTIVITY_METS, kcal_per_min
from constants import ML_FEATURES, ROLLING_COLS

def label_to_kcal(row):
    met = ACTIVITY_METS.get(row['label'], 1.0)
    return kcal_per_min(met, float(row.get('weight_kg', 70.0)))

def engineer_features(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    for col in ROLLING_COLS:
        if col in df.columns:
            df[f'{col}_roll5_mean'] = df[col].rolling(5, min_periods=1).mean()
            df[f'{col}_roll5_std']  = df[col].rolling(5, min_periods=1).std().fillna(0)
    return df

def build_feature_matrix(df):
    base_cols = [c for c in ML_FEATURES if c in df.columns]
    rolling_cols = [c for c in df.columns if c.endswith(('_roll5_mean', '_roll5_std'))]
    all_cols = base_cols + rolling_cols
    X = df[all_cols].fillna(0.0).values
    return X, all_cols

def main():
    print("Loading training data...")
    df = pd.read_csv("training_data.csv")
    df['kcal_per_min'] = df.apply(label_to_kcal, axis=1)
    df = engineer_features(df)
    
    X, feature_names = build_feature_matrix(df)
    y = df['kcal_per_min'].values
    
    print(f"Extracted {X.shape[1]} features.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building Keras Model...")
    
    # Use Normalization layer so scaling is built into the TFLite model!
    norm_layer = tf.keras.layers.Normalization(axis=-1)
    norm_layer.adapt(X_train)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        norm_layer,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', metrics=['mae'])
                  
    print("Training...")
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, verbose=1)
    
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae:.4f} kcal/min")
    
    print("Exporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model to Android assets directory
    out_dir = "mobile_native/app/src/main/assets"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "calorie_model.tflite")
    
    with open(out_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Model saved to {out_path}")

if __name__ == '__main__':
    main()
