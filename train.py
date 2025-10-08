"""
ISS Docking Vision Training Script
Train ResNet50 model for ISS docking position regression
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import project modules
from resnet_model import build_resnet_regression
from data_split import create_training_datasets
from ml_visualizer import visualize_training_results

# ==================== CONFIGURATION ====================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224 
IMAGE_DIR = "data/train"
MODEL_SAVE_PATH = "models/resnet_docking.h5"
# ========================================================


def main():
    print("🚀 ISS Docking Vision Training")
    print(f"Config: {BATCH_SIZE} batch, {EPOCHS} epochs, {IMG_SIZE}x{IMG_SIZE}px images")
    
    # all datasets using data_split.py 
    train_ds, val_ds, test_ds = create_training_datasets(
        data_dir="data", image_dir=IMAGE_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    
    # Build and compile model using resnet_model.py
    os.makedirs("models", exist_ok=True)
    model = build_resnet_regression(learning_rate=LEARNING_RATE)  # Pass learning rate
    
    # Train model
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger("logs/training_history.csv", append=False)
        ]
    )
    
    # Evaluate and save
    # test_loss, test_mae = model.evaluate(test_ds)
    # print(f"✅ Test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}")
    print(f"💾 Model saved: {MODEL_SAVE_PATH}")
    print(f"📊 Training history saved: logs/training_history.csv")
    
    # Visualize training results using the ML Training Visualizer
    visualize_training_results(history, experiment_name="ISS_Docking_ResNet50")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()