"""
ISS Docking Vision Training Script with Output Normalization
Train ResNet50 model for ISS docking position regression (x, y, distance)
Uses independent MinMaxScalers for each output
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import project modules
from resnet_model import build_resnet_regression
from data_split import create_training_datasets, load_data_splits
from helpers import (
    create_output_scalers, 
    normalize_outputs, 
    denormalize_outputs,
    save_output_scalers,
    load_output_scalers
)

# ==================== CONFIGURATION ====================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224 
IMAGE_DIR = "data/resized"  # Use resized images
MODEL_SAVE_PATH = "models/resnet_docking.h5"
SCALERS_SAVE_PATH = "models/output_scalers.pkl"
# ========================================================


class OutputNormalizationLayer(tf.keras.layers.Layer):
    """Custom layer to normalize outputs during training"""
    
    def __init__(self, scalers, **kwargs):
        super().__init__(**kwargs)
        self.scalers = scalers
        
        # Extract normalization parameters from scalers
        self.x_min = tf.constant(scalers['x'].data_min_[0], dtype=tf.float32)
        self.x_scale = tf.constant(scalers['x'].scale_[0], dtype=tf.float32)
        
        self.y_min = tf.constant(scalers['y'].data_min_[0], dtype=tf.float32)
        self.y_scale = tf.constant(scalers['y'].scale_[0], dtype=tf.float32)
        
        self.dist_min = tf.constant(scalers['distance'].data_min_[0], dtype=tf.float32)
        self.dist_scale = tf.constant(scalers['distance'].scale_[0], dtype=tf.float32)
    
    def call(self, outputs):
        """Normalize each output independently"""
        x = outputs[:, 0:1]
        y = outputs[:, 1:2]
        dist = outputs[:, 2:3]
        
        # Apply MinMax normalization: (value - min) * scale
        x_norm = (x - self.x_min) * self.x_scale
        y_norm = (y - self.y_min) * self.y_scale
        dist_norm = (dist - self.dist_min) * self.dist_scale
        
        return tf.concat([x_norm, y_norm, dist_norm], axis=1)


class OutputDenormalizationLayer(tf.keras.layers.Layer):
    """Custom layer to denormalize outputs during inference"""
    
    def __init__(self, scalers, **kwargs):
        super().__init__(**kwargs)
        self.scalers = scalers
        
        # Extract denormalization parameters
        self.x_min = tf.constant(scalers['x'].data_min_[0], dtype=tf.float32)
        self.x_scale = tf.constant(scalers['x'].scale_[0], dtype=tf.float32)
        
        self.y_min = tf.constant(scalers['y'].data_min_[0], dtype=tf.float32)
        self.y_scale = tf.constant(scalers['y'].scale_[0], dtype=tf.float32)
        
        self.dist_min = tf.constant(scalers['distance'].data_min_[0], dtype=tf.float32)
        self.dist_scale = tf.constant(scalers['distance'].scale_[0], dtype=tf.float32)
    
    def call(self, normalized_outputs):
        """Denormalize each output independently"""
        x_norm = normalized_outputs[:, 0:1]
        y_norm = normalized_outputs[:, 1:2]
        dist_norm = normalized_outputs[:, 2:3]
        
        # Apply inverse MinMax: value / scale + min
        x = x_norm / self.x_scale + self.x_min
        y = y_norm / self.y_scale + self.y_min
        dist = dist_norm / self.dist_scale + self.dist_min
        
        return tf.concat([x, y, dist], axis=1)


def create_normalized_datasets(data_dir="data", image_dir="data/resized", 
                               batch_size=32, img_size=224):
    """
    Create datasets with normalized outputs.
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, scalers)
    """
    print("📊 Creating datasets with output normalization...")
    
    # Load data splits
    train_df, val_df, test_df = load_data_splits(data_dir)
    
    # Extract outputs from training data to fit scalers
    train_outputs = train_df[['x', 'y', 'distance']].values
    
    # Create and fit scalers on training data only
    scalers = create_output_scalers(train_outputs, feature_range=(0, 1))
    
    # Normalize outputs
    train_outputs_norm, _ = normalize_outputs(train_outputs, scalers)
    val_outputs_norm, _ = normalize_outputs(val_df[['x', 'y', 'distance']].values, scalers)
    test_outputs_norm, _ = normalize_outputs(test_df[['x', 'y', 'distance']].values, scalers)
    
    # Create TensorFlow datasets
    IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
    IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])
    
    def preprocess_function(filename, outputs):
        # Load and preprocess image
        image_path = tf.strings.join([str(image_dir) + "/", filename])
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        
        return image, outputs
    
    def create_dataset(df, outputs_norm, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((
            df['filename'].values,
            outputs_norm.astype(np.float32)
        ))
        dataset = dataset.map(preprocess_function, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(1000)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_ds = create_dataset(train_df, train_outputs_norm, shuffle=True)
    val_ds = create_dataset(val_df, val_outputs_norm, shuffle=False)
    test_ds = create_dataset(test_df, test_outputs_norm, shuffle=False)
    
    print(f"✅ Created normalized datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_ds, val_ds, test_ds, scalers


def evaluate_model(model, test_ds, test_df, scalers):
    """
    Evaluate model and report metrics in original scale.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset (with normalized outputs)
        test_df: Test DataFrame (with original outputs)
        scalers: Output scalers for denormalization
    """
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION (Original Scale)")
    print("="*60)
    
    # Make predictions (normalized)
    predictions_norm = model.predict(test_ds, verbose=0)
    
    # Denormalize predictions
    predictions_original = denormalize_outputs(predictions_norm, scalers)
    
    # Get actual values (original scale)
    actual_original = test_df[['x', 'y', 'distance']].values
    
    # Calculate errors in original scale
    errors = np.abs(predictions_original - actual_original)
    
    # Overall metrics
    mae_overall = np.mean(errors)
    rmse_overall = np.sqrt(np.mean((predictions_original - actual_original)**2))
    
    # Per-output metrics
    mae_x = np.mean(errors[:, 0])
    mae_y = np.mean(errors[:, 1])
    mae_dist = np.mean(errors[:, 2])
    
    rmse_x = np.sqrt(np.mean((predictions_original[:, 0] - actual_original[:, 0])**2))
    rmse_y = np.sqrt(np.mean((predictions_original[:, 1] - actual_original[:, 1])**2))
    rmse_dist = np.sqrt(np.mean((predictions_original[:, 2] - actual_original[:, 2])**2))
    
    print(f"\n🎯 Overall Metrics:")
    print(f"   Mean Absolute Error (MAE): {mae_overall:.2f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse_overall:.2f}")
    
    print(f"\n📍 X Coordinate:")
    print(f"   MAE: {mae_x:.2f} pixels")
    print(f"   RMSE: {rmse_x:.2f} pixels")
    
    print(f"\n📍 Y Coordinate:")
    print(f"   MAE: {mae_y:.2f} pixels")
    print(f"   RMSE: {rmse_y:.2f} pixels")
    
    print(f"\n📏 Distance:")
    print(f"   MAE: {mae_dist:.2f} meters")
    print(f"   RMSE: {rmse_dist:.2f} meters")
    
    print("\n🔍 Sample Predictions (first 5):")
    print(f"{'Actual':<30} {'Predicted':<30} {'Error':<20}")
    print("-"*80)
    for i in range(min(5, len(actual_original))):
        actual = actual_original[i]
        pred = predictions_original[i]
        error = errors[i]
        print(f"({actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f})".ljust(30), end="")
        print(f"({pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f})".ljust(30), end="")
        print(f"({error[0]:.1f}, {error[1]:.1f}, {error[2]:.1f})")
    
    print("="*60)
    
    return {
        'mae_overall': mae_overall,
        'rmse_overall': rmse_overall,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_dist': mae_dist,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_dist': rmse_dist
    }


def main():
    print("🚀 ISS Docking Vision Training with Output Normalization")
    print(f"Config: {BATCH_SIZE} batch, {EPOCHS} epochs, {IMG_SIZE}x{IMG_SIZE}px images")
    print(f"Output normalization: 3 independent MinMaxScalers for x, y, distance")
    print("="*60)
    
    # Create datasets with normalized outputs
    train_ds, val_ds, test_ds, scalers = create_normalized_datasets(
        data_dir="data", 
        image_dir=IMAGE_DIR, 
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE
    )
    
    # Save scalers for inference
    os.makedirs("models", exist_ok=True)
    save_output_scalers(scalers, SCALERS_SAVE_PATH)
    
    # Build and compile model
    model = build_resnet_regression(learning_rate=LEARNING_RATE)
    
    print(f"\n🏗️ Model Architecture:")
    print(f"   Input: {IMG_SIZE}x{IMG_SIZE}x3 (ImageNet normalized)")
    print(f"   Base: ResNet50 (frozen)")
    print(f"   Output: 3 values (x, y, distance) - normalized to [0, 1]")
    
    # Train model
    print(f"\n🎓 Training for {EPOCHS} epochs...")
    os.makedirs("logs", exist_ok=True)
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_SAVE_PATH, 
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger("logs/training_history.csv", append=False),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                verbose=1
            )
        ],
        verbose=1
    )
    
    print(f"\n💾 Model saved: {MODEL_SAVE_PATH}")
    print(f"📊 Training history saved: logs/training_history.csv")
    print(f"🔧 Output scalers saved: {SCALERS_SAVE_PATH}")
    
    # Evaluate on test set
    test_df = pd.read_csv("data/test_split.csv")
    metrics = evaluate_model(model, test_ds, test_df, scalers)
    
    print("\n✅ Training complete!")
    print(f"\n📈 Final Test Performance:")
    print(f"   Overall RMSE: {metrics['rmse_overall']:.2f}")
    print(f"   X RMSE: {metrics['rmse_x']:.2f} pixels")
    print(f"   Y RMSE: {metrics['rmse_y']:.2f} pixels")
    print(f"   Distance RMSE: {metrics['rmse_dist']:.2f} meters")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
