"""
ISS Docking Model Training Script
Minimal training pipeline using helper functions
"""

import pandas as pd
import tensorflow as tf
from pathlib import Path

# Import model builder
from resnet_model import build_resnet_regression

# Configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = (224, 224)
IMAGE_DIR = "data/train"
MODEL_PATH = "models/resnet_docking.h5"


def create_dataset(df, image_dir, batch_size, img_size, shuffle=False):
    """Create TensorFlow dataset from DataFrame with x, y, distance outputs."""
    def preprocess(filename, labels):
        # Load and preprocess image
        image_path = tf.strings.join([image_dir + "/", filename])
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Normalize labels to [0, 1]
        labels_norm = labels / [512.0, 512.0, 512.0]  # x, y, distance
        return image, labels_norm
    
    dataset = tf.data.Dataset.from_tensor_slices((
        df['filename'].values,
        df[['x', 'y', 'distance']].values.astype('float32')
    ))
    
    if shuffle:
        dataset = dataset.shuffle(1000)
    
    return dataset.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    print("🚀 ISS Docking Model Training")
    
    # Load split data
    train_df = pd.read_csv("data/train_split.csv")
    val_df = pd.read_csv("data/val_split.csv")
    test_df = pd.read_csv("data/test_split.csv")
    
    print(f"📊 Data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Create datasets
    train_ds = create_dataset(train_df, IMAGE_DIR, BATCH_SIZE, IMG_SIZE, shuffle=True)
    val_ds = create_dataset(val_df, IMAGE_DIR, BATCH_SIZE, IMG_SIZE)
    test_ds = create_dataset(test_df, IMAGE_DIR, BATCH_SIZE, IMG_SIZE)
    
    # Build model
    model = build_resnet_regression(learning_rate=LEARNING_RATE)
    print(f"✅ Model built with learning rate {LEARNING_RATE}")
    
    # Train
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger("logs/training_history.csv")
        ]
    )
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    results = model.evaluate(test_ds)
    print(f"✅ Test results: {dict(zip(model.metrics_names, results))}")
    print(f"💾 Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()