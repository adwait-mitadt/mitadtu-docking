"""ISS Docking inference.

This module mirrors the preprocessing and output scaling used in training:
- image: JPEG decode -> resize to 224x224 -> float32 -> divide by 255
- labels during training: [x, y, distance] divided by 512

Inference loads a fully trained Keras model from a single `.h5` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import load_coordinates, show_image


IMG_SIZE = 224
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)
INPUT_SCALE = 255.0
OUTPUT_SCALE = 512.0

DEFAULT_MODEL_PATH = Path("models") / "resnet_docking_best.h5"


def load_and_preprocess_image(image_path: str | Path, target_size: tuple[int, int] = TARGET_SIZE) -> tf.Tensor:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        img_size (int): Target image size
    
    Returns:
        np.ndarray: Preprocessed image ready for model input
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = tf.io.read_file(str(image_path))
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / INPUT_SCALE
    return tf.expand_dims(image, axis=0)


def load_trained_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """Load a fully trained Keras model from disk (.h5)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train/export a full Keras .h5 model or point to an existing *.h5 file."
        )

    return tf.keras.models.load_model(str(model_path), compile=False)


def predict_single_image(
    model: tf.keras.Model,
    image_path: str | Path,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> dict[str, float]:
    """
    Make prediction on a single image and convert outputs back to
    the original label scale.
    
    Args:
        model: Trained Keras model
        image_path (str): Path to the image
        img_size (int): Image size
    
    Returns:
        dict: Prediction results in original scale
    """
    # Preprocess image
    image = load_and_preprocess_image(image_path, target_size)

    # Model outputs are in training label scale: value / 512
    prediction_norm = model(image, training=False).numpy()
    
    # Convert to original scale (training divided labels by 512)
    prediction_original = prediction_norm * OUTPUT_SCALE
    
    result = {
        "x": float(prediction_original[0, 0]),
        "y": float(prediction_original[0, 1]),
        "distance": float(prediction_original[0, 2]),
        "x_normalized": float(prediction_norm[0, 0]),
        "y_normalized": float(prediction_norm[0, 1]),
        "distance_normalized": float(prediction_norm[0, 2]),
    }
    
    return result


def predict_batch(
    model: tf.keras.Model,
    image_paths: Iterable[str | Path],
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """
    Make predictions on multiple images.
    
    Args:
        model: Trained Keras model
        image_paths (list): List of image paths
        img_size (int): Image size
    
    Returns:
        np.ndarray: Predictions in original scale, shape (n_images, 3)
    """
    # Preprocess all images
    images: list[tf.Tensor] = []
    for image_path in image_paths:
        image = load_and_preprocess_image(image_path, target_size)
        images.append(image[0])

    if not images:
        return np.zeros((0, 3), dtype=np.float32)

    batch = tf.stack(images, axis=0)

    # Make predictions (normalized)
    predictions_norm = model(batch, training=False).numpy()
    
    # Convert to original scale (training divided labels by 512)
    predictions_original = predictions_norm * OUTPUT_SCALE
    
    return predictions_original


def main():
    """Example usage of the inference functions."""
    
    print("🚀 ISS Docking Inference Example")
    print("="*60)
    
    print(f"📂 Loading model from: {DEFAULT_MODEL_PATH}")
    model = load_trained_model(DEFAULT_MODEL_PATH)
    
    # Example: Predict on a single image
    print("\n" + "="*60)
    print("🖼️ Single Image Prediction Example")
    print("="*60)
    
    # Prefer the same source folder used during training (train.py uses `data/train`).
    test_image_path = "data/train/0.jpg"
    
    if Path(test_image_path).exists():
        result = predict_single_image(model, test_image_path)
        
        print(f"\n📸 Image: {test_image_path}")
        print(f"\n🎯 Predictions (Original Scale):")
        print(f"   X coordinate: {result['x']:.2f} pixels")
        print(f"   Y coordinate: {result['y']:.2f} pixels")
        print(f"   Distance: {result['distance']:.2f} meters")
        
        print(f"\n📊 Predictions (Model Output Scale = value / 512):")
        print(f"   X: {result['x_normalized']:.4f}")
        print(f"   Y: {result['y_normalized']:.4f}")
        print(f"   Distance: {result['distance_normalized']:.4f}")

        image_id = int(Path(test_image_path).stem)
        actual_x, actual_y = load_coordinates(image_id)
        plt.ion()
        show_image(image_id)
        plt.scatter(result["x"], result["y"], color="blue", label="Predicted")
        plt.legend()
        plt.ioff()
        plt.show()
    else:
        print(f"⚠️  Test image not found: {test_image_path}")
    
    # Example: Batch prediction
    print("\n" + "="*60)
    print("📦 Batch Prediction Example")
    print("="*60)
    
    image_dir = Path("data/train")
    test_images = list(image_dir.glob("*.jpg"))[:5]  # First 5 images
    
    if test_images:
        print(f"\n🖼️ Predicting on {len(test_images)} images...")
        predictions = predict_batch(model, test_images)
        
        
        print(f"\n📊 Batch Predictions (Original Scale):")
        print(f"{'Image':<20} {'X (px)':<12} {'Y (px)':<12} {'Distance (m)':<15}")
        print("-"*60)
        for img_path, pred in zip(test_images, predictions):
            print(f"{img_path.name:<20} {pred[0]:<12.2f} {pred[1]:<12.2f} {pred[2]:<15.2f}")
    else:
        print(f"⚠️  No test images found in: {image_dir}")
    
    print("\n" + "="*60)
    print("✅ Inference examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
