"""
ISS Docking Inference Script
Make predictions on new images with proper denormalization
"""

import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from helpers import load_output_scalers, denormalize_outputs


def load_and_preprocess_image(image_path, img_size=224):
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        img_size (int): Target image size
    
    Returns:
        np.ndarray: Preprocessed image ready for model input
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def predict_single_image(model, scalers, image_path, img_size=224):
    """
    Make prediction on a single image and denormalize the output.
    
    Args:
        model: Trained Keras model
        scalers: Dictionary of output scalers
        image_path (str): Path to the image
        img_size (int): Image size
    
    Returns:
        dict: Prediction results in original scale
    """
    # Preprocess image
    image = load_and_preprocess_image(image_path, img_size)
    
    # Make prediction (normalized [0, 1])
    prediction_norm = model.predict(image, verbose=0)
    
    # Denormalize to original scale
    prediction_original = denormalize_outputs(prediction_norm, scalers)
    
    result = {
        'x': prediction_original[0, 0],
        'y': prediction_original[0, 1],
        'distance': prediction_original[0, 2],
        'x_normalized': prediction_norm[0, 0],
        'y_normalized': prediction_norm[0, 1],
        'distance_normalized': prediction_norm[0, 2]
    }
    
    return result


def predict_batch(model, scalers, image_paths, img_size=224):
    """
    Make predictions on multiple images.
    
    Args:
        model: Trained Keras model
        scalers: Dictionary of output scalers
        image_paths (list): List of image paths
        img_size (int): Image size
    
    Returns:
        np.ndarray: Predictions in original scale, shape (n_images, 3)
    """
    # Preprocess all images
    images = []
    for image_path in image_paths:
        image = load_and_preprocess_image(image_path, img_size)
        images.append(image[0])  # Remove batch dimension
    
    images = np.array(images)
    
    # Make predictions (normalized)
    predictions_norm = model.predict(images, verbose=0)
    
    # Denormalize to original scale
    predictions_original = denormalize_outputs(predictions_norm, scalers)
    
    return predictions_original


def main():
    """Example usage of the inference functions"""
    
    print("🚀 ISS Docking Inference Example")
    print("="*60)
    
    # Load trained model
    MODEL_PATH = "models/resnet_docking.h5"
    SCALERS_PATH = "models/output_scalers.pkl"
    
    print(f"📂 Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"📂 Loading output scalers from: {SCALERS_PATH}")
    scalers = load_output_scalers(SCALERS_PATH)
    
    print("\n📊 Scaler Information:")
    for name, scaler in scalers.items():
        print(f"   {name.upper()}:")
        print(f"      Min: {scaler.data_min_[0]:.2f}")
        print(f"      Max: {scaler.data_max_[0]:.2f}")
    
    # Example: Predict on a single image
    print("\n" + "="*60)
    print("🖼️ Single Image Prediction Example")
    print("="*60)
    
    test_image_path = "data/resized/0.jpg"
    
    if Path(test_image_path).exists():
        result = predict_single_image(model, scalers, test_image_path)
        
        print(f"\n📸 Image: {test_image_path}")
        print(f"\n🎯 Predictions (Original Scale):")
        print(f"   X coordinate: {result['x']:.2f} pixels")
        print(f"   Y coordinate: {result['y']:.2f} pixels")
        print(f"   Distance: {result['distance']:.2f} meters")
        
        print(f"\n📊 Predictions (Normalized [0, 1]):")
        print(f"   X: {result['x_normalized']:.4f}")
        print(f"   Y: {result['y_normalized']:.4f}")
        print(f"   Distance: {result['distance_normalized']:.4f}")
    else:
        print(f"⚠️  Test image not found: {test_image_path}")
    
    # Example: Batch prediction
    print("\n" + "="*60)
    print("📦 Batch Prediction Example")
    print("="*60)
    
    image_dir = Path("data/resized")
    test_images = list(image_dir.glob("*.jpg"))[:5]  # First 5 images
    
    if test_images:
        print(f"\n🖼️ Predicting on {len(test_images)} images...")
        predictions = predict_batch(model, scalers, test_images)
        
        print(f"\n📊 Batch Predictions (Original Scale):")
        print(f"{'Image':<20} {'X (px)':<12} {'Y (px)':<12} {'Distance (m)':<15}")
        print("-"*60)
        for i, (img_path, pred) in enumerate(zip(test_images, predictions)):
            print(f"{img_path.name:<20} {pred[0]:<12.2f} {pred[1]:<12.2f} {pred[2]:<15.2f}")
    else:
        print(f"⚠️  No test images found in: {image_dir}")
    
    print("\n" + "="*60)
    print("✅ Inference examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
