"""
Example script demonstrating the normalization/denormalization pipeline
for ISS Docking training and inference.
"""

from helpers import (
    create_output_scalers, 
    normalize_outputs, 
    denormalize_outputs,
    save_output_scalers,
    load_output_scalers
)
from data_split import create_training_datasets
import numpy as np


def training_example():
    """
    Example: Training workflow with normalization
    """
    print("=" * 60)
    print("TRAINING WORKFLOW - Normalization Example")
    print("=" * 60)
    
    # Step 1: Create normalized datasets
    print("\n📊 Creating training datasets with normalization...")
    train_ds, val_ds, test_ds, scalers = create_training_datasets(
        data_dir="data",
        image_dir="data/resized",
        batch_size=32,
        img_size=224
    )
    
    if train_ds is None:
        print("❌ Failed to create datasets. Check if TensorFlow is installed.")
        return None
    
    print("\n✅ Datasets created with:")
    print("   INPUT normalization: ImageNet (mean/std)")
    print("   OUTPUT normalization: MinMaxScaler [0, 1] per feature")
    print("   Scalers saved to: data/output_scalers.pkl")
    
    return scalers


def inference_example():
    """
    Example: Inference workflow with denormalization
    """
    print("\n" + "=" * 60)
    print("INFERENCE WORKFLOW - Denormalization Example")
    print("=" * 60)
    
    # Step 1: Load scalers
    print("\n📂 Loading output scalers...")
    try:
        scalers = load_output_scalers(filepath='data/output_scalers.pkl')
        print("✅ Scalers loaded successfully!")
    except FileNotFoundError:
        print("❌ Scalers not found. Run training first!")
        return
    
    # Step 2: Simulate model predictions (normalized outputs in [0, 1])
    print("\n🤖 Simulating model predictions...")
    predictions_normalized = np.array([
        [0.5, 0.6, 0.3],  # Normalized [x, y, distance]
        [0.2, 0.8, 0.5],
        [0.9, 0.1, 0.7]
    ])
    print(f"   Normalized predictions shape: {predictions_normalized.shape}")
    print(f"   Normalized values (sample):\n{predictions_normalized}")
    
    # Step 3: Denormalize predictions back to original scale
    print("\n🔄 Denormalizing predictions to original pixel scale...")
    predictions_original = denormalize_outputs(predictions_normalized, scalers)
    print(f"   Denormalized predictions:\n{predictions_original}")
    
    # Step 4: Extract individual coordinates
    print("\n📍 Extracted coordinates:")
    for i, pred in enumerate(predictions_original):
        x, y, distance = pred
        print(f"   Image {i+1}: x={x:.2f}px, y={y:.2f}px, distance={distance:.2f}")
    
    return predictions_original


def complete_workflow_example():
    """
    Complete example showing the full normalization/denormalization cycle
    """
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Example raw outputs (pixel coordinates)
    raw_outputs = np.array([
        [100.5, 150.2, 45.8],  # x, y, distance in original scale
        [200.1, 80.5, 120.3],
        [50.0, 200.0, 30.5]
    ])
    
    print("\n1️⃣ ORIGINAL DATA (pixel coordinates):")
    print(f"   Shape: {raw_outputs.shape}")
    print(f"   Values:\n{raw_outputs}")
    
    # Create scalers and normalize
    print("\n2️⃣ NORMALIZING OUTPUTS...")
    scalers = create_output_scalers(raw_outputs, feature_range=(0, 1))
    normalized_outputs, _ = normalize_outputs(raw_outputs, scalers=scalers)
    print(f"   Normalized to [0, 1]:\n{normalized_outputs}")
    
    # Simulate model training and prediction (model works with normalized data)
    print("\n3️⃣ MODEL TRAINING (uses normalized data)...")
    print("   Model learns to predict values in [0, 1] range")
    
    # Denormalize predictions
    print("\n4️⃣ DENORMALIZING PREDICTIONS...")
    denormalized = denormalize_outputs(normalized_outputs, scalers)
    print(f"   Back to original scale:\n{denormalized}")
    
    # Verify round-trip accuracy
    print("\n5️⃣ VERIFICATION (round-trip accuracy):")
    difference = np.abs(raw_outputs - denormalized)
    max_error = np.max(difference)
    print(f"   Max error: {max_error:.10f}")
    print(f"   Status: {'✅ Perfect!' if max_error < 1e-6 else '❌ Error detected'}")


if __name__ == "__main__":
    print("🚀 ISS Docking Normalization/Denormalization Examples")
    print("=" * 60)
    
    # Example 1: Complete workflow demonstration
    complete_workflow_example()
    
    # Example 2: Training workflow (requires data files)
    print("\n\n" + "=" * 60)
    print("Note: The following examples require actual data files")
    print("=" * 60)
    
    response = input("\nRun training example? (requires data files) [y/N]: ")
    if response.lower() == 'y':
        scalers = training_example()
        
        if scalers:
            response = input("\nRun inference example? [y/N]: ")
            if response.lower() == 'y':
                inference_example()
    else:
        print("\n✅ Complete! Review the code to understand the workflow.")
