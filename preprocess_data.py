"""
Preprocessing Script for ISS Docking Images
============================================

This script processes raw images and saves them in a preprocessed format
that can be directly fed to the model for training.

Features:
- Loads images with their coordinates
- Resizes to target size (224x224 by default)
- Normalizes using ImageNet statistics
- Saves processed images as .npy files for fast loading
- Generates metadata CSV for easy tracking
"""

from helpers import batch_process_and_save_images, load_processed_images
from pathlib import Path

def main():
    """Main preprocessing pipeline"""
    
    print("="*70)
    print("🚀 ISS DOCKING IMAGE PREPROCESSING PIPELINE")
    print("="*70)
    
    # Configuration
    csv_file = "data/train_labels.csv"
    image_folder = "data/train"
    output_folder = "data/processed/images"
    metadata_file = "data/processed/metadata/metadata.csv"
    target_size = (224, 224)  # Width, Height
    normalize = True
    
    # ImageNet normalization statistics (used by most pre-trained models)
    mean = [0.485, 0.456, 0.406]  # RGB
    std = [0.229, 0.224, 0.225]   # RGB
    
    print("\n📋 Configuration:")
    print(f"   Input CSV: {csv_file}")
    print(f"   Input Images: {image_folder}")
    print(f"   Output Folder: {output_folder}")
    print(f"   Metadata File: {metadata_file}")
    print(f"   Target Size: {target_size}")
    print(f"   Normalize: {normalize}")
    if normalize:
        print(f"   Mean: {mean}")
        print(f"   Std: {std}")
    print()
    
    # Process all images
    metadata_df = batch_process_and_save_images(
        csv_file=csv_file,
        image_folder=image_folder,
        output_folder=output_folder,
        metadata_file=metadata_file,
        target_size=target_size,
        normalize=normalize,
        mean=mean,
        std=std
    )
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total images processed: {len(metadata_df)}")
    print(f"   Processed images location: {output_folder}")
    print(f"   Metadata file: {metadata_file}")
    
    # Test loading a few processed images
    print("\n" + "="*70)
    print("🧪 TESTING: Loading first 10 processed images...")
    print("="*70)
    
    images, coords, meta = load_processed_images(
        metadata_file=metadata_file,
        max_images=10
    )
    
    print(f"\n✅ Test successful!")
    print(f"   Loaded images shape: {images.shape}")
    print(f"   Loaded coordinates shape: {coords.shape}")
    print(f"   Sample coordinates: {coords[:3]}")
    
    print("\n" + "="*70)
    print("🎯 READY FOR MODEL TRAINING!")
    print("="*70)
    print("\nTo load processed data for training, use:")
    print("   from helpers import load_processed_images")
    print("   images, coords, metadata = load_processed_images()")
    print()

if __name__ == "__main__":
    main()
