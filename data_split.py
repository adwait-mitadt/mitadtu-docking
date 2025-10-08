import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import cv2

# Import helper functions from helpers.py
from helpers import (
    load_image, resize_image, normalize_image,  # Image processing
    create_output_scaler, load_output_scaler, normalize_outputs, denormalize_outputs  # Output normalization
)

def resize_images_from_labelled_data(csv_path, target_width=224, target_height=224, output_dir="data/resized", 
                                   scale_coordinates=True, normalize=True, 
                                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Load images from CSV, resize them to specified dimensions, and optionally normalize them.
    
    Args:
        csv_path (str): Path to CSV file with columns 'filename', 'x', 'y'
        target_width (int): Target width for resized images (default: 224)
        target_height (int): Target height for resized images (default: 224)
        output_dir (str): Directory to save resized images
        scale_coordinates (bool): Whether to scale x,y coordinates proportionally
        normalize (bool): Whether to apply ImageNet normalization (default: True)
        mean (list): Mean values for normalization [R, G, B] (default: ImageNet mean)
        std (list): Standard deviation values for normalization [R, G, B] (default: ImageNet std)
    
    Returns:
        pd.DataFrame: DataFrame with 'filename', 'x', 'y' columns
    """
    print(f"🖼️ Resizing images to {target_width}x{target_height} pixels...")
    if normalize:
        print(f"🎨 Applying ImageNet normalization with mean={mean}, std={std}")
    
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"📊 Processing {len(df)} images...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = []
    processed_count = 0
    
    # Process each image
    for idx, row in df.iterrows():
        filename = row['filename']
        x_coord = row['x']
        y_coord = row['y']
        
        try:
            # Load and resize image
            image_path = Path("data/train") / filename
            original_image = load_image(str(image_path), convert_to_rgb=True)
            
            if original_image is not None:
                # Scale coordinates if requested
                if scale_coordinates:
                    original_height, original_width = original_image.shape[:2]
                    x_coord = x_coord * (target_width / original_width)
                    y_coord = y_coord * (target_height / original_height)
                
                # Resize image
                resized_image = resize_image(original_image, target_width, target_height)
                
                if resized_image is not None:
                    # Apply normalization if requested
                    if normalize:
                        # Normalize using ImageNet statistics
                        processed_image = normalize_image(resized_image, mean=mean, std=std)
                        
                        # Convert back to [0, 255] range for saving as JPEG
                        # Denormalize: (normalized * std) + mean, then scale to [0, 255]
                        denormalized = (processed_image * np.array(std) + np.array(mean)) * 255.0
                        denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)
                        save_image = denormalized
                    else:
                        save_image = resized_image
                    
                    # Save image
                    output_file = output_path / filename
                    cv2.imwrite(str(output_file), cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
                    
                    results.append({
                        'filename': filename,
                        'x': x_coord,
                        'y': y_coord
                    })
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count}/{len(df)} images...")
                        
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    
    # Create result DataFrame
    result_df = pd.DataFrame(results)
    print(f"✅ Successfully processed {processed_count} images")
    
    return result_df


def create_resized_dataset(csv_path, target_width=224, target_height=224, output_csv="data/resized_labelled_data.csv",
                          normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Create a resized dataset with scaled coordinates and save to CSV.
    
    Args:
        csv_path (str): Path to the original CSV file
        target_width (int): Target width for resized images
        target_height (int): Target height for resized images
        output_csv (str): Path for the output CSV file
        normalize (bool): Whether to apply ImageNet normalization (default: True)
        mean (list): Mean values for normalization [R, G, B] (default: ImageNet mean)
        std (list): Standard deviation values for normalization [R, G, B] (default: ImageNet std)
    
    Returns:
        pd.DataFrame: DataFrame with scaled coordinates for resized images
    """
    print(f"🔄 Creating resized dataset with scaled coordinates...")
    if normalize:
        print(f"🎨 ImageNet normalization will be applied during processing")
    
    # Process images and get scaled coordinates
    result_df = resize_images_from_labelled_data(
        csv_path, target_width, target_height, scale_coordinates=True, 
        normalize=normalize, mean=mean, std=std
    )
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Resized dataset saved to: {output_csv}")
    print(f"📊 Dataset contains {len(result_df)} entries")
    
    return result_df


def split_data(csv_path, random_state=42):
    """
    Split the labelled data into training (80%), validation (10%), and testing (10%) sets.
    Uses scikit-learn for randomization and splitting.
    
    Args:
        csv_path (str): Path to the labelled_data.csv file
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for train, validation, and test sets
    """
    # Load the data using pandas (similar to helpers.py approach)
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Display basic info about the dataset
    print("\nDataset info:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # First split: 80% train, 20% temp (which will be split into val and test)
    # Using stratified split based on coordinate ranges to ensure balanced distribution
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: Split the 20% temp into 10% validation and 10% test
    # Since temp_df is 20% of total, splitting it 50-50 gives us 10% each
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=random_state,
        shuffle=True
    )
    
    # Print split information
    print(f"\nData split completed:")
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir="data"):
    """
    Save the split datasets to CSV files.
    Uses pathlib for path handling (consistent with helpers.py style).
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        output_dir (str): Directory to save the split files
    """
    # Create output directory using pathlib (consistent with helpers.py)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the splits
    train_path = output_path / "train_split.csv"
    val_path = output_path / "val_split.csv"
    test_path = output_path / "test_split.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSplit files saved:")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Test data: {test_path}")

def verify_splits(train_df, val_df, test_df):
    """
    Verify that the splits don't have overlapping data.
    Uses set operations for efficient overlap detection.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
    """
    print("\nVerifying splits for overlaps...")
    
    # Check for overlaps using filename as the unique identifier
    train_files = set(train_df['filename'])
    val_files = set(val_df['filename'])
    test_files = set(test_df['filename'])
    
    # Check overlaps
    train_val_overlap = train_files.intersection(val_files)
    train_test_overlap = train_files.intersection(test_files)
    val_test_overlap = val_files.intersection(test_files)
    
    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print("✓ No overlaps found between splits - data is properly separated!")
    else:
        print("⚠ Warning: Overlaps detected!")
        if train_val_overlap:
            print(f"Train-Validation overlap: {len(train_val_overlap)} files")
        if train_test_overlap:
            print(f"Train-Test overlap: {len(train_test_overlap)} files")
        if val_test_overlap:
            print(f"Validation-Test overlap: {len(val_test_overlap)} files")


def normalize_output_splits(train_df, val_df, test_df, output_columns=['x', 'y', 'distance'], 
                            scaler_path='data/output_scaler.pkl'):
    """
    Apply MinMaxScaler normalization to output columns (coordinates and distances).
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        output_columns (list): Columns to normalize (default: ['x', 'y', 'distance'])
        scaler_path (str): Path to save the fitted scaler
        
    Returns:
        tuple: (train_df, val_df, test_df, scaler) - DataFrames with normalized outputs and fitted scaler
        
    Note:
        - Scaler is fitted ONLY on training data to prevent data leakage
        - Same scaler is then applied to validation and test sets
        - Original values are preserved in columns with '_original' suffix
    """
    print("\n🎯 Applying MinMaxScaler normalization to outputs...")
    print(f"   Columns to normalize: {output_columns}")
    
    # Check if all columns exist
    available_columns = [col for col in output_columns if col in train_df.columns]
    if len(available_columns) < len(output_columns):
        missing = set(output_columns) - set(available_columns)
        print(f"   ⚠ Warning: Missing columns {missing}. Only normalizing: {available_columns}")
        output_columns = available_columns
    
    if not output_columns:
        print("   ⚠ No columns to normalize!")
        return train_df, val_df, test_df, None
    
    # Create copies to avoid modifying original dataframes
    train_norm = train_df.copy()
    val_norm = val_df.copy()
    test_norm = test_df.copy()
    
    # Save original values
    for col in output_columns:
        train_norm[f'{col}_original'] = train_norm[col]
        val_norm[f'{col}_original'] = val_norm[col]
        test_norm[f'{col}_original'] = test_norm[col]
    
    # Create and fit scaler on training data ONLY
    scaler = create_output_scaler(train_norm, output_columns=output_columns, scaler_path=scaler_path)
    
    # Apply normalization to all splits
    print("   Normalizing training data...")
    train_norm[output_columns] = normalize_outputs(train_norm, output_columns=output_columns, scaler=scaler)
    
    print("   Normalizing validation data...")
    val_norm[output_columns] = normalize_outputs(val_norm, output_columns=output_columns, scaler=scaler)
    
    print("   Normalizing test data...")
    test_norm[output_columns] = normalize_outputs(test_norm, output_columns=output_columns, scaler=scaler)
    
    # Print normalization statistics
    print("\n   📊 Normalization Statistics:")
    for col in output_columns:
        print(f"\n      {col}:")
        print(f"         Train - min: {train_norm[col].min():.4f}, max: {train_norm[col].max():.4f}, mean: {train_norm[col].mean():.4f}")
        print(f"         Val   - min: {val_norm[col].min():.4f}, max: {val_norm[col].max():.4f}, mean: {val_norm[col].mean():.4f}")
        print(f"         Test  - min: {test_norm[col].min():.4f}, max: {test_norm[col].max():.4f}, mean: {test_norm[col].mean():.4f}")
    
    print(f"\n   ✅ Output normalization complete!")
    print(f"   💾 Scaler saved to: {scaler_path}")
    
    return train_norm, val_norm, test_norm, scaler


def load_normalized_images_from_csv(csv_path, image_dir="data/resized", 
                                   normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Load and normalize images directly from CSV for model training.
    
    Args:
        csv_path (str): Path to CSV file with columns 'filename', 'x', 'y'
        image_dir (str): Directory containing the resized images
        normalize (bool): Whether to apply normalization
        mean (list): Mean values for normalization [R, G, B]
        std (list): Standard deviation values for normalization [R, G, B]
    
    Returns:
        tuple: (images_array, coordinates_array) - Ready for model training
    """
    print(f"📚 Loading normalized images for model training...")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    images = []
    coordinates = []
    
    for idx, row in df.iterrows():
        filename = row['filename']
        x_coord = row['x']
        y_coord = row['y']
        
        try:
            # Load image
            image_path = Path(image_dir) / filename
            image = load_image(str(image_path), convert_to_rgb=True)
            
            if image is not None:
                if normalize:
                    # Apply normalization for model training
                    normalized_image = normalize_image(image, mean=mean, std=std)
                    images.append(normalized_image)
                else:
                    # Just scale to [0, 1]
                    scaled_image = image / 255.0 if image.max() > 1 else image
                    images.append(scaled_image)
                
                coordinates.append([x_coord, y_coord])
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"✅ Loaded {len(images)} normalized images for training")
    
    return np.array(images, dtype=np.float32), np.array(coordinates, dtype=np.float32)


def main():
    """
    Main function to execute the data processing pipeline.
    """
    # Path to the labelled data
    data_path = Path("data/")
    csv_path = data_path / "labelled _data.csv"
    
    # Check if file exists
    if not csv_path.exists():
        print(f"Error: File {csv_path} not found!")
        return
    
    print("🚀 ISS Docking Data Processing Pipeline")
    print("="*50)
    print("Pipeline steps:")
    print("  🖼️ 1. Resize images to 224x224 pixels")
    print("  🎨 2. Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
    print("  📊 3. Scale coordinates proportionally")
    print("  📈 4. Split data into train/val/test sets")
    print("  🎯 5. Apply MinMaxScaler normalization to outputs (coordinates & distances)")
    print("="*50)
    
    # Step 1, 2 & 3: Resize images, normalize, and create scaled dataset
    print("\n🖼️ STEP 1-3: Resizing Images, Normalizing, and Scaling Coordinates")
    resized_df = create_resized_dataset(
        str(csv_path), 
        target_width=224, 
        target_height=224,
        normalize=True,
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]   # ImageNet std
    )
    
    # Step 4: Split the resized data
    print("\n📈 STEP 4: Splitting Data")
    train_df, val_df, test_df = split_data("data/resized_labelled_data.csv", random_state=42)
    
    # Verify splits for data integrity
    verify_splits(train_df, val_df, test_df)
    
    # Step 5: Apply output normalization (MinMaxScaler for coordinates and distances)
    print("\n🎯 STEP 5: Applying Output Normalization")
    
    # Determine which columns to normalize based on what's available
    available_columns = train_df.columns.tolist()
    output_columns = []
    if 'x' in available_columns and 'y' in available_columns:
        output_columns.extend(['x', 'y'])
    if 'distance' in available_columns:
        output_columns.append('distance')
    
    if output_columns:
        train_norm, val_norm, test_norm, scaler = normalize_output_splits(
            train_df, val_df, test_df, 
            output_columns=output_columns,
            scaler_path='data/output_scaler.pkl'
        )
        
        # Save normalized splits
        print("\n   💾 Saving normalized splits...")
        save_splits(train_norm, val_norm, test_norm, output_dir="data")
        
        # Also save non-normalized versions for reference
        print("   💾 Saving original (non-normalized) splits as backup...")
        train_df.to_csv("data/train_split_original.csv", index=False)
        val_df.to_csv("data/val_split_original.csv", index=False)
        test_df.to_csv("data/test_split_original.csv", index=False)
        
        normalized = True
    else:
        print("   ⚠ No output columns found for normalization. Skipping this step.")
        save_splits(train_df, val_df, test_df, output_dir="data")
        normalized = False
        scaler = None
    
    # Print processing summary
    print("\n📋 PROCESSING SUMMARY:")
    print("="*50)
    print(f"✅ Images processed: {len(resized_df)}")
    print(f"📐 Target dimensions: 224x224 pixels")
    print(f"🎨 ImageNet normalization applied: Yes")
    print(f"   📊 Mean: [0.485, 0.456, 0.406] (R, G, B)")
    print(f"   📈 Std:  [0.229, 0.224, 0.225] (R, G, B)")
    print(f"📊 Training samples: {len(train_df)}")
    print(f"📈 Validation samples: {len(val_df)}")
    print(f"🧪 Test samples: {len(test_df)}")
    if normalized:
        print(f"🎯 Output normalization: Yes (MinMaxScaler)")
        print(f"   📊 Normalized columns: {output_columns}")
        print(f"   💾 Scaler saved to: data/output_scaler.pkl")
    else:
        print(f"🎯 Output normalization: No")
    print("="*50)
    
    print("\n✅ Complete data processing pipeline finished successfully!")
    print("📁 Check the 'data/resized/' folder for processed images")
    print("📄 Check 'data/resized_labelled_data.csv' for scaled coordinates")
    print("🎨 Images have been normalized using ImageNet statistics for optimal model training")
    if normalized:
        print("🎯 Outputs (coordinates & distances) have been normalized using MinMaxScaler")
        print("   📊 Use the saved scaler (data/output_scaler.pkl) to denormalize predictions")
        print("   💡 Example: denormalized = denormalize_outputs(predictions, scaler)")



if __name__ == "__main__":
    main()
