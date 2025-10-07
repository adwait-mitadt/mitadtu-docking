"""
Script to split validation labels equally into validation and test sets.
- Original val_labels_original.csv (2000 samples) will be split into:
  - New val_labels.csv (50% = 1000 samples)
  - New test_labels.csv (50% = 1000 samples)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_validation_equally():
    """Split validation data equally into validation and test sets"""
    
    # Load the original validation labels
    val_original_path = 'data/val_labels_original.csv'
    val_original = pd.read_csv(val_original_path)
    
    print(f"Original validation set size: {len(val_original)} samples")
    
    # Split equally: 50% validation, 50% test
    val_data, test_data = train_test_split(
        val_original, 
        test_size=0.5,  # 50% for test
        random_state=42,  # For reproducibility
        stratify=None  # No stratification needed for regression task
    )
    
    print(f"New validation set size: {len(val_data)} samples (50%)")
    print(f"New test set size: {len(test_data)} samples (50%)")
    
    # Save the new splits
    val_data.to_csv('data/val_labels.csv', index=False)
    test_data.to_csv('data/test_labels.csv', index=False)
    
    print("\nFiles created:")
    print(f"- data/val_labels.csv: {len(val_data)} samples (50% validation)")
    print(f"- data/test_labels.csv: {len(test_data)} samples (50% test)")
    
    # Display sample statistics
    print(f"\nSample statistics:")
    print(f"Validation set - x range: [{val_data['x'].min()}, {val_data['x'].max()}], y range: [{val_data['y'].min()}, {val_data['y'].max()}]")
    print(f"Test set - x range: [{test_data['x'].min()}, {test_data['x'].max()}], y range: [{test_data['y'].min()}, {test_data['y'].max()}]")
    
    # Verify no overlap
    val_filenames = set(val_data['filename'])
    test_filenames = set(test_data['filename'])
    overlap = len(val_filenames & test_filenames)
    total_unique = len(val_filenames | test_filenames)
    
    print(f"\nVerification:")
    print(f"- No overlapping files: {overlap == 0}")
    print(f"- Total unique files: {total_unique}")
    print(f"- All files accounted for: {total_unique == len(val_original)}")
    
    return val_data, test_data

if __name__ == "__main__":
    val_data, test_data = split_validation_equally()