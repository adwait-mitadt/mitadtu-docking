import pandas as pd
from pathlib import Path

def add_distance_to_split_files():
    """
    Add the distance column to train_split.csv, val_split.csv, and test_split.csv
    by matching filenames with labelled_data.csv
    """
    # Load the labelled data with distance column
    labelled_data = pd.read_csv('data/labelled _data.csv')
    print(f"📊 Loaded labelled_data.csv with {len(labelled_data)} rows")
    print(f"Columns: {labelled_data.columns.tolist()}")
    
    # Create a dictionary for quick lookup: filename -> distance
    distance_lookup = dict(zip(labelled_data['filename'], labelled_data['distance']))
    
    # Process each split file
    split_files = ['train_split.csv', 'val_split.csv', 'test_split.csv']
    
    for split_file in split_files:
        file_path = Path('data') / split_file
        
        if not file_path.exists():
            print(f"⚠️  {split_file} not found, skipping...")
            continue
        
        # Load the split file
        df = pd.read_csv(file_path)
        print(f"\n📂 Processing {split_file}...")
        print(f"   Original shape: {df.shape}")
        print(f"   Original columns: {df.columns.tolist()}")
        
        # Add distance column by looking up each filename
        df['distance'] = df['filename'].map(distance_lookup)
        
        # Check for any missing distances
        missing_count = df['distance'].isna().sum()
        if missing_count > 0:
            print(f"   ⚠️  Warning: {missing_count} rows have missing distance values")
            missing_files = df[df['distance'].isna()]['filename'].tolist()
            print(f"   Missing files: {missing_files[:5]}...")  # Show first 5
        else:
            print(f"   ✅ All distances added successfully!")
        
        # Save the updated file
        df.to_csv(file_path, index=False)
        print(f"   💾 Saved updated {split_file}")
        print(f"   New shape: {df.shape}")
        print(f"   New columns: {df.columns.tolist()}")
        
        # Show a sample
        print(f"   Sample data:")
        print(df.head(3).to_string(index=False))
    
    print("\n✅ All split files have been updated with distance column!")

if __name__ == "__main__":
    add_distance_to_split_files()
