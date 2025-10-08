import pandas as pd
import os

def add_distances_to_labelled_data():
    """
    Add distances column from train.csv to labelled_data.csv
    """
    # Read the files
    train_df = pd.read_csv('data/train.csv')
    labelled_df = pd.read_csv('data/labelled _data.csv')
    
    print(f"train.csv shape: {train_df.shape}")
    print(f"labelled_data.csv shape: {labelled_df.shape}")
    
    # Extract ImageID from filename in labelled_data (remove .jpg extension)
    labelled_df['ImageID'] = labelled_df['filename'].str.replace('.jpg', '').astype(int)
    
    # Merge the dataframes on ImageID
    merged_df = labelled_df.merge(train_df[['ImageID', 'distance']], on='ImageID', how='left')
    
    # Reorder columns and drop the temporary ImageID column
    final_df = merged_df[['filename', 'x', 'y', 'distance']]
    
    print(f"Merged dataframe shape: {final_df.shape}")
    print(f"First few rows:")
    print(final_df.head())
    
    # Check for any missing distances
    missing_distances = final_df['distance'].isna().sum()
    print(f"Missing distances: {missing_distances}")
    
    # Save the updated labelled_data.csv
    final_df.to_csv('data/labelled _data.csv', index=False)
    print("Successfully updated labelled _data.csv with distances column!")

if __name__ == "__main__":
    add_distances_to_labelled_data()