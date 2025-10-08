import pandas as pd
import ast

def convert_csv_to_labels(input_file, output_file):
    """
    Convert CSV file with ImageID and location to labels format.
    
    Args:
        input_file: Path to input CSV (e.g., train_split.csv)
        output_file: Path to output CSV (e.g., train_labels.csv)
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Create filename column by adding .jpg to ImageID
    df['filename'] = df['ImageID'].astype(str) + '.jpg'
    
    # Parse the location column and extract x, y
    # Location is in format "[x, y]"
    locations = df['location'].apply(ast.literal_eval)
    df['x'] = locations.apply(lambda loc: loc[0])
    df['y'] = locations.apply(lambda loc: loc[1])
    
    # Select only the required columns
    result_df = df[['filename', 'x', 'y']]
    
    # Save to output file
    result_df.to_csv(output_file, index=False)
    print(f"Converted {input_file} to {output_file}")
    print(f"Total rows: {len(result_df)}")
    print(f"\nFirst few rows:")
    print(result_df.head())

# Convert train_split.csv to train_labels.csv
convert_csv_to_labels('data/train_split.csv', 'data/train_labels.csv')

print("\n" + "="*50 + "\n")

# Convert validation_split.csv to val_labels.csv
convert_csv_to_labels('data/validation_split.csv', 'data/val_labels.csv')
