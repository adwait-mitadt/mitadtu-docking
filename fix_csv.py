import pandas as pd

# Read the CSV with error handling
try:
    df = pd.read_csv('logs/training_history.csv', on_bad_lines='skip')
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Keep only first 200 rows (epochs 0-199)
    if len(df) > 200:
        df = df.iloc[:200]
        print(f"Trimmed to {len(df)} rows")
    
    # Save it back
    df.to_csv('logs/training_history.csv', index=False)
    print("✅ CSV file fixed and saved!")
except Exception as e:
    print(f"❌ Error: {e}")
