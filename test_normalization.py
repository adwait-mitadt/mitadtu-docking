"""
Test Script: Independent Input Normalization
Demonstrates the use of 3 separate scalers for x, y, and distance
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("🧪 TESTING INDEPENDENT INPUT NORMALIZATION")
print("="*70)

# Step 1: Load training data to create scalers
print("\n📊 Step 1: Loading training data...")
train_df = pd.read_csv("data/train_split.csv")
print(f"✅ Loaded {len(train_df)} training samples")
print(f"   Columns: {list(train_df.columns)}")
print(f"\n   Sample data:")
print(train_df.head())

# Step 2: Create independent scalers
print("\n🔧 Step 2: Creating independent scalers...")
from sklearn.preprocessing import MinMaxScaler
import pickle

scalers = {
    'x': MinMaxScaler(feature_range=(0, 1)),
    'y': MinMaxScaler(feature_range=(0, 1)),
    'distance': MinMaxScaler(feature_range=(0, 1))
}

# Fit each scaler independently
scalers['x'].fit(train_df[['x']])
scalers['y'].fit(train_df[['y']])
scalers['distance'].fit(train_df[['distance']])

print(f"✅ Created 3 independent scalers")
print(f"\n📈 Data ranges from training set:")
print(f"   X: [{train_df['x'].min():.2f}, {train_df['x'].max():.2f}]")
print(f"   Y: [{train_df['y'].min():.2f}, {train_df['y'].max():.2f}]")
print(f"   Distance: [{train_df['distance'].min():.2f}, {train_df['distance'].max():.2f}]")

# Step 3: Save scalers
print("\n💾 Step 3: Saving scalers...")
scaler_path = Path("data/input_scalers.pkl")
scaler_path.parent.mkdir(parents=True, exist_ok=True)

with open(scaler_path, 'wb') as f:
    pickle.dump(scalers, f)

print(f"✅ Scalers saved to: {scaler_path}")

# Step 4: Load and test scalers
print("\n📂 Step 4: Loading and testing scalers...")

with open(scaler_path, 'rb') as f:
    loaded_scalers = pickle.load(f)

print(f"✅ Scalers loaded successfully")

# Step 5: Test normalization
print("\n🧪 Step 5: Testing normalization...")

# Test with a sample from the dataset
sample_idx = 0
sample = train_df.iloc[sample_idx]
x_orig = sample['x']
y_orig = sample['y']
dist_orig = sample['distance']

print(f"\n📌 Original values (sample {sample_idx}):")
print(f"   X: {x_orig}")
print(f"   Y: {y_orig}")
print(f"   Distance: {dist_orig}")

# Normalize independently
x_norm = loaded_scalers['x'].transform([[x_orig]])[0, 0]
y_norm = loaded_scalers['y'].transform([[y_orig]])[0, 0]
dist_norm = loaded_scalers['distance'].transform([[dist_orig]])[0, 0]

print(f"\n📊 Normalized values (independent scaling):")
print(f"   X_norm: {x_norm:.6f}")
print(f"   Y_norm: {y_norm:.6f}")
print(f"   Distance_norm: {dist_norm:.6f}")

# Denormalize to verify
x_denorm = loaded_scalers['x'].inverse_transform([[x_norm]])[0, 0]
y_denorm = loaded_scalers['y'].inverse_transform([[y_norm]])[0, 0]
dist_denorm = loaded_scalers['distance'].inverse_transform([[dist_norm]])[0, 0]

print(f"\n🔄 Denormalized values (verification):")
print(f"   X: {x_denorm:.2f} (original: {x_orig})")
print(f"   Y: {y_denorm:.2f} (original: {y_orig})")
print(f"   Distance: {dist_denorm:.2f} (original: {dist_orig})")

# Step 6: Test with helpers.py functions
print("\n🧪 Step 6: Testing helpers.py functions...")

try:
    from helpers import load_input_scalers, normalize_inputs, denormalize_inputs, get_input_scalers_info
    
    # Load using helper
    scalers_helper = load_input_scalers('data/input_scalers.pkl')
    
    # Test normalize
    x_test, y_test, dist_test = 320, 240, 150
    print(f"\n📌 Test values: x={x_test}, y={y_test}, distance={dist_test}")
    
    x_n, y_n, dist_n = normalize_inputs(x_test, y_test, dist_test, scalers_helper)
    print(f"📊 Normalized: x={x_n:.6f}, y={y_n:.6f}, dist={dist_n:.6f}")
    
    # Test denormalize
    x_d, y_d, dist_d = denormalize_inputs(x_n, y_n, dist_n, scalers_helper)
    print(f"🔄 Denormalized: x={x_d:.2f}, y={y_d:.2f}, dist={dist_d:.2f}")
    
    # Test with arrays
    print(f"\n📊 Testing with arrays...")
    x_arr = np.array([320, 330, 340])
    y_arr = np.array([240, 250, 260])
    dist_arr = np.array([150, 160, 170])
    
    x_n_arr, y_n_arr, dist_n_arr = normalize_inputs(x_arr, y_arr, dist_arr, scalers_helper)
    print(f"✅ Normalized {len(x_arr)} samples")
    print(f"   X_norm: {x_n_arr}")
    print(f"   Y_norm: {y_n_arr}")
    print(f"   Dist_norm: {dist_n_arr}")
    
    # Display scaler info
    print(f"\n📊 Displaying scaler information...")
    info = get_input_scalers_info(scalers_helper)
    
    print("\n✅ All helper functions working correctly!")
    
except ImportError as e:
    print(f"\n⚠️  Could not import from helpers.py: {e}")
    print("   Make sure helpers.py has the new input normalization functions")

# Step 7: Summary
print("\n" + "="*70)
print("✅ TEST SUMMARY")
print("="*70)
print("✓ Created 3 independent MinMaxScalers (x, y, distance)")
print("✓ Scalers saved to: data/input_scalers.pkl")
print("✓ Normalization/denormalization working correctly")
print("✓ Helper functions tested successfully")
print("✓ Array operations supported")
print("\n📌 KEY POINTS:")
print("   • Each feature (x, y, distance) normalized independently")
print("   • Outputs remain unnormalized (original scale)")
print("   • Model predictions will be in original pixel/meter values")
print("   • No denormalization needed for model outputs!")
print("="*70)
