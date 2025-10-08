# Output Normalization with MinMaxScaler - Documentation

## Overview
This document explains the output normalization feature added to the ISS Docking project using scikit-learn's MinMaxScaler for coordinates and distances.

## Why Output Normalization?

### Benefits:
1. **Faster Training**: Neural networks train faster when all outputs are on the same scale
2. **Improved Convergence**: Normalized outputs [0, 1] help the model converge more reliably
3. **Balanced Learning**: Prevents one output (e.g., distance) from dominating the loss function
4. **Better Gradients**: Ensures stable gradients during backpropagation

### What Gets Normalized:
- **X coordinates** (pixel values)
- **Y coordinates** (pixel values)  
- **Distance** (meters) - if available

All outputs are scaled to the range [0, 1] using MinMaxScaler.

---

## New Functions Added

### In `helpers.py`:

#### 1. `create_output_scaler(train_df, output_columns, scaler_path)`
Creates and fits a MinMaxScaler on training data only.

```python
from helpers import create_output_scaler

scaler = create_output_scaler(
    train_df,
    output_columns=['x', 'y', 'distance'],
    scaler_path='data/output_scaler.pkl'
)
```

**Important**: The scaler is ONLY fitted on training data to prevent data leakage!

---

#### 2. `load_output_scaler(scaler_path)`
Loads a previously saved scaler for inference.

```python
from helpers import load_output_scaler

scaler = load_output_scaler('data/output_scaler.pkl')
```

---

#### 3. `normalize_outputs(data, output_columns, scaler)`
Normalizes output values to [0, 1] range.

```python
from helpers import normalize_outputs

normalized = normalize_outputs(
    data=val_df,
    output_columns=['x', 'y', 'distance'],
    scaler=scaler
)
```

---

#### 4. `denormalize_outputs(normalized_data, scaler)`
Converts normalized predictions back to original scale.

```python
from helpers import denormalize_outputs

# After model prediction
predictions_normalized = model.predict(X_test)  # Range: [0, 1]

# Convert back to original scale
predictions_original = denormalize_outputs(predictions_normalized, scaler)
# predictions_original now has:
#   - x, y in pixels (e.g., 0-640)
#   - distance in meters (e.g., 0-400)
```

---

### In `data_split.py`:

#### `normalize_output_splits(train_df, val_df, test_df, output_columns, scaler_path)`
Applies MinMaxScaler normalization to all data splits.

```python
train_norm, val_norm, test_norm, scaler = normalize_output_splits(
    train_df, val_df, test_df,
    output_columns=['x', 'y', 'distance'],
    scaler_path='data/output_scaler.pkl'
)
```

**Features**:
- Preserves original values in `*_original` columns
- Fits scaler only on training data
- Applies same transformation to val and test sets
- Saves scaler for later use

---

## Complete Pipeline

### Step 1: Process Data (Run `data_split.py`)

```bash
python data_split.py
```

This will:
1. ✅ Resize images to 224x224
2. ✅ Apply ImageNet normalization to images
3. ✅ Scale coordinates proportionally
4. ✅ Split data (80% train, 10% val, 10% test)
5. ✅ **Apply MinMaxScaler to outputs**
6. ✅ Save normalized splits to CSV files
7. ✅ Save scaler to `data/output_scaler.pkl`

### Generated Files:
```
data/
├── train_split.csv              # Normalized coordinates & distances
├── val_split.csv                # Normalized coordinates & distances
├── test_split.csv               # Normalized coordinates & distances
├── train_split_original.csv     # Original scale (backup)
├── val_split_original.csv       # Original scale (backup)
├── test_split_original.csv      # Original scale (backup)
└── output_scaler.pkl            # Fitted MinMaxScaler
```

---

## Training Workflow

### 1. Load Normalized Data

```python
import pandas as pd
from helpers import load_output_scaler

# Load normalized training data
train_df = pd.read_csv('data/train_split.csv')

# Prepare inputs and outputs
# X_train = load_images(train_df['filename'])  # Your image loading code
y_train = train_df[['x', 'y', 'distance']].values  # Already normalized [0, 1]

print(f"Output range: [{y_train.min():.4f}, {y_train.max():.4f}]")
# Output range: [0.0000, 1.0000]
```

### 2. Train Model

```python
# Train with normalized outputs
model.compile(
    optimizer='adam',
    loss='mse',  # Works well with normalized outputs
    metrics=['mae']
)

model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
```

### 3. Make Predictions

```python
# Model predicts normalized values [0, 1]
predictions_normalized = model.predict(X_test)
```

### 4. Denormalize Predictions

```python
from helpers import load_output_scaler, denormalize_outputs

# Load the scaler
scaler = load_output_scaler('data/output_scaler.pkl')

# Convert predictions back to original scale
predictions_original = denormalize_outputs(predictions_normalized, scaler)

# Now you have:
# predictions_original[:, 0] = x coordinates in pixels
# predictions_original[:, 1] = y coordinates in pixels  
# predictions_original[:, 2] = distance in meters
```

---

## Example: Complete Training Script

```python
import pandas as pd
import numpy as np
from tensorflow import keras
from helpers import load_output_scaler, denormalize_outputs

# Load normalized data
train_df = pd.read_csv('data/train_split.csv')
val_df = pd.read_csv('data/val_split.csv')
test_df = pd.read_csv('data/test_split.csv')

# Prepare normalized outputs (already in [0, 1])
y_train = train_df[['x', 'y', 'distance']].values
y_val = val_df[['x', 'y', 'distance']].values
y_test = test_df[['x', 'y', 'distance']].values

# Load images (your implementation)
# X_train, X_val, X_test = load_images(...)

# Build and train model
model = build_model()  # Your model architecture
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

# Evaluate on test set
predictions_normalized = model.predict(X_test)

# Denormalize predictions
scaler = load_output_scaler('data/output_scaler.pkl')
predictions_original = denormalize_outputs(predictions_normalized, scaler)
y_test_original = denormalize_outputs(y_test, scaler)

# Calculate metrics on original scale
mae_x = np.mean(np.abs(predictions_original[:, 0] - y_test_original[:, 0]))
mae_y = np.mean(np.abs(predictions_original[:, 1] - y_test_original[:, 1]))
mae_dist = np.mean(np.abs(predictions_original[:, 2] - y_test_original[:, 2]))

print(f"MAE X (pixels): {mae_x:.2f}")
print(f"MAE Y (pixels): {mae_y:.2f}")
print(f"MAE Distance (meters): {mae_dist:.2f}")
```

---

## Verification & Testing

Run the example script to verify everything works:

```bash
python scaler_usage_example.py
```

This will show:
1. ✅ How to use normalized data for training
2. ✅ How to denormalize predictions
3. ✅ Comparison of normalized vs original values
4. ✅ Verification that denormalization is accurate

---

## Important Notes

### ⚠️ Data Leakage Prevention
- The scaler is **ONLY fitted on training data**
- The same scaler transformation is applied to validation and test sets
- Never fit a new scaler on validation or test data!

### 💾 Scaler Persistence
- The scaler is saved to `data/output_scaler.pkl`
- **Always use the same scaler** for training and inference
- Keep the scaler file with your trained model

### 🔄 Normalization Order
1. **Images**: ImageNet normalization (done during image processing)
2. **Outputs**: MinMaxScaler normalization (done during data splitting)

### 📊 Data Files
- **`*_split.csv`**: Use these for training (normalized outputs)
- **`*_split_original.csv`**: Backup files with original values
- **`output_scaler.pkl`**: Required for denormalization

---

## Troubleshooting

### Q: Model predictions are all between 0 and 1?
**A**: That's correct! The model is trained on normalized outputs. Use `denormalize_outputs()` to convert back.

### Q: Should I normalize the test set differently?
**A**: No! Always use the scaler fitted on training data for all datasets.

### Q: Can I use a different normalization range?
**A**: MinMaxScaler uses [0, 1] by default. You can change this by modifying the scaler's `feature_range` parameter.

### Q: What if I only have coordinates (no distance)?
**A**: The code automatically detects available columns. Just use `output_columns=['x', 'y']`.

---

## Summary

✅ **What you get**:
- Normalized outputs in [0, 1] range for better training
- Automatic handling of coordinates and distances
- Easy denormalization of predictions
- No data leakage (scaler fitted only on training data)

✅ **How to use**:
1. Run `python data_split.py` to generate normalized data
2. Train your model using the normalized splits
3. Use `denormalize_outputs()` to convert predictions back

✅ **Files to keep**:
- `data/output_scaler.pkl` - Required for inference
- `data/*_split.csv` - Normalized data for training
- `data/*_split_original.csv` - Original values for reference
