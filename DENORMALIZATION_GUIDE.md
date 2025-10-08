# Denormalization Guide - MinMaxScaler

## Overview
This guide explains how to use the MinMaxScaler normalization and denormalization functions in `helpers.py` for your ISS docking project.

## Available Functions

### 1. `create_output_scaler(outputs, feature_range=(0, 1))`
Creates and fits a MinMaxScaler on your training outputs.

**Parameters:**
- `outputs`: numpy array of shape (n_samples, n_features), e.g., (N, 3) for [x, y, distance]
- `feature_range`: tuple, desired range of transformed data (default: (0, 1))

**Returns:**
- Fitted MinMaxScaler object

**Example:**
```python
from helpers import create_output_scaler

train_outputs = np.array([[100, 200, 50.5], [150, 250, 75.2]])
scaler = create_output_scaler(train_outputs)
```

---

### 2. `normalize_outputs(outputs, scaler=None, feature_range=(0, 1))`
Normalizes outputs using MinMaxScaler.

**Parameters:**
- `outputs`: numpy array to normalize
- `scaler`: (optional) Pre-fitted scaler. If None, creates and fits a new one
- `feature_range`: tuple, desired range (default: (0, 1))

**Returns:**
- Tuple of (normalized_outputs, scaler)

**Example:**
```python
from helpers import normalize_outputs

# Automatically creates and fits scaler
normalized_outputs, scaler = normalize_outputs(train_outputs)
```

---

### 3. `denormalize_outputs(normalized_outputs, scaler)` ⭐
**THIS IS THE KEY FUNCTION FOR DENORMALIZATION**

Converts normalized predictions back to original scale.

**Parameters:**
- `normalized_outputs`: numpy array of normalized predictions (from your model)
- `scaler`: The fitted scaler used during normalization

**Returns:**
- numpy array of denormalized outputs in original scale

**Example:**
```python
from helpers import denormalize_outputs

# After model prediction
predictions_norm = model.predict(X_test)  # Shape: (N, 3)
predictions_original = denormalize_outputs(predictions_norm, scaler)
# Now predictions_original has [x, y, distance] in original pixel/meter values
```

---

### 4. `save_output_scaler(scaler, filepath='data/output_scaler.pkl')`
Saves a fitted scaler to disk for later use.

**Parameters:**
- `scaler`: Fitted MinMaxScaler object
- `filepath`: Path to save the scaler (default: 'data/output_scaler.pkl')

**Returns:**
- String path where scaler was saved

**Example:**
```python
from helpers import save_output_scaler

save_output_scaler(scaler, 'models/my_scaler.pkl')
```

---

### 5. `load_output_scaler(filepath='data/output_scaler.pkl')`
Loads a saved scaler from disk.

**Parameters:**
- `filepath`: Path to the saved scaler file

**Returns:**
- Loaded MinMaxScaler object

**Example:**
```python
from helpers import load_output_scaler

scaler = load_output_scaler('models/my_scaler.pkl')
```

---

### 6. `get_scaler_info(scaler)`
Displays information about a fitted scaler.

**Parameters:**
- `scaler`: Fitted MinMaxScaler object

**Returns:**
- Dictionary containing scaler information

**Example:**
```python
from helpers import get_scaler_info

info = get_scaler_info(scaler)
# Prints: feature range, n_features, data_min, data_max, etc.
```

---

## Complete Workflow Example

### Training Phase
```python
from helpers import normalize_outputs, save_output_scaler
import numpy as np

# 1. Load your training data
# Assuming you have coordinates and distances: [x, y, distance]
train_outputs = np.array([
    [100, 200, 50.5],
    [150, 250, 75.2],
    [200, 300, 100.8],
    # ... more training samples
])

# 2. Normalize outputs
normalized_outputs, scaler = normalize_outputs(train_outputs)

# 3. Train your model with normalized outputs
model.fit(train_images, normalized_outputs)

# 4. Save the scaler for later use during inference
save_output_scaler(scaler, 'models/output_scaler.pkl')

# 5. Save your trained model
model.save('models/my_model.h5')
```

### Inference Phase (Prediction & Denormalization)
```python
from helpers import load_output_scaler, denormalize_outputs
from tensorflow import keras

# 1. Load your trained model
model = keras.models.load_model('models/my_model.h5')

# 2. Load the scaler (IMPORTANT!)
scaler = load_output_scaler('models/output_scaler.pkl')

# 3. Prepare test images
test_images = load_and_preprocess_test_images()

# 4. Make predictions (these will be in normalized space 0-1)
predictions_normalized = model.predict(test_images)
# Shape: (n_samples, 3) with values in [0, 1]

# 5. Denormalize to get original scale ⭐
predictions_original = denormalize_outputs(predictions_normalized, scaler)
# Shape: (n_samples, 3) with values in original [x, y, distance] scale

# 6. Use the predictions
for i, pred in enumerate(predictions_original):
    x, y, distance = pred
    print(f"Image {i}: Position=({x:.1f}, {y:.1f}), Distance={distance:.1f}m")
```

---

## Quick Reference

| Task | Function |
|------|----------|
| Normalize training outputs | `normalize_outputs(outputs)` |
| Denormalize predictions ⭐ | `denormalize_outputs(predictions, scaler)` |
| Save scaler | `save_output_scaler(scaler, filepath)` |
| Load scaler | `load_output_scaler(filepath)` |
| View scaler info | `get_scaler_info(scaler)` |

---

## Important Notes

1. **Always save your scaler** after training - you need it for denormalization!
2. **Use the same scaler** for denormalization that was used for normalization
3. **Scaler must be fitted** on training data only, not validation or test data
4. The scaler file is small (few KB) - always keep it with your model
5. For predictions, the workflow is: `model.predict() → denormalize_outputs() → original scale`

---

## Common Errors and Solutions

### Error: "Scaler cannot be None"
**Solution:** Load the scaler first using `load_output_scaler()`

### Error: "FileNotFoundError: Scaler file not found"
**Solution:** Make sure you saved the scaler during training, check the filepath

### Wrong output values
**Solution:** Ensure you're using the same scaler that was fitted during training

---

## Example Output

```
📊 Original Training Outputs:
[[100.  200.   50.5]
 [150.  250.   75.2]]

✅ Normalized Outputs (0-1 range):
[[0.    0.    0.   ]
 [1.    1.    1.   ]]

🤖 Model Predictions (normalized):
[[0.5 0.5 0.5]]

🎯 Denormalized Predictions (original scale):
[[125.  225.  62.85]]
```

This means: x=125 pixels, y=225 pixels, distance=62.85 meters
