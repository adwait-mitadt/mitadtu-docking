# 🎯 Complete Normalization & Denormalization Guide

## Overview

This guide explains the complete normalization/denormalization pipeline for the ISS Docking project with **3 independent output variables**: `x`, `y`, and `distance`.

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT (Images)                    OUTPUT (x, y, distance)      │
│  ─────────────                     ──────────────────────       │
│                                                                  │
│  Raw Image (0-255)                 Raw Values:                  │
│         ↓                          - x: pixels (0-448)          │
│  Resize to 224x224                 - y: pixels (0-448)          │
│         ↓                          - distance: meters (0-400)   │
│  Scale to [0,1]: /255                      ↓                    │
│         ↓                          Create 3 scalers             │
│  ImageNet normalize:                       ↓                    │
│  (pixel - mean) / std              Fit on training data         │
│         ↓                                  ↓                    │
│  ┌─────────────┐                   Normalize each:             │
│  │   ResNet50  │                   - x_scaler: x → [0,1]       │
│  │   (frozen)  │                   - y_scaler: y → [0,1]       │
│  │      ↓      │                   - dist_scaler: dist → [0,1] │
│  │   Dense     │                          ↓                    │
│  │   Dropout   │                   Save scalers.pkl            │
│  │      ↓      │                          ↓                    │
│  │  Output(3)  │ ←────────────────  Normalized [0,1]          │
│  └─────────────┘                                                │
│         ↓                                                        │
│  Loss: RMSE on normalized values                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  New Image                         Load scalers.pkl             │
│      ↓                                    ↓                     │
│  Same preprocessing                 ┌──────────┐                │
│  (ImageNet normalization)           │  Model   │                │
│      ↓                              │ Predicts │                │
│  ┌─────────┐                        │  [0,1]   │                │
│  │  Model  │ ───────────────────→   └──────────┘                │
│  └─────────┘                              ↓                     │
│      ↓                           Denormalize each:              │
│  Output [0,1]                    - x: pred/x_scale + x_min      │
│      ↓                           - y: pred/y_scale + y_min      │
│  Denormalize using scalers       - dist: pred/d_scale + d_min   │
│      ↓                                   ↓                      │
│  Original scale:                 Final predictions:             │
│  - x: pixels                     - x: 150.5 pixels              │
│  - y: pixels                     - y: 220.3 pixels              │
│  - distance: meters              - distance: 85.7 meters        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Details

### 1. **Input Normalization (Images)**

**Location**: Applied in `create_training_datasets()` and `create_normalized_datasets()`

```python
# Step 1: Scale to [0, 1]
image = image / 255.0

# Step 2: ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB
image = (image - IMAGENET_MEAN) / IMAGENET_STD
```

**Why**: ResNet50 was pre-trained on ImageNet with these statistics.

**Result**: Pixel values typically in range `[-2.5, 2.5]`

---

### 2. **Output Normalization (x, y, distance)**

**Location**: `helpers.py` - `create_output_scalers()` and `normalize_outputs()`

#### 2.1 Creating Scalers

```python
from helpers import create_output_scalers

# Training data outputs
train_outputs = train_df[['x', 'y', 'distance']].values
# Shape: (n_samples, 3)

# Create 3 independent MinMaxScalers
scalers = create_output_scalers(train_outputs, feature_range=(0, 1))
```

**Result**: Dictionary with 3 scalers:
```python
{
    'x': MinMaxScaler fitted on x coordinates,
    'y': MinMaxScaler fitted on y coordinates,
    'distance': MinMaxScaler fitted on distances
}
```

#### 2.2 Normalization Formula

For each output independently:

```python
# MinMax normalization
normalized_value = (value - min_value) * scale_factor

# Where:
# scale_factor = 1 / (max_value - min_value)
```

**Example**:
```python
# X coordinate
x_min = 0.0
x_max = 448.0
x_range = 448.0
x_scale = 1 / 448.0 = 0.00223

# For x = 150:
x_normalized = (150 - 0) * 0.00223 = 0.335
```

#### 2.3 Why 3 Independent Scalers?

**Critical Reason**: Each output has a different scale:
- **x**: 0-448 pixels
- **y**: 0-448 pixels  
- **distance**: 0-400 meters

If we used a single scaler:
- ❌ Distance (0-400) would dominate
- ❌ x and y would be compressed
- ❌ Model would prioritize distance over coordinates

With independent scalers:
- ✅ Each output normalized to [0, 1] independently
- ✅ Equal importance during training
- ✅ Better gradient flow
- ✅ Improved model performance

---

### 3. **Output Denormalization**

**Location**: `helpers.py` - `denormalize_outputs()`

#### 3.1 Denormalization Formula

```python
# Inverse MinMax
original_value = normalized_value / scale_factor + min_value
```

**Example**:
```python
# For x_normalized = 0.335:
x_original = 0.335 / 0.00223 + 0 = 150.0 pixels
```

#### 3.2 Usage in Inference

```python
from helpers import load_output_scalers, denormalize_outputs

# Load saved scalers
scalers = load_output_scalers('models/output_scalers.pkl')

# Model predicts normalized values [0, 1]
predictions_norm = model.predict(images)
# Shape: (n_samples, 3), values in [0, 1]

# Denormalize to original scale
predictions_original = denormalize_outputs(predictions_norm, scalers)
# Shape: (n_samples, 3), values in original units
```

---

## 🚀 Complete Usage Example

### Training

```python
from train_with_normalization import main

# This script:
# 1. Loads training data
# 2. Creates 3 independent scalers for x, y, distance
# 3. Normalizes outputs to [0, 1]
# 4. Trains model on normalized data
# 5. Saves model AND scalers
main()
```

### Inference

```python
from inference import predict_single_image
from helpers import load_output_scalers
import tensorflow as tf

# Load model and scalers
model = tf.keras.models.load_model('models/resnet_docking.h5')
scalers = load_output_scalers('models/output_scalers.pkl')

# Predict on new image
result = predict_single_image(model, scalers, 'test_image.jpg')

print(f"X: {result['x']:.2f} pixels")
print(f"Y: {result['y']:.2f} pixels")
print(f"Distance: {result['distance']:.2f} meters")
```

---

## 📈 Data Flow Summary

### Training Data Flow

```python
# 1. Raw data
x_raw = 150 pixels
y_raw = 220 pixels
distance_raw = 85.5 meters

# 2. Fit scalers (done once on training data)
scalers = create_output_scalers(train_outputs)

# 3. Normalize for training
x_norm = (150 - 0) * (1/448) = 0.335
y_norm = (220 - 0) * (1/448) = 0.491
dist_norm = (85.5 - 0) * (1/400) = 0.214

# 4. Model trains on [0.335, 0.491, 0.214]
```

### Inference Data Flow

```python
# 1. Model predicts normalized values
prediction_norm = [0.340, 0.495, 0.210]  # [0, 1] range

# 2. Denormalize using saved scalers
x_pred = 0.340 / (1/448) + 0 = 152.3 pixels
y_pred = 0.495 / (1/448) + 0 = 221.8 pixels
dist_pred = 0.210 / (1/400) + 0 = 84.0 meters

# 3. Final prediction: [152.3, 221.8, 84.0]
```

---

## ⚠️ Critical Rules

### ✅ DO

1. **Fit scalers ONLY on training data**
   ```python
   scalers = create_output_scalers(train_outputs)  # ✅
   ```

2. **Save scalers with the model**
   ```python
   save_output_scalers(scalers, 'models/output_scalers.pkl')  # ✅
   ```

3. **Always denormalize predictions**
   ```python
   predictions_original = denormalize_outputs(predictions_norm, scalers)  # ✅
   ```

4. **Use same scalers for train/val/test**
   ```python
   train_norm, _ = normalize_outputs(train_outputs, scalers)
   val_norm, _ = normalize_outputs(val_outputs, scalers)  # ✅ Same scalers
   ```

### ❌ DON'T

1. **Don't fit scalers on test data**
   ```python
   test_scalers = create_output_scalers(test_outputs)  # ❌ WRONG
   ```

2. **Don't use raw predictions**
   ```python
   # ❌ WRONG - predictions are in [0, 1], not pixels/meters
   print(f"X: {predictions_norm[0, 0]} pixels")  
   
   # ✅ CORRECT - denormalize first
   pred_original = denormalize_outputs(predictions_norm, scalers)
   print(f"X: {pred_original[0, 0]} pixels")
   ```

3. **Don't lose the scalers**
   ```python
   # ❌ WRONG - can't denormalize without scalers
   model.save('model.h5')  
   
   # ✅ CORRECT - save both
   model.save('model.h5')
   save_output_scalers(scalers, 'scalers.pkl')
   ```

---

## 📁 Files Modified/Created

### Modified Files
1. **`data_split.py`**
   - Updated to return 3 outputs (x, y, distance)
   - Removed coordinate normalization (done by scalers now)

2. **`helpers.py`**
   - Enhanced scaler functions with logging
   - Maintained 3 independent scalers

### New Files
3. **`train_with_normalization.py`**
   - Complete training script with proper normalization
   - Creates and saves scalers
   - Evaluation in original scale

4. **`inference.py`**
   - Example inference code
   - Single image and batch prediction
   - Proper denormalization

5. **`COMPLETE_NORMALIZATION_GUIDE.md`** (this file)
   - Comprehensive documentation

---

## 🎯 Verification

### Check Scalers are Working

```python
from helpers import load_output_scalers, get_output_scalers_info

scalers = load_output_scalers('models/output_scalers.pkl')
get_output_scalers_info(scalers)
```

**Expected Output**:
```
📊 Output Scalers Information:
============================================================

🎯 X Scaler:
   Min value: 0.00
   Max value: 448.00
   Range: 448.00
   Scale factor: 0.002232
   Feature range: (0, 1)

🎯 Y Scaler:
   Min value: 0.00
   Max value: 448.00
   Range: 448.00
   Scale factor: 0.002232
   Feature range: (0, 1)

🎯 DISTANCE Scaler:
   Min value: 0.00
   Max value: 400.00
   Range: 400.00
   Scale factor: 0.002500
   Feature range: (0, 1)
============================================================
```

### Verify Round-Trip

```python
# Original values
original = np.array([[150, 220, 85.5]])

# Normalize
normalized, scalers = normalize_outputs(original)

# Denormalize
recovered = denormalize_outputs(normalized, scalers)

# Check
assert np.allclose(original, recovered)  # Should be True
```

---

## 🏆 Benefits of This Approach

1. **Independent Scaling**: Each output treated fairly
2. **Preserved Information**: No data loss
3. **Easy Debugging**: Can inspect normalized and original values
4. **Transfer Learning**: Compatible with pre-trained ResNet50
5. **Reusable**: Scalers can be applied to new data
6. **Interpretable**: Predictions in meaningful units (pixels, meters)

---

## 📞 Quick Reference

### Training
```bash
python train_with_normalization.py
```

### Inference
```python
from inference import predict_single_image
from helpers import load_output_scalers
import tensorflow as tf

model = tf.keras.models.load_model('models/resnet_docking.h5')
scalers = load_output_scalers('models/output_scalers.pkl')
result = predict_single_image(model, scalers, 'image.jpg')
```

### Check Scalers
```python
from helpers import load_output_scalers, get_output_scalers_info
scalers = load_output_scalers('models/output_scalers.pkl')
get_output_scalers_info(scalers)
```

---

**Status**: ✅ Complete and Production-Ready  
**Date**: October 8, 2025  
**Version**: 1.0
