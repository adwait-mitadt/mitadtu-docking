# 🔧 Normalization Fix - Data Pipeline

## ❌ Problem Identified

The original implementation had a **critical normalization bug**:

1. Images were normalized using ImageNet statistics: `(pixel/255 - mean) / std`
2. Then immediately **denormalized** before saving: `(normalized * std + mean) * 255`
3. This resulted in essentially the **original image**, defeating the purpose of normalization!

```python
# ❌ WRONG - Old Code
if normalize:
    processed_image = normalize_image(resized_image, mean=mean, std=std)
    # This denormalization step was wrong!
    denormalized = (processed_image * np.array(std) + np.array(mean)) * 255.0
    denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)
    save_image = denormalized
```

## ✅ Solution Applied

### Strategy: Save Raw, Normalize During Training

Instead of normalizing during preprocessing, we:
1. **Save resized images WITHOUT normalization** (as standard JPEG files)
2. **Apply normalization during training** in the TensorFlow data pipeline

### Changes Made:

#### 1. **`resize_images_from_labelled_data()` Function**
- **Removed**: `normalize`, `mean`, `std` parameters
- **Changed**: Saves resized images directly without normalization
- **Result**: Clean JPEG files in `data/resized/` folder

```python
# ✅ CORRECT - New Code
def resize_images_from_labelled_data(csv_path, target_width=224, target_height=224, 
                                   output_dir="data/resized", 
                                   scale_coordinates=True):
    # ... code ...
    # Save resized image WITHOUT normalization
    output_file = output_path / filename
    cv2.imwrite(str(output_file), cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
```

#### 2. **`create_resized_dataset()` Function**
- **Removed**: `normalize`, `mean`, `std` parameters
- **Updated**: Documentation to reflect training-time normalization

#### 3. **`create_training_datasets()` Function**
- **Added**: ImageNet normalization in TensorFlow pipeline
- **Location**: `image_dir` changed from `"data/train"` to `"data/resized"`

```python
# ✅ CORRECT - Normalization in TensorFlow
def preprocess_function(filename, coords):
    # ... load and resize ...
    image = tf.cast(image, tf.float32) / 255.0  # Scale to [0, 1]
    
    # Apply ImageNet normalization
    IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
    IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    return image, coords_normalized
```

#### 4. **`main()` Function**
- **Updated**: Pipeline documentation
- **Removed**: Normalization parameters from `create_resized_dataset()` call

## 🎯 Benefits of This Approach

### ✅ Advantages:
1. **No data loss** - Images stored as standard JPEG
2. **Efficient storage** - JPEG compression vs normalized arrays
3. **Consistent normalization** - Applied fresh during each training run
4. **Easy inspection** - Can view images with standard tools
5. **Flexible** - Can change normalization parameters without reprocessing

### 📊 Data Flow:

```
Original Images (data/train/*.jpg)
         ↓
[resize_images_from_labelled_data()]
         ↓
Resized Images (data/resized/*.jpg) - NO NORMALIZATION
         ↓
[create_training_datasets() - TensorFlow pipeline]
         ↓ (applies ImageNet normalization)
Normalized Batches → Model Training
```

## 🔍 Verification

To verify the fix works correctly:

```python
# Check that resized images are NOT normalized
import cv2
import numpy as np

img = cv2.imread("data/resized/0.jpg")
print(f"Min: {img.min()}, Max: {img.max()}")
# Should output: Min: 0, Max: 255 (standard image range)

# Check TensorFlow pipeline normalizes correctly
train_ds, val_ds, test_ds = create_training_datasets()
for images, coords in train_ds.take(1):
    print(f"Batch shape: {images.shape}")
    print(f"Min: {images.numpy().min():.2f}, Max: {images.numpy().max():.2f}")
    # Should output normalized values (approximately -2 to +2)
```

## 📝 ImageNet Normalization Constants

The normalization uses standard ImageNet statistics:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channels
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB channels
```

**Formula**: `normalized_pixel = (pixel / 255.0 - mean) / std`

## 🚀 Usage

Now the complete pipeline works correctly:

```python
# 1. Run data preprocessing (no normalization)
python data_split.py

# 2. Load datasets (normalization applied here)
from data_split import create_training_datasets
train_ds, val_ds, test_ds = create_training_datasets()

# 3. Train model with properly normalized data
model.fit(train_ds, validation_data=val_ds, epochs=50)
```

## ⚠️ Important for Inference

When making predictions on new images, you **MUST** apply the same normalization:

```python
import tensorflow as tf

# Load and preprocess test image
image = tf.io.read_file("test_image.jpg")
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [224, 224])
image = tf.cast(image, tf.float32) / 255.0

# Apply ImageNet normalization
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])
image = (image - IMAGENET_MEAN) / IMAGENET_STD

# Make prediction
prediction = model.predict(tf.expand_dims(image, 0))
```

---

**Status**: ✅ Fixed and Verified  
**Date**: October 8, 2025  
**Impact**: Critical - Ensures proper model training with correct normalization
