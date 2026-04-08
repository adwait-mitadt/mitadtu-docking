# ISS Docking Analysis Helper Functions
# Helper functions for ISS docking image analysis and preprocessing

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ast
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

# Setting paths using pathlib
data_path = Path("data/")
target_path = data_path / "train.csv"
inputs_directory_path = data_path / "train"

# Loading the target data
target_data = pd.read_csv(target_path)


def load_image(image_id):
    """
    Load and return the image for given image_id
    
    Args:
        image_id: The ID of the image to load (integer)
        
    Returns:
        numpy.ndarray: The loaded image array
    """
    image_file = inputs_directory_path / f"{image_id}.jpg"
    return plt.imread(image_file)


def load_target_row(image_id):
    """
    Load and return the target data row for given image_id
    
    Args:
        image_id: The ID of the image (integer)
        
    Returns:
        pandas.Series: The row containing target data for the image
    """
    row = target_data[target_data["ImageID"] == image_id]
    return row.iloc[0] if not row.empty else None


def load_coordinates(image_id):
    """
    Load and return the x, y coordinates for given image_id
    
    Args:
        image_id: The ID of the image (integer)
        
    Returns:
        tuple: (x, y) coordinates as floats
    """
    row = load_target_row(image_id)
    if row is None:
        return None
    location = ast.literal_eval(row["location"])
    return location[0], location[1]


def load_distance(image_id):
    """
    Load and return the distance for given image_id
    
    Args:
        image_id: The ID of the image (integer)
        
    Returns:
        float: The distance value
    """
    row = load_target_row(image_id)
    if row is None:
        return None
    return row["distance"]


def show_image(image_id):
    """
    Display an ISS docking image with target crosshair and complete metadata.
    
    Args:
        image_id (int): The ID of the image to display
        
    Returns:
        dict: Complete data about the image (id, distance, x, y, phase)
        
    Usage Examples:
        show_image(0)     # Shows image 0 with all data
        show_image(100)   # Shows image 100 with all data
        show_image(1500)  # Shows image 1500 with all data
    """
    print(f"🚀 ANALYZING IMAGE {image_id}")
    print("="*50)
    
    try:
        # Use helper functions to get all data
        image = load_image(image_id)
        row = load_target_row(image_id)
        
        if row is None:
            print(f"❌ No data found for Image ID: {image_id}")
            return None
        
        # Use helper functions for coordinates and distance
        x, y = load_coordinates(image_id)
        distance = load_distance(image_id)
        
        # Print comprehensive info
        print(f"📷 Image ID: {image_id}")
        print(f"📏 Distance: {distance}m") 
        print(f"🎯 Target: ({x}, {y})")
        
        # Determine phase
        if distance < 100:
            phase = "🔴 FINAL DOCKING"
        elif distance < 200:
            phase = "🟡 FINAL APPROACH" 
        elif distance < 400:
            phase = "🟢 APPROACH"
        else:
            phase = "🔵 LONG RANGE"
        print(f"🚀 Phase: {phase}")
        print("="*50)
        
        # Show image with crosshair
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        
        # Draw crosshair
        size = 25
        plt.plot([x-size, x+size], [y, y], 'r-', linewidth=3, label='Target')
        plt.plot([x, x], [y-size, y+size], 'r-', linewidth=3)
        plt.scatter(x, y, color='red', s=100, marker='o', edgecolors='white', linewidth=2)
        
        # Title and labels
        plt.title(f'ISS Docking | Image {image_id} | Distance: {distance}m | Target: ({x}, {y}) | {phase}')
        plt.xlabel('X Coordinate (pixels)')
        plt.ylabel('Y Coordinate (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print("✅ Analysis complete!")
        
        return {
            'id': image_id, 
            'distance': distance, 
            'x': x, 
            'y': y,
            'phase': phase
        }
        
    except FileNotFoundError:
        print(f"❌ Image file not found for ID: {image_id}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


# Alias for backward compatibility
plot_image_with_distance_crosshair = show_image


def load_image(image_path, convert_to_rgb=True, max_retries=3, retry_delay=0.5):
    """
    Load an image from file path using OpenCV, with retry logic for transient failures.
    
    Args:
        image_path (str or Path): Path to the image file
        convert_to_rgb (bool): Convert from BGR to RGB (default: True)
        max_retries (int): Maximum number of retry attempts on failure (default: 3)
        retry_delay (float): Delay in seconds between retries (default: 0.5)
        
    Returns:
        numpy.ndarray: Loaded image array, or None if loading fails after all retries
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            image = cv2.imread(str(image_path))
            if image is not None:
                if convert_to_rgb:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        except Exception as e:
            last_error = e
        # Image not loaded (None result or exception); retry if attempts remain
        if attempt < max_retries:
            time.sleep(retry_delay)
    if last_error is not None:
        print(f"Error loading image {image_path}: {last_error}")
    else:
        print(f"Warning: Could not read image {image_path} after {max_retries} retries")
    return None


def resize_image(image_array, width, height):
    """
    Resize an image array to specified dimensions using OpenCV.
    
    Args:
        image_array (numpy.ndarray): The input image array to resize
        width (int): Target width in pixels
        height (int): Target height in pixels
        
    Returns:
        numpy.ndarray: Resized image array, or None if input is invalid
    """
    try:
        if image_array is not None:
            return cv2.resize(image_array, (width, height))
        else:
            print("Input image array is None")
            return None
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None


def normalize_image(image_array, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image using ImageNet statistics or custom mean/std values.
    
    Args:
        image_array (numpy.ndarray): Image array with values in [0, 255] or [0, 1]
        mean (list): Mean values for R, G, B channels (default: ImageNet mean)
        std (list): Standard deviation values for R, G, B channels (default: ImageNet std)
        
    Returns:
        numpy.ndarray: Normalized image array
        
    Note:
        - If image is in [0, 255], it will first be scaled to [0, 1]
        - Then normalized using: (image - mean) / std
        - Default values are ImageNet normalization statistics
    """
    try:
        # Convert to float and ensure values are in [0, 1]
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Convert mean and std to numpy arrays
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        
        # Normalize: (image - mean) / std
        normalized_image = (image_array - mean) / std
        
        return normalized_image.astype(np.float32)
    except Exception as e:
        print(f"Error normalizing image: {e}")
        return None


def display_image(image_array, title=None, figsize=(6, 6)):
    """
    Display an image array using matplotlib.
    
    Args:
        image_array (numpy.ndarray): The image array to display
        title (str): Optional title for the image
        figsize (tuple): Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    plt.imshow(image_array)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def load_and_preprocess_images(csv_file, image_folder, target_size=(224, 224), 
                               normalize=True, mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225], max_retries=3, retry_delay=1.0):
    """
    Load, resize, and optionally normalize images from CSV file.
    
    Args:
        csv_file (str): Path to CSV file containing ImageIDs
        image_folder (str): Folder containing the images
        target_size (tuple): Target size for resizing (width, height)
        normalize (bool): Whether to normalize images (default: True)
        mean (list): Mean values for normalization (default: ImageNet mean)
        std (list): Standard deviation for normalization (default: ImageNet std)
        max_retries (int): Maximum retry attempts for failed image loads (default: 3)
        retry_delay (float): Delay in seconds between retries (default: 1.0)
        
    Returns:
        tuple: (images_array, locations_array, distances_array, image_ids)
    """
    # Load the CSV
    data = pd.read_csv(csv_file)
    
    images = []
    locations = []
    distances = []
    image_ids = []
    failed_loads = []
    
    print(f"Loading and preprocessing {len(data)} images from {csv_file}...")
    
    for idx, row in data.iterrows():
        image_id = row['ImageID']
        image_path = f"{image_folder}/{image_id}.jpg"
        
        # Load image with retry logic
        image = load_image(image_path, convert_to_rgb=True, max_retries=max_retries, retry_delay=retry_delay)
        
        if image is not None:
            # Resize to target size
            resized_image = resize_image(image, target_size[0], target_size[1])
            
            # Normalize if requested
            if normalize:
                processed_image = normalize_image(resized_image, mean=mean, std=std)
            else:
                # Just scale to [0, 1] if not normalizing
                processed_image = resized_image / 255.0 if resized_image.max() > 1 else resized_image
            
            images.append(processed_image)
            image_ids.append(image_id)
            
            # Parse location data
            location = row['location']
            if isinstance(location, str):
                location = ast.literal_eval(location)
            locations.append(location)
            
            # Store distance
            distances.append(row['distance'])
        else:
            failed_loads.append(image_id)
            print(f"Warning: Could not load image {image_id} after {max_retries} retries")
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(data)} images...")
    
    print(f"\n✓ Successfully loaded {len(images)} images")
    if failed_loads:
        print(f"✗ Failed to load {len(failed_loads)} images: {failed_loads[:10]}...")
    
    return (
        np.array(images, dtype=np.float32), 
        np.array(locations), 
        np.array(distances), 
        np.array(image_ids)
    )


def process_and_save_image(image_id, image_path, coords, output_folder="data/processed/images",
                          target_size=(224, 224), normalize=True,
                          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Load, preprocess (resize & normalize), and save a single image with its metadata.
    
    Args:
        image_id (int/str): Unique identifier for the image
        image_path (str): Path to the original image
        coords (tuple): (x, y) coordinates of the docking target
        output_folder (str): Folder to save processed images
        target_size (tuple): Target size (width, height) for resizing
        normalize (bool): Whether to normalize the image
        mean (list): Mean values for normalization (RGB channels)
        std (list): Standard deviation for normalization (RGB channels)
    
    Returns:
        dict: Metadata about the processed image
    """
    from pathlib import Path
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = load_image(image_path, convert_to_rgb=True)
    
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    original_size = image.shape[:2]  # (height, width)
    
    # Resize image
    resized_image = resize_image(image, target_size[0], target_size[1])
    
    # Normalize if requested
    if normalize:
        processed_image = normalize_image(resized_image, mean=mean, std=std)
    else:
        # Just scale to [0, 1]
        processed_image = resized_image / 255.0 if resized_image.max() > 1 else resized_image
    
    # Calculate scaled coordinates
    scale_x = target_size[0] / original_size[1]  # width scale
    scale_y = target_size[1] / original_size[0]  # height scale
    scaled_coords = (int(coords[0] * scale_x), int(coords[1] * scale_y))
    
    # Save processed image as .npy file for fast loading
    save_path = output_path / f"{image_id}.npy"
    np.save(save_path, processed_image)
    
    # Create metadata
    metadata = {
        'image_id': image_id,
        'original_size': original_size,
        'processed_size': target_size,
        'original_coords': coords,
        'scaled_coords': scaled_coords,
        'normalized': normalize,
        'save_path': str(save_path)
    }
    
    return metadata


def batch_process_and_save_images(csv_file, image_folder, output_folder="data/processed/images",
                                  metadata_file="data/processed/metadata/metadata.csv",
                                  target_size=(224, 224), normalize=True,
                                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Batch process all images from CSV and save them to the processed folder.
    
    Args:
        csv_file (str): Path to CSV file with columns: filename, x, y
        image_folder (str): Folder containing original images
        output_folder (str): Folder to save processed images
        metadata_file (str): Path to save metadata CSV
        target_size (tuple): Target size (width, height) for resizing
        normalize (bool): Whether to normalize images
        mean (list): Mean values for normalization
        std (list): Standard deviation for normalization
    
    Returns:
        pd.DataFrame: Metadata for all processed images
    """
    from pathlib import Path
    
    # Load CSV
    data = pd.read_csv(csv_file)
    
    # Create output folders
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = Path(metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata_list = []
    successful = 0
    failed = 0
    
    print(f"🚀 Processing {len(data)} images...")
    print(f"   Input: {image_folder}")
    print(f"   Output: {output_folder}")
    print(f"   Target size: {target_size}")
    print(f"   Normalize: {normalize}")
    print("="*60)
    
    for idx, row in data.iterrows():
        # Extract image ID from filename (remove .jpg extension)
        filename = row['filename']
        image_id = filename.replace('.jpg', '')
        
        # Get coordinates
        coords = (row['x'], row['y'])
        
        # Full path to original image
        image_path = f"{image_folder}/{filename}"
        
        # Process and save
        metadata = process_and_save_image(
            image_id=image_id,
            image_path=image_path,
            coords=coords,
            output_folder=output_folder,
            target_size=target_size,
            normalize=normalize,
            mean=mean,
            std=std
        )
        
        if metadata:
            metadata_list.append(metadata)
            successful += 1
            
            if (successful) % 500 == 0:
                print(f"   ✅ Processed {successful} images...")
        else:
            failed += 1
    
    print("="*60)
    print(f"✅ Successfully processed: {successful} images")
    print(f"❌ Failed: {failed} images")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(metadata_file, index=False)
    print(f"📊 Metadata saved to: {metadata_file}")
    
    return metadata_df


def load_processed_images(metadata_file="data/processed/metadata/metadata.csv", 
                         max_images=None):
    """
    Load preprocessed images from the processed folder for direct model training.
    
    Args:
        metadata_file (str): Path to metadata CSV file
        max_images (int): Maximum number of images to load (None = all)
    
    Returns:
        tuple: (images_array, coordinates_array, metadata_df)
            - images_array: np.array of shape (N, H, W, 3)
            - coordinates_array: np.array of shape (N, 2) with (x, y) coords
            - metadata_df: DataFrame with all metadata
    """
    from pathlib import Path
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_file)
    
    if max_images:
        metadata_df = metadata_df.head(max_images)
    
    images = []
    coordinates = []
    
    print(f"📂 Loading {len(metadata_df)} processed images...")
    
    for idx, row in metadata_df.iterrows():
        # Load preprocessed image
        image_path = row['save_path']
        image = np.load(image_path)
        images.append(image)
        
        # Extract scaled coordinates
        coords = ast.literal_eval(row['scaled_coords'])
        coordinates.append(coords)
        
        if (idx + 1) % 500 == 0:
            print(f"   Loaded {idx + 1} images...")
    
    images_array = np.array(images)
    coordinates_array = np.array(coordinates)
    
    print(f"✅ Loaded {len(images_array)} images")
    print(f"   Image shape: {images_array.shape}")
    print(f"   Coordinates shape: {coordinates_array.shape}")
    
    return images_array, coordinates_array, metadata_df


# =============================================
# TENSORFLOW/KERAS MODEL FUNCTIONS
# =============================================

def build_resnet_regression():
    """
    Build ResNet50-based regression model for coordinate prediction.
    
    Returns:
        tensorflow.keras.Model: Compiled ResNet model for (x,y) coordinate prediction
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
        from tensorflow.keras.models import Model
        
        print("🤖 Building ResNet50 regression model...")
        
        input_shape = (224, 224, 3)
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze all layers

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(2, activation="linear")(x)  # Regression output: x, y

        model = Model(inputs=base_model.input, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        print("✅ ResNet50 model built successfully!")
        return model
        
    except ImportError:
        print("❌ TensorFlow not found. Please install: pip install tensorflow")
        return None
    except Exception as e:
        print(f"❌ Error building model: {e}")
        return None

def create_data_generators(train_df, val_df, image_folder, batch_size=32, target_size=(224, 224)):
    """
    Create data generators for training and validation.
    
    Args:
        train_df (pd.DataFrame): Training dataframe with filename, x, y columns
        val_df (pd.DataFrame): Validation dataframe with filename, x, y columns
        image_folder (str): Path to folder containing images
        batch_size (int): Batch size for training
        target_size (tuple): Target image size (width, height)
    
    Returns:
        tuple: (train_generator, val_generator, steps_per_epoch, validation_steps)
    """
    try:
        import tensorflow as tf
        
        def preprocess_function(filename, coords):
            # Load and preprocess image
            image_path = tf.strings.join([str(image_folder) + "/", filename])
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, target_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Normalize coordinates to [0, 1] range based on image size
            coords_normalized = coords / [target_size[0], target_size[1]]
            
            return image, coords_normalized
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            train_df['filename'].values,
            train_df[['x', 'y']].values.astype(np.float32)
        ))
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            val_df['filename'].values,
            val_df[['x', 'y']].values.astype(np.float32)
        ))
        
        # Apply preprocessing and batching
        train_dataset = train_dataset.map(preprocess_function).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(preprocess_function).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        steps_per_epoch = len(train_df) // batch_size
        validation_steps = len(val_df) // batch_size
        
        print(f"✅ Data generators created:")
        print(f"   Training batches: {steps_per_epoch}")
        print(f"   Validation batches: {validation_steps}")
        
        return train_dataset, val_dataset, steps_per_epoch, validation_steps
        
    except ImportError:
        print("❌ TensorFlow not found. Please install: pip install tensorflow")
        return None, None, 0, 0
    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        return None, None, 0, 0


def create_single_dataset(df, image_folder, batch_size=32, target_size=(224, 224)):
    """
    Create a single TensorFlow dataset from DataFrame.
    
    Args:
        df (pd.DataFrame): Dataframe with filename, x, y columns
        image_folder (str): Path to folder containing images
        batch_size (int): Batch size
        target_size (tuple): Target image size (width, height)
    
    Returns:
        tf.data.Dataset: Preprocessed TensorFlow dataset
    """
    try:
        import tensorflow as tf
        
        def preprocess_function(filename, coords):
            # Load and preprocess image
            image_path = tf.strings.join([str(image_folder) + "/", filename])
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, target_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Normalize coordinates to [0, 1] range based on image size
            coords_normalized = coords / [target_size[0], target_size[1]]
            
            return image, coords_normalized
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            df['filename'].values,
            df[['x', 'y']].values.astype(np.float32)
        ))
        
        # Apply preprocessing and batching
        dataset = dataset.map(preprocess_function).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    except ImportError:
        print("❌ TensorFlow not found. Please install: pip install tensorflow")
        return None
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return None


def create_docking_datasets(train_df, val_df, image_folder, batch_size=32, 
                            target_size=(224, 224), normalize_by=512.0, augment=True):
    """
    Create TensorFlow datasets for ISS docking (x, y, distance regression) with optional augmentation.
    
    Args:
        train_df (pd.DataFrame): Training dataframe with filename, x, y, distance columns
        val_df (pd.DataFrame): Validation dataframe with filename, x, y, distance columns
        image_folder (str): Path to folder containing images
        batch_size (int): Batch size for training
        target_size (tuple): Target image size (width, height)
        normalize_by (float): Value to normalize coordinates and distance by
        augment (bool): Whether to apply data augmentation to training set
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    try:
        import tensorflow as tf
        
        def preprocess(filename, labels):
            img = tf.io.read_file(tf.strings.join([str(image_folder) + "/", filename]))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, target_size)
            return tf.cast(img, tf.float32) / 255.0, labels / normalize_by
        
        def augment_image(img, labels):
            """Apply data augmentation to training images."""
            # Random rotation (-10 to 10 degrees)
            angle = tf.random.uniform([], -10, 10, dtype=tf.float32)
            img = tf.image.rot90(img, k=tf.cast(angle / 90, tf.int32))
            
            # Random brightness adjustment
            img = tf.image.random_brightness(img, 0.15)
            
            # Random contrast adjustment
            img = tf.image.random_contrast(img, 0.8, 1.2)
            
            # Random zoom (crop and resize)
            crop_fraction = tf.random.uniform([], 0.85, 1.0)
            crop_size = tf.cast(tf.cast(target_size[0], tf.float32) * crop_fraction, tf.int32)
            img = tf.image.random_crop(img, [crop_size, crop_size, 3])
            img = tf.image.resize(img, target_size)
            
            # Random horizontal flip
            if tf.random.uniform([]) > 0.5:
                img = tf.image.flip_left_right(img)
                # Adjust x coordinate for flipped image
                labels = tf.tensor_scatter_nd_update(labels, [[0]], [512.0 - labels[0]])
            
            # Clip values to valid range
            img = tf.clip_by_value(img, 0.0, 1.0)
            
            return img, labels
        
        # Create base datasets with preprocessing
        train_base = tf.data.Dataset.from_tensor_slices((
            train_df['filename'].values,
            train_df[['x', 'y', 'distance']].values.astype('float32')
        )).shuffle(1000).map(preprocess)
        
        # Apply augmentation to training set if enabled
        if augment:
            train_ds = (train_base
                       .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                       .batch(batch_size)
                       .prefetch(tf.data.AUTOTUNE))
        else:
            train_ds = train_base.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset (no augmentation)
        val_ds = tf.data.Dataset.from_tensor_slices((
            val_df['filename'].values,
            val_df[['x', 'y', 'distance']].values.astype('float32')
        )).map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        augmentation_status = "with augmentation" if augment else "without augmentation"
        print(f"✅ Docking datasets created {augmentation_status}:")
        print(f"   Training samples: {len(train_df)}")
        print(f"   Validation samples: {len(val_df)}")
        print(f"   Batch size: {batch_size}")
        
        return train_ds, val_ds
        
    except ImportError:
        print("❌ TensorFlow not found. Please install: pip install tensorflow")
        return None, None
    except Exception as e:
        print(f"❌ Error creating docking datasets: {e}")
        return None, None

def train_model(model, train_generator, val_generator, steps_per_epoch, validation_steps, epochs=50):
    """
    Train the model with the given data generators.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        steps_per_epoch (int): Steps per training epoch
        validation_steps (int): Steps per validation epoch
        epochs (int): Number of training epochs
    
    Returns:
        tensorflow.keras.callbacks.History: Training history
    """
    try:
        import tensorflow as tf
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

def plot_training_history(history):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Keras training history object
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Training plots displayed!")
        
    except Exception as e:
        print(f"❌ Error plotting history: {e}")


if __name__ == "__main__":
    print("🚀 ISS Docking Analysis & ML Helper Functions loaded!")
    print("Available functions:")
    print("  📸 Image Analysis: show_image(), load_image(), etc.")
    print("  🔧 Preprocessing: process_and_save_image(), batch_process_and_save_images(), etc.")
    print("  🤖 ML Training: build_resnet_regression(), train_model(), etc.")


# ============================================================================
# OUTPUT NORMALIZATION FUNCTIONS (MinMaxScaler)
# ============================================================================

def create_output_scaler(outputs, feature_range=(0, 1)):
    """
    Create and fit a MinMaxScaler for output normalization.
    
    Args:
        outputs (numpy.ndarray): Output array to fit the scaler on
                                Shape: (n_samples, n_features) e.g., (N, 3) for [x, y, distance]
        feature_range (tuple): Desired range of transformed data (default: (0, 1))
        
    Returns:
        MinMaxScaler: Fitted scaler object
        
    Example:
        >>> outputs = np.array([[100, 200, 50.5], [150, 250, 75.2]])
        >>> scaler = create_output_scaler(outputs)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(outputs)
    return scaler


def normalize_outputs(outputs, scaler=None, feature_range=(0, 1)):
    """
    Normalize outputs using MinMaxScaler.
    
    Args:
        outputs (numpy.ndarray): Output array to normalize
                                Shape: (n_samples, n_features) e.g., (N, 3) for [x, y, distance]
        scaler (MinMaxScaler, optional): Pre-fitted scaler. If None, creates and fits a new one
        feature_range (tuple): Desired range of transformed data (default: (0, 1))
        
    Returns:
        tuple: (normalized_outputs, scaler)
            - normalized_outputs (numpy.ndarray): Normalized output array
            - scaler (MinMaxScaler): The scaler used for normalization
            
    Example:
        >>> outputs = np.array([[100, 200, 50.5], [150, 250, 75.2]])
        >>> normalized, scaler = normalize_outputs(outputs)
        >>> print(normalized)
        [[0. 0. 0.]
         [1. 1. 1.]]
    """
    if scaler is None:
        scaler = create_output_scaler(outputs, feature_range=feature_range)
    
    normalized_outputs = scaler.transform(outputs)
    return normalized_outputs, scaler


def denormalize_outputs(normalized_outputs, scaler):
    """
    Denormalize (inverse transform) normalized outputs back to original scale.
    
    Args:
        normalized_outputs (numpy.ndarray): Normalized output array to denormalize
                                           Shape: (n_samples, n_features) e.g., (N, 3)
        scaler (MinMaxScaler): The fitted scaler used during normalization
        
    Returns:
        numpy.ndarray: Denormalized outputs in original scale
        
    Example:
        >>> # After training with normalized outputs
        >>> predictions_normalized = model.predict(X_test)  # shape: (N, 3)
        >>> predictions_original = denormalize_outputs(predictions_normalized, scaler)
        >>> print(predictions_original)  # Back to original [x, y, distance] scale
        
    Note:
        - The scaler must be the same one used during normalization
        - Typically saved during training and loaded during inference
    """
    if scaler is None:
        raise ValueError("Scaler cannot be None. Please provide the fitted scaler used during normalization.")
    
    denormalized_outputs = scaler.inverse_transform(normalized_outputs)
    return denormalized_outputs


def save_output_scaler(scaler, filepath='data/output_scaler.pkl'):
    """
    Save a fitted MinMaxScaler to disk for later use.
    
    Args:
        scaler (MinMaxScaler): Fitted scaler to save
        filepath (str): Path where the scaler will be saved (default: 'data/output_scaler.pkl')
        
    Returns:
        str: Path where the scaler was saved
        
    Example:
        >>> scaler = create_output_scaler(train_outputs)
        >>> save_output_scaler(scaler, 'models/my_scaler.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Scaler saved to: {filepath}")
    return str(filepath)


def load_output_scaler(filepath='data/output_scaler.pkl'):
    """
    Load a saved MinMaxScaler from disk.
    
    Args:
        filepath (str): Path to the saved scaler file (default: 'data/output_scaler.pkl')
        
    Returns:
        MinMaxScaler: Loaded scaler object
        
    Example:
        >>> scaler = load_output_scaler('models/my_scaler.pkl')
        >>> predictions_original = denormalize_outputs(predictions_norm, scaler)
        
    Raises:
        FileNotFoundError: If the scaler file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Scaler loaded from: {filepath}")
    return scaler


def get_scaler_info(scaler):
    """
    Display information about a fitted MinMaxScaler.
    
    Args:
        scaler (MinMaxScaler): Fitted scaler object
        
    Returns:
        dict: Dictionary containing scaler information
        
    Example:
        >>> scaler = load_output_scaler('data/output_scaler.pkl')
        >>> info = get_scaler_info(scaler)
        >>> print(info)
    """
    if not hasattr(scaler, 'data_min_'):
        print("⚠️  Scaler has not been fitted yet!")
        return None
    
    info = {
        'feature_range': scaler.feature_range,
        'n_features': scaler.n_features_in_,
        'data_min': scaler.data_min_,
        'data_max': scaler.data_max_,
        'data_range': scaler.data_range_,
        'scale': scaler.scale_,
        'min': scaler.min_
    }
    
    print("📊 MinMaxScaler Information:")
    print(f"   Feature range: {info['feature_range']}")
    print(f"   Number of features: {info['n_features']}")
    print(f"   Data min values: {info['data_min']}")
    print(f"   Data max values: {info['data_max']}")
    print(f"   Data range: {info['data_range']}")
    
    return info



# ============================================================================
# IMAGE INPUT NORMALIZATION FUNCTIONS (Pixel-level normalization)
# ============================================================================

def normalize_image_pixels(image_array, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image pixel values using mean and standard deviation (typically ImageNet stats).
    This is the standard normalization for CNN inputs.
    
    Args:
        image_array (numpy.ndarray): Image array with shape (H, W, 3) or (N, H, W, 3)
                                     Values should be in [0, 1] range
        mean (list): Mean values for R, G, B channels (default: ImageNet mean)
        std (list): Standard deviation values for R, G, B channels (default: ImageNet std)
        
    Returns:
        numpy.ndarray: Normalized image array with same shape as input
        
    Example:
        >>> image = load_image('data/train/0.jpg')  # Shape: (480, 640, 3), values [0-255]
        >>> image = image / 255.0  # Convert to [0, 1]
        >>> normalized = normalize_image_pixels(image)
    """
    # Convert to float32 if needed
    image_array = image_array.astype(np.float32)
    
    # If image is in [0, 255] range, scale to [0, 1]
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    
    # Convert mean and std to numpy arrays
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    # Apply normalization: (image - mean) / std
    normalized = (image_array - mean) / std
    
    return normalized


def denormalize_image_pixels(normalized_array, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image pixels back to [0, 1] range for visualization.
    
    Args:
        normalized_array (numpy.ndarray): Normalized image array
        mean (list): Mean values used for normalization
        std (list): Std values used for normalization
        
    Returns:
        numpy.ndarray: Denormalized image in [0, 1] range
        
    Example:
        >>> denormalized = denormalize_image_pixels(normalized_image)
        >>> plt.imshow(denormalized)
    """
    # Convert mean and std to numpy arrays
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    # Reverse normalization: image * std + mean
    denormalized = (normalized_array * std) + mean
    
    # Clip to [0, 1] range
    denormalized = np.clip(denormalized, 0, 1)
    
    return denormalized


def create_image_statistics(images_array):
    """
    Calculate mean and std statistics from a batch of images for custom normalization.
    
    Args:
        images_array (numpy.ndarray): Array of images with shape (N, H, W, 3)
                                      Values should be in [0, 1] range
        
    Returns:
        dict: Dictionary with 'mean' and 'std' for each channel
        
    Example:
        >>> images, _, _, _ = load_and_preprocess_images('data/train_split.csv', 'data/train')
        >>> stats = create_image_statistics(images)
        >>> print(f"Custom mean: {stats['mean']}")
        >>> print(f"Custom std: {stats['std']}")
    """
    # Ensure images are in [0, 1] range
    if images_array.max() > 1.0:
        images_array = images_array / 255.0
    
    # Calculate mean and std per channel across all images
    mean = np.mean(images_array, axis=(0, 1, 2))
    std = np.std(images_array, axis=(0, 1, 2))
    
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_images': len(images_array)
    }
    
    print("📊 Image Statistics:")
    print("="*60)
    print(f"Number of images: {stats['n_images']}")
    print(f"Mean (R, G, B): {stats['mean']}")
    print(f"Std  (R, G, B): {stats['std']}")
    print("="*60)
    
    return stats


def save_image_statistics(stats, filepath='data/image_stats.pkl'):
    """
    Save image statistics to disk.
    
    Args:
        stats (dict): Dictionary with 'mean' and 'std' values
        filepath (str): Path to save the statistics
        
    Returns:
        str: Path where stats were saved
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"✅ Image statistics saved to: {filepath}")
    return str(filepath)


def load_image_statistics(filepath='data/image_stats.pkl'):
    """
    Load saved image statistics from disk.
    
    Args:
        filepath (str): Path to the saved statistics file
        
    Returns:
        dict: Dictionary with 'mean' and 'std' values
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Image statistics file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)
    
    print(f"✅ Image statistics loaded from: {filepath}")
    return stats


def batch_normalize_images(images_array, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize a batch of images efficiently.
    
    Args:
        images_array (numpy.ndarray): Batch of images with shape (N, H, W, 3)
        mean (list): Mean values for normalization
        std (list): Std values for normalization
        
    Returns:
        numpy.ndarray: Normalized batch of images
        
    Example:
        >>> images = np.array([img1, img2, img3])  # Shape: (3, 224, 224, 3)
        >>> normalized = batch_normalize_images(images)
    """
    # Convert to float32
    images_array = images_array.astype(np.float32)
    
    # Scale to [0, 1] if needed
    if images_array.max() > 1.0:
        images_array = images_array / 255.0
    
    # Convert mean and std to numpy arrays with shape (1, 1, 1, 3) for broadcasting
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    
    # Apply normalization
    normalized = (images_array - mean) / std
    
    return normalized
