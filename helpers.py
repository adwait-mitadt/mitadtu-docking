# Imports for data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setting paths using pathlib
data_path = Path("data/")
target_path = data_path / "train.csv"
inputs_directory_path = data_path / "train"

# Loading the target data
target_data = pd.read_csv(target_path)


def load_data(image_id, return_format='both'):
    """
    Load image and/or target data for a specific image ID.
    
    Args:
        image_id: The ID of the image to load
        return_format: What to return - 'image', 'target', or 'both' (default)
        
    Returns:
        - If return_format='image': numpy.ndarray (the loaded image)
        - If return_format='target': dict with keys 'distance', 'location', 'x', 'y' or None
        - If return_format='both': tuple (image, target_data) or (None, None) if not found
    """
    image = None
    target = None
    
    # Load image if needed
    if return_format in ['image', 'both']:
        try:
            input_image_path = inputs_directory_path / f"{image_id}.jpg"
            image = plt.imread(input_image_path)
        except FileNotFoundError:
            if return_format == 'image':
                print(f"Image with ID {image_id} not found.")
                return None
            elif return_format == 'both':
                print(f"Image with ID {image_id} not found.")
                return None, None
    
    # Load target data if needed
    if return_format in ['target', 'both']:
        # Get the corresponding target data
        target_data_row = target_data[target_data["ImageID"] == image_id]
        
        if not target_data_row.empty:
            # Extract data
            distance = target_data_row["distance"].values[0]
            location = target_data_row["location"].values[0]
            
            # Parse the location string to extract x, y coordinates
            if isinstance(location, str):
                # Remove brackets and split by comma
                location_coords = location.strip("[]").split(", ")
                x, y = int(location_coords[0]), int(location_coords[1])
            else:
                # If it's already a list
                x, y = location[0], location[1]
            
            target = {
                'distance': distance,
                'location': location,
                'x': x,
                'y': y
            }
        else:
            if return_format == 'target':
                return None
            elif return_format == 'both':
                print(f"Target data for image ID {image_id} not found.")
                return None, None
    
    # Return based on format
    if return_format == 'image':
        return image
    elif return_format == 'target':
        return target
    else:  # 'both'
        return image, target


def plot_image_with_distance_crosshair(image_id):
    """
    This function plots an image of the ISS from the input data and adds a crosshair at
    the dock based on the target value.
    The crosshair is colored based on the distance from the target data for the image.

    Args:
        image_id: the image id to be plotted
    """
    # Load image and target data using the consolidated helper function
    input_image, target = load_data(image_id, return_format='both')
    
    if input_image is None or target is None:
        print(f"Could not load image or target data for image ID {image_id}")
        return

    # Extract data from target dictionary
    x, y = target['x'], target['y']
    distance = target['distance']

    # Create a colormap based on distance
    # Get min and max distances for normalization
    min_distance = target_data["distance"].min()
    max_distance = target_data["distance"].max()

    # Normalize distance to 0-1 range for colormap
    normalized_distance = (distance - min_distance) / (max_distance - min_distance)

    # Create figure with colorbar
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(input_image)

    # Create scatter plot with color based on distance
    scatter = ax.scatter(
        [x],
        [y],
        c=[distance],
        cmap="viridis",
        s=300,
        marker="x",
        vmin=min_distance,
        vmax=max_distance,
        edgecolors="white",
        linewidths=2,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Distance", rotation=270, labelpad=20, fontsize=12)

    ax.set_title(f"Image {image_id} - Target Location: ({x}, {y}) - Distance: {distance}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
