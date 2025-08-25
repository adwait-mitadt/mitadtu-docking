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


def load_image(image_id):
    """
    Load an image from the input directory.
    """
    # Load an input image
    input_image_path = inputs_directory_path / f"{image_id}.jpg"
    input_image = plt.imread(input_image_path)
    return input_image


def plot_image_with_distance_crosshair(image_id):
    """
    This function plots an image of the ISS from the input data and adds a crosshair at
    the dock based on the target value.
    The crosshair is colored based on the distance from the target data for the image.

    Args:
        image_id: the inage id to be plotted
    """
    # Load an input image
    input_image = load_image(image_id)

    # Get the corresponding target data
    target_data_row = target_data[target_data["ImageID"] == image_id]

    # Display the location in the target data as a colored crosshair on the image
    if not target_data_row.empty:
        # Extract coordinates from the location column
        location = target_data_row["location"].values[0]
        distance = target_data_row["distance"].values[0]

        # Parse the location string to extract x, y coordinates
        if isinstance(location, str):
            # Remove brackets and split by comma
            location = location.strip("[]").split(", ")
            x, y = int(location[0]), int(location[1])
        else:
            # If it's already a list
            x, y = location[0], location[1]

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
