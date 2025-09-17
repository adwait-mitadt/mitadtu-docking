# ISS Docking Analysis Helper Functions
# Simple, fast helper function for ISS docking image analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path

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
    row = target_data[target_data["ImageID"] == image_id]
    if row.empty:
        return None
    location = ast.literal_eval(row.iloc[0]["location"])
    return location[0], location[1]


def load_distance(image_id):
    """
    Load and return the distance for given image_id
    
    Args:
        image_id: The ID of the image (integer)
        
    Returns:
        float: The distance value
    """
    row = target_data[target_data["ImageID"] == image_id]
    if row.empty:
        return None
    return row.iloc[0]["distance"]


def show_image(image_id):
    """
    SUPER SIMPLE ALL-IN-ONE FUNCTION - Just call show_image(image_id) and get everything!
    
    Args:
        image_id: The ID of the image to display (integer)
        
    Returns:
        dict: Complete data about the image
        
    Usage Examples:
        show_image(0)     # Shows image 0 with all data
        show_image(100)   # Shows image 100 with all data
        show_image(1500)  # Shows image 1500 with all data
    """
    print(f"🚀 ANALYZING IMAGE {image_id}")
    print("="*50)
    
    try:
        # Load image
        image_file = inputs_directory_path / f"{image_id}.jpg"
        image = plt.imread(image_file)
        
        # Get data from CSV
        row = target_data[target_data["ImageID"] == image_id]
        if row.empty:
            print(f"❌ No data found for Image ID: {image_id}")
            return None
            
        row = row.iloc[0]  # Get first (and only) row
        distance = row["distance"]
        location = ast.literal_eval(row["location"])
        x, y = location[0], location[1]
        
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
        
        # Return structured data
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


def help_show_image():
    """Print usage instructions for show_image function"""
    print("🚀 ISS DOCKING IMAGE ANALYZER")
    print("="*50)
    print("Usage: show_image(image_id)")
    print()
    print("Examples:")
    print("  show_image(0)      # Analyze image 0")
    print("  show_image(100)    # Analyze image 100")
    print("  show_image(1500)   # Analyze image 1500")
    print()
    print("Features:")
    print("  ✅ Displays image with crosshair")
    print("  ✅ Prints all target data")
    print("  ✅ Shows approach phase")
    print("  ✅ Returns structured data")
    print("  ✅ Super fast execution")
    print()
    print("Just call: show_image(any_image_id)")


if __name__ == "__main__":
    print("ISS Docking Analysis Helper Functions loaded!")
    print("Use show_image(image_id) to analyze any image.")
