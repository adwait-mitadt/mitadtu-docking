# ISS Docking Analysis Helper Functions
# Simple, fast helper function for ISS docking image analysis

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import cv2
from pathlib import Path
from PIL import Image

# ============================================================================
# CONFIGURATION AND DATA LOADING
# ============================================================================
# Setting paths using pathlib
data_path = Path("data/")
target_path = data_path / "train.csv"
inputs_directory_path = data_path / "train"

# Loading the target data
target_data = pd.read_csv(target_path)

# ============================================================================
# CORE DATA LOADING FUNCTIONS
# ============================================================================

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

# ============================================================================
# PHASE AND COLOR ANALYSIS FUNCTIONS
# ============================================================================

def get_docking_phase(distance):
    """
    Determine the docking phase and color band based on distance.
    
    Args:
        distance (float): The distance value to categorize
        
    Returns:
        dict: Dictionary containing phase name, color, and emoji
    """
    if distance < 50:
        return {
            'phase': 'FINAL DOCKING',
            'color': 'green',
            'emoji': '🔴',
            'description': 'Critical docking phase'
        }
    elif distance < 100:
        return {
            'phase': 'FINAL APPROACH',
            'color': 'yellow', 
            'emoji': '🟡',
            'description': 'Final approach phase'
        }
    elif distance < 200:
        return {
            'phase': 'APPROACH',
            'color': 'yellow',
            'emoji': '🟡', 
            'description': 'Approach phase'
        }
    elif distance < 400:
        return {
            'phase': 'APPROACH',
            'color': 'red',
            'emoji': '🟢',
            'description': 'Controlled approach'
        }
    else:
        return {
            'phase': 'LONG RANGE',
            'color': 'red',
            'emoji': '🔵',
            'description': 'Long range navigation'
        }

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def resize_image(image_array, x, y):
    """
    Resize an image array to specified dimensions using OpenCV.
    
    Args:
        image_array (numpy.ndarray): The input image array to resize
        x (int): Target width in pixels
        y (int): Target height in pixels
        
    Returns:
        numpy.ndarray: Resized image array, or None if input is invalid
        
    Raises:
        Exception: If there's an error during resizing process
    """
    try:
        if image_array is not None:
            # Resize using cv2 (works with numpy arrays)
            resized_image = cv2.resize(image_array, (x, y))
            return resized_image
        else:
            print("Input image array is None")
            return None
            
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

# ============================================================================
# IMAGE DISPLAY FUNCTIONS
# ============================================================================

def show_image(image_array):
    """
    Display an image array using matplotlib.
    
    Args:
        image_array (numpy.ndarray): The image array to display
        
    Returns:
        None: Displays the image in a matplotlib figure
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()


def load_and_display_image(image_path):
    """
    Load and display an image from the specified file path.
    
    Args:
        image_path (str): The file path to the image to load and display
        
    Returns:
        None: Displays the image if found, prints error message if not found
    """
    if os.path.exists(image_path):
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print(f"Image not found at {image_path}")


def mark_target_on_image(image_path, target_row):
    """
    Display an image with a red cross marker at the target coordinates.
    
    Args:
        image_path (str): The file path to the image to display
        target_row (pandas.DataFrame): DataFrame row containing target location data
        
    Returns:
        None: Displays the image with target marker if found, prints error if not found
        
    Notes:
        The target_row must contain a 'location' column with coordinates in format [x, y]
    """
    if os.path.exists(image_path):
        image = Image.open(image_path)
        plt.imshow(image)
        # Extract target coordinates from 'location' column
        location = target_row['location'].values[0]
        # If location is a string like '[215, 158]', convert to list
        if isinstance(location, str):
            location = ast.literal_eval(location)
        x, y = location[0], location[1]
        # Mark the target with a red cross
        plt.scatter(x, y, color='red', s=100, marker='x')  # s is the size of the marker
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print(f"Image not found at {image_path}")

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def show_image_analysis(image_id):
    """
    SUPER SIMPLE ALL-IN-ONE FUNCTION - Just call show_image_analysis(image_id) and get everything!
    
    Args:
        image_id: The ID of the image to display (integer)
        
    Returns:
        dict: Complete data about the image
        
    Usage Examples:
        show_image_analysis(0)     # Shows image 0 with all data
        show_image_analysis(100)   # Shows image 100 with all data
        show_image_analysis(1500)  # Shows image 1500 with all data
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
        
        # Get phase information
        phase_info = get_docking_phase(distance)
        
        # Print comprehensive info
        print(f"📷 Image ID: {image_id}")
        print(f"📏 Distance: {distance}m") 
        print(f"🎯 Target: ({x}, {y})")
        print(f"🚀 Phase: {phase_info['emoji']} {phase_info['phase']}")
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
        plt.title(f'ISS Docking | Image {image_id} | Distance: {distance}m | Target: ({x}, {y}) | {phase_info["emoji"]} {phase_info["phase"]}')
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
            'phase': phase_info['phase'],
            'color': phase_info['color']
        }
        
    except FileNotFoundError:
        print(f"❌ Image file not found for ID: {image_id}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

# ============================================================================
# HELP AND UTILITY FUNCTIONS
# ============================================================================

def help_show_image():
    """Print usage instructions for show_image_analysis function"""
    print("🚀 ISS DOCKING IMAGE ANALYZER")
    print("="*50)
    print("Usage: show_image_analysis(image_id)")
    print()
    print("Examples:")
    print("  show_image_analysis(0)      # Analyze image 0")
    print("  show_image_analysis(100)    # Analyze image 100")
    print("  show_image_analysis(1500)   # Analyze image 1500")
    print()
    print("Features:")
    print("  ✅ Displays image with crosshair")
    print("  ✅ Prints all target data")
    print("  ✅ Shows approach phase")
    print("  ✅ Returns structured data")
    print("  ✅ Super fast execution")
    print()
    print("Just call: show_image_analysis(any_image_id)")
    print()
    print("Note: For displaying image arrays, use show_image(image_array)")


if __name__ == "__main__":
    print("ISS Docking Analysis Helper Functions loaded!")
    print("Use show_image_analysis(image_id) to analyze any image.")
    print("Use show_image(image_array) to display image arrays.")
