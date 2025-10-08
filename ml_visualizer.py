"""
ML Training Visualizer Module
Extracted from ml_training_visualizer.ipynb for clean imports
"""

import matplotlib.pyplot as plt
import time
import csv
import os


def visualize_training_results(history, experiment_name="training_run"):
    """
    Complete training visualizer function - visualizes training and validation loss
    
    Args:
        history: Keras training history object from model.fit()
        experiment_name (str): Name for the experiment (used in filenames)
    
    Returns:
        dict: Summary of created files and metrics
    """
    # Extract loss values from history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    print(f"🚀 Starting visualization for: {experiment_name}")
    print(f"📊 Data: {len(train_loss)} epochs")
    
    results = {
        "experiment_name": experiment_name,
        "epochs": len(train_loss),
        "files_created": []
    }
    
    # === CSV LOGGING ===
    csv_filename = f"{experiment_name}_metrics.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow(['Epoch', 'Training_Loss', 'Validation_Loss'])
            
            for epoch, (t_loss, v_loss) in enumerate(zip(train_loss, val_loss), 1):
                writer_csv.writerow([epoch, t_loss, v_loss])
        
        results["csv_file"] = csv_filename
        results["files_created"].append(f"CSV data: {csv_filename}")
        print(f"📁 Metrics saved to: {csv_filename}")
        
    except Exception as e:
        print(f"⚠️ CSV logging failed: {e}")
    
    # === MATPLOTLIB PLOTTING ===
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 7))
    
    # Plot with enhanced styling
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss', 
             linewidth=2.5, markersize=7, alpha=0.8)
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss', 
             linewidth=2.5, markersize=7, alpha=0.8)
    
    # Enhanced plot styling
    plt.title(f'{experiment_name}: Training vs Validation Loss', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epochs', fontsize=14, fontweight='semibold')
    plt.ylabel('Loss', fontsize=14, fontweight='semibold')
    plt.legend(fontsize=12, framealpha=0.9, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for overfitting detection
    if len(val_loss) > 3:
        min_val_idx = val_loss.index(min(val_loss))
        if min_val_idx < len(val_loss) - 2:  # Check if minimum is not at the end
            plt.annotate('Potential Overfitting Zone', 
                        xy=(min_val_idx + 2, val_loss[min_val_idx + 1]), 
                        xytext=(min_val_idx + 3, max(val_loss) * 0.8),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, color='red', alpha=0.8)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{experiment_name}_loss_curve.png"
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        results["plot_file"] = plot_filename
        results["files_created"].append(f"Plot image: {plot_filename}")
        print(f"🎨 Plot saved to: {plot_filename}")
    except Exception as e:
        print(f"⚠️ Plot saving failed: {e}")
    
    # Show plot
    plt.show()
    
    # === SUMMARY ===
    print("\n" + "="*50)
    print(f"🎉 Visualization Complete for: {experiment_name}")
    print("="*50)
    print(f"📊 Processed {len(train_loss)} epochs")
    print(f"📈 Final Training Loss: {train_loss[-1]:.4f}")
    print(f"📉 Final Validation Loss: {val_loss[-1]:.4f}")
    print(f"🎯 Best Validation Loss: {min(val_loss):.4f} (Epoch {val_loss.index(min(val_loss)) + 1})")
    
    print("\n📁 Files Created:")
    for file_info in results["files_created"]:
        print(f"   ✅ {file_info}")
    
    return results