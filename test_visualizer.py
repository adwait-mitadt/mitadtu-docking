"""
Test script to load training history CSV and test the ML visualizer
"""

import pandas as pd
import os
from ml_visualizer import visualize_training_results

class MockHistory:
    """Mock Keras history object to simulate training results"""
    def __init__(self, train_loss, val_loss):
        self.history = {
            'loss': train_loss,
            'val_loss': val_loss
        }

def load_and_visualize_csv(csv_path):
    """
    Load training history from CSV and visualize it
    
    Args:
        csv_path (str): Path to the training history CSV file
    """
    try:
        # Load the CSV file
        print(f"📁 Loading training data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"📊 CSV shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📈 Data preview:")
        print(df.head())
        
        # Extract loss values
        if 'loss' in df.columns and 'val_loss' in df.columns:
            train_loss = df['loss'].tolist()
            val_loss = df['val_loss'].tolist()
        elif 'Training_Loss' in df.columns and 'Validation_Loss' in df.columns:
            train_loss = df['Training_Loss'].tolist()
            val_loss = df['Validation_Loss'].tolist()
        else:
            print("❌ Could not find loss columns in CSV")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Create mock history object
        mock_history = MockHistory(train_loss, val_loss)
        
        # Visualize using the ML visualizer
        print("\n🎨 Testing ML Visualizer...")
        results = visualize_training_results(
            mock_history, 
            experiment_name="CSV_Test_Run"
        )
        
        print(f"\n✅ Visualization test completed!")
        print(f"📋 Results: {results}")
        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("💡 Make sure to run training first to generate the CSV file")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")

def test_with_sample_data():
    """Test the visualizer with sample data"""
    print("🧪 Testing ML Visualizer with sample data...")
    
    # Create sample training data (showing overfitting)
    sample_train_loss = [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15]
    sample_val_loss = [2.7, 2.1, 1.6, 1.3, 1.1, 1.0, 1.1, 1.2, 1.4, 1.6]
    
    # Create mock history object
    mock_history = MockHistory(sample_train_loss, sample_val_loss)
    
    # Test the visualizer
    results = visualize_training_results(
        mock_history, 
        experiment_name="Sample_Test_Run"
    )
    
    print(f"\n✅ Sample data test completed!")
    return results

if __name__ == "__main__":
    print("🚀 ML Visualizer Test Script")
    print("=" * 50)
    
    # First, test with sample data
    print("\n1️⃣ Testing with sample data:")
    test_results = test_with_sample_data()
    
    # Then try to load actual CSV data
    csv_path = "logs/training_history.csv"
    print(f"\n2️⃣ Testing with actual CSV data ({csv_path}):")
    
    if os.path.exists(csv_path):
        load_and_visualize_csv(csv_path)
    else:
        print(f"❌ CSV file not found: {csv_path}")
        print("💡 Run your training script first to generate training data")
        
        # Create a sample CSV for testing
        print("\n📝 Creating sample CSV for testing...")
        sample_data = {
            'epoch': list(range(1, 11)),
            'loss': [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15],
            'val_loss': [2.7, 2.1, 1.6, 1.3, 1.1, 1.0, 1.1, 1.2, 1.4, 1.6]
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Save sample data
        df_sample = pd.DataFrame(sample_data)
        sample_csv_path = "logs/sample_training_history.csv"
        df_sample.to_csv(sample_csv_path, index=False)
        print(f"✅ Sample CSV created: {sample_csv_path}")
        
        # Test with the sample CSV
        print("\n3️⃣ Testing with sample CSV:")
        load_and_visualize_csv(sample_csv_path)