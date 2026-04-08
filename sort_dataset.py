import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import ast
from pathlib import Path
from tqdm import tqdm

import pickle

def load_and_preprocess_image(image_path, img_size=224):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    image = np.expand_dims(image, axis=0)
    return image

def main():
    print("Loading full dataset...")
    csv_path = "data/train.csv"
    df = pd.read_csv(csv_path)

    if len(df) != 10000:
        print(f"Warning: expected 10000 rows but found {len(df)} rows in {csv_path}.")

    # train.csv stores location as a string like "[215, 158]"
    parsed_locations = df['location'].apply(ast.literal_eval)
    df['gt_x'] = parsed_locations.apply(lambda p: float(p[0]))
    df['gt_y'] = parsed_locations.apply(lambda p: float(p[1]))
    df['gt_distance'] = df['distance'].astype(float)
    
    # Load model and scaler
    MODEL_PATH = "models/resnet_docking.h5"
    SCALER_PATH = "data/output_scaler.pkl"
    
    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    print(f"Loading scaler from {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Store predictions and errors
    predictions = []
    missing_images = 0
    
    image_dir = Path("data/train") # or whatever your image folder is
    
    print("Running inference and calculating overall error...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_id = int(row['ImageID'])
        img_path = image_dir / f"{img_id}.jpg"
        
        gt_x = row['gt_x']
        gt_y = row['gt_y']
        gt_distance = row['gt_distance']
        
        if not img_path.exists():
            missing_images += 1
            continue
            
        # Predict
        image_input = load_and_preprocess_image(str(img_path))
        prediction_norm = model.predict(image_input, verbose=0)
        
        # Denormalize
        prediction_original = scaler.inverse_transform(prediction_norm)
        
        pred_x = prediction_original[0, 0]
        pred_y = prediction_original[0, 1]
        pred_distance = prediction_original[0, 2]
        
        # Calculate errors (example logic: MSE or MAE)
        # Weighting x/y error and distance error - customize as needed
        pos_error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        dist_error = abs(pred_distance - gt_distance)
        
        # Define a combined error metric to sort by
        # e.g., simply adding standard deviations could work, but using a weighted sum for simplicity
        total_error = pos_error + dist_error
        
        predictions.append({
            'image_number': img_id,
            'overall_error': total_error
        })
        
    results_df = pd.DataFrame(predictions)
    if results_df.empty:
        print("No predictions were generated. Check your CSV/image paths.")
        return
    
    results_df = results_df.sort_values(by='overall_error', ascending=True)
    results_df.to_csv("sorted_dataset_by_error.csv", index=False)

    print(f"Processed images: {len(results_df)}")
    print(f"Missing images skipped: {missing_images}")
    print("Done! Check 'sorted_dataset_by_error.csv' with columns: image_number, overall_error.")

if __name__ == "__main__":
    main()
