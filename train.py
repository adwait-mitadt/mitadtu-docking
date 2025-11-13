"""ISS Docking Model Training"""
import pandas as pd
import tensorflow as tf
from pathlib import Path
from resnet_model import build_resnet_regression

BATCH_SIZE, EPOCHS, LR = 32, 20, 1e-4

def main():
    print(" Training ISS Docking Model")
    train_df = pd.read_csv("data/train_split.csv")
    val_df = pd.read_csv("data/val_split.csv")
    print(f" {len(train_df)} training samples")
    print(f" {len(val_df)} validation samples")
    
    def preprocess(filename, labels):
        img = tf.io.read_file(tf.strings.join(["data/train/", filename]))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        return tf.cast(img, tf.float32) / 255.0, labels / [512.0, 512.0, 512.0]
    
    train_ds = tf.data.Dataset.from_tensor_slices((
        train_df['filename'].values,
        train_df[['x', 'y', 'distance']].values.astype('float32')
    )).shuffle(1000).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((
        val_df['filename'].values,
        val_df[['x', 'y', 'distance']].values.astype('float32')
    )).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    model = build_resnet_regression(learning_rate=LR)
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=EPOCHS, 
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint("models/resnet_docking.h5", save_best_only=True, monitor='val_loss'),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            tf.keras.callbacks.CSVLogger("logs/training_history.csv")
        ]
    )
    
    print(f"\n Final Training | Loss: {history.history['loss'][-1]:.4f} | RMSE X: {history.history['rmse_x'][-1]:.4f} | Y: {history.history['rmse_y'][-1]:.4f} | Dist: {history.history['rmse_dist'][-1]:.4f}")
    print(f" Final Validation | Loss: {history.history['val_loss'][-1]:.4f} | RMSE X: {history.history['val_rmse_x'][-1]:.4f} | Y: {history.history['val_rmse_y'][-1]:.4f} | Dist: {history.history['val_rmse_dist'][-1]:.4f}")
    print(" Model saved to models/resnet_docking.h5")

if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
