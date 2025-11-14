"""ISS Docking Model Training"""
import pandas as pd
import tensorflow as tf
from pathlib import Path
from resnet_model import build_resnet_regression

BATCH_SIZE, EPOCHS, LR = 32, 70, 1e-4
LOAD_MODEL = "models/resnet_docking_epoch_100.h5"  # Load 100th epoch model
INITIAL_EPOCH = 100  # Continue from epoch 100

def main():
    print(" Training ISS Docking Model (Continuing from Epoch 100)")
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
    
    # Load the 100th epoch model without compiling
    print(f"\n Loading model from {LOAD_MODEL}...")
    model = tf.keras.models.load_model(LOAD_MODEL, compile=False)
    print(f" Model loaded successfully!")
    print(f" Continuing training from epoch {INITIAL_EPOCH} to {INITIAL_EPOCH + EPOCHS}")
    
    # Custom RMSE metric for each output
    def rmse_x(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])))
    
    def rmse_y(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1])))
    
    def rmse_dist(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2])))
    
    def total_loss(y_true, y_pred):
        return rmse_x(y_true, y_pred) + rmse_y(y_true, y_pred) + rmse_dist(y_true, y_pred)
    
    # Compile with same settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=total_loss,
        metrics=[rmse_x, rmse_y, rmse_dist, total_loss]
    )
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=INITIAL_EPOCH + EPOCHS,
        initial_epoch=INITIAL_EPOCH,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "models/resnet_docking_epoch_{epoch:03d}.h5", 
                save_freq=5,
                save_best_only=False
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "models/resnet_docking_best.h5", 
                save_best_only=True, 
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
            tf.keras.callbacks.CSVLogger("logs/training_history.csv", append=True)
        ]
    )
    
    print(f"\n Final Training | Loss: {history.history['loss'][-1]:.4f} | RMSE X: {history.history['rmse_x'][-1]:.4f} | Y: {history.history['rmse_y'][-1]:.4f} | Dist: {history.history['rmse_dist'][-1]:.4f}")
    print(f" Final Validation | Loss: {history.history['val_loss'][-1]:.4f} | RMSE X: {history.history['val_rmse_x'][-1]:.4f} | Y: {history.history['val_rmse_y'][-1]:.4f} | Dist: {history.history['val_rmse_dist'][-1]:.4f}")
    print(f"\n Best model saved to models/resnet_docking_best.h5")
    print(f" Checkpoint models saved every 5 epochs in models/")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
