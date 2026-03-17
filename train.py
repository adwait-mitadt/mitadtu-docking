"""ISS Docking Model Training"""
import pandas as pd
import tensorflow as tf
from pathlib import Path
from resnet_model import build_resnet_regression
from helpers import create_docking_datasets

BATCH_SIZE, EPOCHS, LR = 32, 200, 1e-4
CHECKPOINT_DIR = "checkpoints"  # Directory for TF checkpoints
BEST_WEIGHTS_PATH = "models/resnet_docking_best.weights.h5"

def main():
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" GPU(s) available and configured: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f" GPU configuration error: {e}")
    else:
        print(" No GPU found. Training on CPU.")
    
    print(" Training ISS Docking Model (Starting from Scratch)")
    train_df = pd.read_csv("data/train_split.csv")
    val_df = pd.read_csv("data/val_split.csv")
    print(f" {len(train_df)} training samples")
    print(f" {len(val_df)} validation samples")
    
    # Create datasets using helper function
    train_ds, val_ds = create_docking_datasets(
        train_df, val_df, 
        image_folder="data/train",
        batch_size=BATCH_SIZE,
        target_size=(224, 224),
        normalize_by=512.0
    )
    
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    # Build model (already compiled with optimizer and metrics in resnet_model.py)
    model = build_resnet_regression(learning_rate=LR)
    
    # Create checkpoint to save model AND optimizer state
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer, epoch=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, 
        directory=CHECKPOINT_DIR, 
        max_to_keep=5
    )
    
    # Try to restore from latest checkpoint
    initial_epoch = 0
    if checkpoint_manager.latest_checkpoint:
        try:
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            initial_epoch = int(checkpoint.epoch.numpy())
            print(f"\n Restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
            print(f" Resuming training from epoch {initial_epoch} to {EPOCHS}")
        except Exception as e:
            print(f"\n Failed to restore checkpoint (likely architecture mismatch): {e}")
            print(f" Starting fresh training from epoch 0 for {EPOCHS} epochs.")
    else:
        print(f"\n No checkpoint found. Starting fresh training from epoch 0 for {EPOCHS} epochs.")

    if initial_epoch >= EPOCHS:
        print(f"\n Checkpoint epoch ({initial_epoch}) is already >= configured EPOCHS ({EPOCHS}).")
        print(" Increase EPOCHS to continue training, or clear checkpoints to restart from scratch.")
        return
    
    # Custom callback to save checkpoint with optimizer state
    class CheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, checkpoint, checkpoint_manager, checkpoint_var):
            super().__init__()
            self.checkpoint = checkpoint
            self.checkpoint_manager = checkpoint_manager
            self.checkpoint_var = checkpoint_var
            
        def on_epoch_end(self, epoch, logs=None):
            # Update the epoch variable in checkpoint
            self.checkpoint_var.assign(epoch + 1)
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_path = self.checkpoint_manager.save()
                print(f"\n Checkpoint saved at epoch {epoch + 1}: {save_path}")
    
    # Custom callback to save H5 model every 5 epochs
    class ModelCheckpointEvery5(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                filepath = f"models/resnet_docking_epoch_{epoch + 1:03d}.weights.h5"
                self.model.save_weights(filepath)
                print(f" Weights saved: {filepath}")
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[
            CheckpointCallback(checkpoint, checkpoint_manager, checkpoint.epoch),
            ModelCheckpointEvery5(),
            tf.keras.callbacks.ModelCheckpoint(
                BEST_WEIGHTS_PATH,
                save_weights_only=True,
                save_best_only=True, 
                monitor='val_loss',
                verbose=1
            ),
            # tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            tf.keras.callbacks.CSVLogger("logs/training_history.csv", append=True)
        ]
    )

    def _safe_last(history_dict, key):
        values = history_dict.get(key, [])
        return values[-1] if values else float('nan')
    
    print(
        f"\n Final Training | Loss: {_safe_last(history.history, 'loss'):.4f}"
        f" | RMSE X: {_safe_last(history.history, 'rmse_x'):.4f}"
        f" | Y: {_safe_last(history.history, 'rmse_y'):.4f}"
        f" | Dist: {_safe_last(history.history, 'rmse_dist'):.4f}"
    )
    print(
        f" Final Validation | Loss: {_safe_last(history.history, 'val_loss'):.4f}"
        f" | RMSE X: {_safe_last(history.history, 'val_rmse_x'):.4f}"
        f" | Y: {_safe_last(history.history, 'val_rmse_y'):.4f}"
        f" | Dist: {_safe_last(history.history, 'val_rmse_dist'):.4f}"
    )
    print(f"\n Best weights saved to {BEST_WEIGHTS_PATH}")
    print(" Checkpoint weights saved every 5 epochs in models/")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
