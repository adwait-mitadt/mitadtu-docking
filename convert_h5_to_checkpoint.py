"""Convert epoch 100 .h5 model to TensorFlow checkpoint with optimizer state"""
import tensorflow as tf
from resnet_model import build_resnet_regression

# Load the epoch 100 model
print("Loading epoch 100 model from .h5 file...")
model = build_resnet_regression()

# Define metrics (must match training)
def rmse_x(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])))

def rmse_y(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1])))

def rmse_dist(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2])))

def total_loss(y_true, y_pred):
    return rmse_x(y_true, y_pred) + rmse_y(y_true, y_pred) + rmse_dist(y_true, y_pred)

# Create optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compile model
model.compile(
    optimizer=optimizer,
    loss=total_loss,
    metrics=[rmse_x, rmse_y, rmse_dist, total_loss]
)

# Load weights from best model (which is from the first 100 epochs)
print("Loading weights from models/resnet_docking_best.h5...")
model = tf.keras.models.load_model("models/resnet_docking_best.h5", compile=False)
print("✓ Model loaded successfully")

# Recompile with optimizer
model.compile(
    optimizer=optimizer,
    loss=total_loss,
    metrics=[rmse_x, rmse_y, rmse_dist, total_loss]
)
print("✓ Model recompiled with fresh optimizer")

# Create checkpoint
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(100))
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, 
    directory="checkpoints", 
    max_to_keep=5
)

# Save checkpoint
save_path = checkpoint_manager.save()
print(f"✓ Checkpoint saved at: {save_path}")
print(f"✓ Ready to resume training from epoch 100")
print(f"\nNote: Optimizer state (momentum, etc.) is initialized fresh since")
print(f"the original .h5 file doesn't contain optimizer state.")
