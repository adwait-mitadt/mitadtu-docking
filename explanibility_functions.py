"""Explainability helpers for the ISS docking ResNet regression model."""

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


TARGET_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_docking_model(model_path):
    """Load trained docking model without requiring custom compile objects."""
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(img_path, target_size=TARGET_SIZE, use_imagenet_norm=False):
    """Load an image and preprocess to model input format (1, H, W, 3)."""
    img = keras.utils.load_img(img_path, target_size=target_size)
    arr = keras.utils.img_to_array(img).astype(np.float32) / 255.0

    if use_imagenet_norm:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return np.expand_dims(arr, axis=0)


def predict_outputs(model, img_batch):
    """Return model predictions as [x, y, distance]."""
    return model(img_batch, training=False).numpy()


def get_output_gradients(model, img_batch, output_index):
    """Compute d(output_index)/d(input_image) for regression output index."""
    images = tf.cast(img_batch, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images, training=False)
        target = preds[:, output_index]

    grads = tape.gradient(target, images)
    return grads


def get_integrated_gradients(model, img_batch, output_index, baseline=None, num_steps=50):
    """Compute Integrated Gradients for one image batch of shape (1, H, W, 3)."""
    if img_batch.ndim != 4 or img_batch.shape[0] != 1:
        raise ValueError("img_batch must have shape (1, H, W, 3).")

    img = tf.cast(img_batch[0], tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(img)
    else:
        baseline = tf.cast(baseline, tf.float32)
        if baseline.shape != img.shape:
            raise ValueError("baseline must have shape (H, W, 3) matching input image.")

    alphas = tf.linspace(0.0, 1.0, num_steps + 1)
    interpolated = tf.stack([baseline + a * (img - baseline) for a in alphas], axis=0)

    grads = []
    for i in range(interpolated.shape[0]):
        interp_batch = tf.expand_dims(interpolated[i], axis=0)
        grad = get_output_gradients(model, interp_batch, output_index)[0]
        grads.append(grad)

    grads = tf.stack(grads, axis=0)
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    return (img - baseline) * avg_grads


def make_attribution_map(attributions, clip_percentile=99):
    """Convert (H, W, 3) attributions to a normalized (H, W) heatmap."""
    heatmap = tf.reduce_sum(tf.abs(attributions), axis=-1).numpy()
    vmax = np.percentile(heatmap, clip_percentile)
    if vmax <= 0:
        vmax = np.max(heatmap) + 1e-8
    return np.clip(heatmap, 0, vmax) / (vmax + 1e-8)


def plot_explanations(
    img_batch,
    ig_heatmap=None,
    title="Explainability",
    alpha=0.45,
    save_path=None,
    show_plot=True,
):
    """Plot input image and optional Integrated Gradients overlay."""
    image_vis = np.clip(img_batch[0], 0.0, 1.0)

    cols = 1 + int(ig_heatmap is not None)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1:
        axes = [axes]

    idx = 0
    axes[idx].imshow(image_vis)
    axes[idx].set_title("Input")
    axes[idx].axis("off")
    idx += 1

    if ig_heatmap is not None:
        axes[idx].imshow(image_vis)
        axes[idx].imshow(ig_heatmap, cmap="jet", alpha=alpha)
        axes[idx].set_title("Integrated Gradients")
        axes[idx].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.sa<<<<<<< HEAD
    return fig
=======
    return fig
>>>>>>> origin/main
, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig
