from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf


def build_resnet_regression(
    learning_rate=1e-4,
    weight_x=0.35,
    weight_y=1.20,
    weight_dist=1.45,
    bias_penalty_y=0.10,
    bias_penalty_dist=0.15,
):
    # Custom RMSE metrics for each output
    def total_loss(y_true, y_pred):
        weighted_rmse = (
            weight_x * rmse_x(y_true, y_pred)
            + weight_y * rmse_y(y_true, y_pred)
            + weight_dist * rmse_dist(y_true, y_pred)
        )

        # Penalize consistent directional drift seen in evaluation.
        y_bias_penalty = bias_penalty_y * tf.abs(tf.reduce_mean(y_pred[:, 1] - y_true[:, 1]))
        dist_bias_penalty = bias_penalty_dist * tf.abs(tf.reduce_mean(y_pred[:, 2] - y_true[:, 2]))

        return weighted_rmse + y_bias_penalty + dist_bias_penalty

    def rmse_x(y_true, y_pred):
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred[:, 0] - y_true[:, 0])))

    def rmse_y(y_true, y_pred):
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred[:, 1] - y_true[:, 1])))

    def rmse_dist(y_true, y_pred):
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred[:, 2] - y_true[:, 2])))

    input_shape = (224, 224, 3)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze all layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(3, activation="relu")(x)  # Regression output: x, y, distance

    model = Model(inputs=base_model.input, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=total_loss,
        metrics=[total_loss, rmse_x, rmse_y, rmse_dist],
    )
    return model


if __name__ == "__main__":
    model = build_resnet_regression()
    model.summary()    