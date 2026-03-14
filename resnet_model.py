from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf


NUM_OUTPUTS = 3  # Number of regression outputs: x, y, distance


def build_resnet_regression(learning_rate=1e-4):
    # Custom RMSE metrics for each output
    def rmse_x(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred[:, 0] - y_true[:, 0])))

    def rmse_y(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred[:, 1] - y_true[:, 1])))

    def rmse_dist(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred[:, 2] - y_true[:, 2])))
    
    def total_loss(y_true, y_pred):
        return (rmse_x(y_true, y_pred) + rmse_y(y_true, y_pred) + rmse_dist(y_true, y_pred)) / float(NUM_OUTPUTS)

    input_shape = (224, 224, 3)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze all layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="softplus")(x)
    x = Dense(256, activation="softplus")(x)
    x = Dense(32, activation="softplus")(x)
    outputs = Dense(3, activation="softplus")(x)  # Regression output: x, y, distance

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