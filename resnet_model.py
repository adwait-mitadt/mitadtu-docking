import keras
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.models import Model


def build_resnet_regression(learning_rate=1e-4):
    # Custom RMSE metrics for each output
    def total_loss(y_true, y_pred):
        return (rmse_x(y_true, y_pred) + rmse_y(y_true, y_pred) + rmse_dist(y_true, y_pred)) / 3

    def rmse_x(y_true, y_pred):
        return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred[:, 0] - y_true[:, 0])))

    def rmse_y(y_true, y_pred):
        return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred[:, 1] - y_true[:, 1])))

<<<<<<< HEAD
	def rmse_distance(y_true, y_pred):
		return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred[:, 2] - y_true[:, 2])))
=======
    def rmse_dist(y_true, y_pred):
        return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred[:, 2] - y_true[:, 2])))
>>>>>>> f3d5072164c4b12c810d0156078758927a490d52

    input_shape = (224, 224, 3)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze all layers

    x = base_model.output
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(3, activation="sigmoid")(x)  # Regression output: x, y, distance

<<<<<<< HEAD
	model = Model(inputs=base_model.input, outputs=outputs)
	optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(
		optimizer=optimizer,
		loss=rmse,
		metrics=[rmse, rmse_x, rmse_y, rmse_distance],
	)
	return model
=======
    model = Model(inputs=base_model.input, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=total_loss,
        metrics=[total_loss, rmse_x, rmse_y, rmse_dist],
    )
    return model
>>>>>>> f3d5072164c4b12c810d0156078758927a490d52


def build_resnet50():
    """
    Build ResNet50 model for ISS docking position regression.
    Returns uncompiled model for flexible training configuration.

    Returns:
            tf.keras.Model: ResNet50-based regression model
    """
    input_shape = (224, 224, 3)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze all layers for transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation="linear")(x)  # Regression output: x, y, distance

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


if __name__ == "__main__":
    model = build_resnet_regression()
    model.summary()
