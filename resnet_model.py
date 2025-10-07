import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

def build_resnet_regression():
	input_shape = (224, 224, 3)
	base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
	base_model.trainable = False  # Freeze all layers

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(256, activation="relu")(x)
	x = Dropout(0.5)(x)
	outputs = Dense(2, activation="linear")(x)  # Regression output: x, y

	model = Model(inputs=base_model.input, outputs=outputs)
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
	return model

if __name__ == "__main__":
	model = build_resnet_regression()
	model.summary()