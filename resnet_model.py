import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

def build_resnet_regression(learning_rate=1e-4):
	def rmse(y_true, y_pred):
		return K.sqrt(K.mean(K.square(y_pred - y_true)))
	input_shape = (224, 224, 3)
	base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
	base_model.trainable = False  # Freeze all layers

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(256, activation="relu")(x)
	x = Dropout(0.5)(x)
	outputs = Dense(2, activation="linear")(x)  # Regression output: x, y

	model = Model(inputs=base_model.input, outputs=outputs)
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss=rmse, metrics=[rmse])
	return model


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
	outputs = Dense(2, activation="linear")(x)  # Regression output: x, y coordinates

	model = Model(inputs=base_model.input, outputs=outputs)
	return model

if __name__ == "__main__":
	model = build_resnet_regression()
	model.summary()