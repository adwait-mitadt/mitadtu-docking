import keras.backend as K

import keras
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.models import Model

def build_resnet_regression():
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
	optimizer = keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(optimizer=optimizer, loss=rmse, metrics=[rmse])
	return model

if __name__ == "__main__":
	model = build_resnet_regression()
	model.summary()