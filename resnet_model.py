import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_resnet_model(num_classes):
    # Load ResNet50 with ImageNet weights, exclude top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Create the full model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    # Example usage
    num_classes = 10  # Replace with the number of classes in your dataset
    model = build_resnet_model(num_classes)
    model.summary()