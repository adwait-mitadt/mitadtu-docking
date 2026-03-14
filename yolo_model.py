"""
YOLOv8 model architecture for object detection with optional distance regression head.

This module provides functions to build YOLOv8 detectors using keras_cv and to attach
an image-level distance regression head to the detector.
"""

from typing import Optional, Tuple, Dict, Any
import tensorflow as tf

try:
    import keras_cv
except ImportError:
    raise ImportError(
        "keras_cv is required but not installed. "
        "Install it with: pip install keras-cv"
    )

# Default image dimensions for dummy forward pass during feature extraction
_DUMMY_IMAGE_HEIGHT = 224
_DUMMY_IMAGE_WIDTH = 224
_DUMMY_IMAGE_CHANNELS = 3


def build_yolo_detector(
    num_classes: int = 1,
    backbone_preset: str = "yolo_v8_s_backbone_coco",
    fpn_depth: int = 1,
    bounding_box_format: str = "xyxy",
    compile_model: bool = True,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    classification_loss: str = "binary_crossentropy",
    box_loss: str = "ciou",
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build a YOLOv8 object detector using keras_cv.

    Args:
        num_classes: Number of object classes to detect.
        backbone_preset: Preset name for the YOLOv8 backbone. Options include:
            - "yolo_v8_xs_backbone_coco"
            - "yolo_v8_s_backbone_coco"
            - "yolo_v8_m_backbone_coco"
            - "yolo_v8_l_backbone_coco"
            - "yolo_v8_xl_backbone_coco"
        fpn_depth: Depth of the Feature Pyramid Network.
        bounding_box_format: Format for bounding boxes. Common formats:
            - "xyxy": [x_min, y_min, x_max, y_max]
            - "xywh": [x_center, y_center, width, height]
            - "rel_xyxy": relative coordinates [x_min, y_min, x_max, y_max]
        compile_model: Whether to compile the model with optimizer and losses.
        optimizer: Custom optimizer. If None and compile_model=True, uses Adam.
        classification_loss: Loss function for classification head.
        box_loss: Loss function for bounding box regression. Options:
            - "ciou": Complete IoU loss
            - "giou": Generalized IoU loss
            - "iou": Standard IoU loss
        learning_rate: Learning rate for the optimizer (used if optimizer is None).

    Returns:
        A compiled (if compile_model=True) YOLOv8 detector model.

    Raises:
        ImportError: If keras_cv is not installed.
        ValueError: If invalid preset or configuration is provided.
    """
    # Build the backbone
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(backbone_preset)

    # Build the detector
    detector = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        backbone=backbone,
        fpn_depth=fpn_depth,
        bounding_box_format=bounding_box_format,
    )

    # Compile if requested
    if compile_model:
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        detector.compile(
            optimizer=optimizer,
            classification_loss=classification_loss,
            box_loss=box_loss,
        )

    return detector


def add_image_level_distance_head(
    detector: tf.keras.Model,
    hidden_units: Tuple[int, ...] = (256, 64),
    activation: str = "relu",
    distance_activation: str = "relu",
) -> tf.keras.Model:
    """
    Attach an image-level distance regression head to a YOLOv8 detector.

    This function creates a multi-output model that produces both object detections
    and a single scalar distance prediction per image. The distance head operates on
    pooled features from the detector's backbone.

    Feature extraction strategy:
    - Extracts features from the detector's backbone using a dummy forward pass
    - Applies GlobalAveragePooling2D to aggregate spatial information
    - Passes through dense layers to predict a single distance value

    The distance head is "image-level" because it produces ONE distance value per
    input image, not per detected object. This is suitable for tasks like:
    - Predicting distance to the nearest object in the scene
    - Estimating overall scene depth
    - Regression tasks that depend on global image features

    IMPORTANT: The caller (developer integrating this model) must handle multi-output losses when training.
    The model returns a dictionary with two keys:
    - "detections": the original YOLOv8 detection output
    - "distance": a single scalar distance prediction

    Args:
        detector: A YOLOv8 detector model (from build_yolo_detector).
        hidden_units: Tuple of hidden layer sizes for the distance head.
        activation: Activation function for hidden layers.
        distance_activation: Activation function for the final distance output.
            Use "relu" to enforce non-negative distances, or "linear" for
            unconstrained regression.

    Returns:
        A multi-output Keras Model with keys: {"detections", "distance"}.

    Raises:
        RuntimeError: If feature extraction from the backbone fails.
        AttributeError: If the detector doesn't have a backbone attribute.

    Example:
        >>> detector = build_yolo_detector(num_classes=1, compile_model=False)
        >>> multi_output_model = add_image_level_distance_head(detector)
        >>> outputs = multi_output_model(images)
        >>> detections = outputs["detections"]
        >>> distances = outputs["distance"]
    """
    if not hasattr(detector, "backbone"):
        raise AttributeError(
            "The provided detector model does not have a 'backbone' attribute. "
            "Ensure you are passing a valid YOLOv8 detector."
        )

    # Create input layer with the same shape as detector input
    input_layer = tf.keras.layers.Input(shape=(None, None, 3), name="image_input")

    # Get detector outputs
    detector_output = detector(input_layer)

    # Extract features from backbone for distance head
    try:
        # Run a dummy forward pass to extract features
        dummy_input = tf.zeros([1, _DUMMY_IMAGE_HEIGHT, _DUMMY_IMAGE_WIDTH, _DUMMY_IMAGE_CHANNELS])
        backbone_features = detector.backbone(dummy_input)

        # Handle different output types (dict, list, or tensor)
        if isinstance(backbone_features, dict):
            # Use the last feature map if dict
            feature_key = list(backbone_features.keys())[-1]
            features = detector.backbone(input_layer)[feature_key]
        elif isinstance(backbone_features, (list, tuple)):
            # Use the last feature map if list/tuple
            features = detector.backbone(input_layer)[-1]
        else:
            # Single tensor output
            features = detector.backbone(input_layer)

    except Exception as e:
        raise RuntimeError(
            f"Failed to extract features from detector backbone: {e}\n"
            "Ensure the detector has a properly initialized backbone."
        )

    # Build distance regression head
    x = tf.keras.layers.GlobalAveragePooling2D(name="distance_pool")(features)

    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            units, activation=activation, name=f"distance_dense_{i}"
        )(x)

    distance_output = tf.keras.layers.Dense(
        1, activation=distance_activation, name="distance_output"
    )(x)

    # Create multi-output model
    multi_output_model = tf.keras.Model(
        inputs=input_layer,
        outputs={"detections": detector_output, "distance": distance_output},
        name="yolo_with_distance",
    )

    return multi_output_model


if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 Model Smoke Test")
    print("=" * 60)

    # Test 1: Build and run basic detector
    print("\n[Test 1] Building YOLOv8 detector...")
    try:
        detector = build_yolo_detector(
            num_classes=1,
            backbone_preset="yolo_v8_s_backbone_coco",
            compile_model=False,
        )
        print("✓ Detector built successfully")

        print("\n[Test 1] Running forward pass...")
        dummy_image = tf.zeros([1, 224, 224, 3])
        output = detector(dummy_image, training=False)
        print(f"✓ Forward pass successful")
        print(f"  Output type: {type(output)}")

    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        print("\nPlease ensure keras_cv is installed:")
        print("  pip install keras-cv tensorflow")
        exit(1)

    # Test 2: Add distance head and run multi-output forward pass
    print("\n[Test 2] Adding image-level distance head...")
    try:
        multi_model = add_image_level_distance_head(
            detector,
            hidden_units=(256, 64),
            activation="relu",
            distance_activation="relu",
        )
        print("✓ Distance head attached successfully")

        print("\n[Test 2] Running multi-output forward pass...")
        multi_output = multi_model(dummy_image, training=False)
        print("✓ Multi-output forward pass successful")
        print(f"  Output keys: {list(multi_output.keys())}")
        print(f"  Distance shape: {multi_output['distance'].shape}")

    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        exit(1)

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nModel is ready for integration into your training pipeline.")
    print("Remember to handle multi-output losses when training the")
    print("model with the distance head attached.")
