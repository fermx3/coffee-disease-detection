"""
This module will be responsible for:
- buil model architecture
- loading the trained model
- preprocessing incoming images
- running predictions
"""
import os
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, Sequential

# class names
CLASS_NAMES = ['healthy', 'cerscospora', 'leaf_rust', 'miner', 'phoma']


# ---- paths (adjust names if needed) ----
WEIGHTS_H5_PATH = "data/coffe_model.h5"     # weights or full .h5 (we'll attempt to use as weights)
FULL_KERAS_PATH = "data/coffe_model.keras"  # full Keras format (we'll try this first if it works)

# ---- global cache ----
model = None  # cached model (singleton)


def build_model() -> tf.keras.Model:
    """
    Recreate the architecture used during training
    Returns a Keras model.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(500, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(72, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(5, activation="softmax")  # Assuming 5 classes for coffee diseases
    ])

    return model


def load_model() -> tf.keras.Model:
    """
    rebuild the architecture and load weights from .h5.
    """
    global model # the change i apply here will be reflected globally
    if model is not None:
        return model  # return cached model

    # 1) Try full model (.keras) first
    try:
        if os.path.exists(FULL_KERAS_PATH):
            print(f"[registry] Trying to load FULL model from {FULL_KERAS_PATH} ...")
            model = tf.keras.models.load_model(FULL_KERAS_PATH, compile=False)
            print("[registry] Full model loaded.")
            return model
    except Exception as e:
        print(f"[registry] Loading FULL model failed: {e}\n"
              f"[registry] Falling back to architecture+weights...")

    # 2) Fallback: rebuild architecture and load weights
    print("[registry] Rebuilding architecture and loading weights...")
    model = build_model()

    if os.path.exists(WEIGHTS_H5_PATH):
        try:
            # This will work if the .h5 contains weights compatible with the architecture above.
            model.load_weights(WEIGHTS_H5_PATH)
            print("[registry] Weights loaded from H5.")
        except Exception as e:
            print(f"[registry] Could not load weights from {WEIGHTS_H5_PATH}: {e}")
            print("[registry] If this fails, save weights explicitly in training with:\n"
                  "    model.save_weights('coffee_vgg16.weights.h5')\n"
                  "and put that file under data/, then set WEIGHTS_H5_PATH accordingly.")
    else:
        print(f"[registry] Weights file not found at {WEIGHTS_H5_PATH}."
              " You can export weights in training with:"
              " model.save_weights('coffee_vgg16.weights.h5')")

    return model


def preprocess_image(img_source) -> tf.Tensor:
    """
    Convert an image (path or bytes) into a tensor suitable for model prediction.
    - Accepts file path (str) or bytes
    - Resizes to (224,224)
    - Converts to array, expands batch dim, and applies VGG16 preprocessing
    """
    # Load from bytes or path
    if isinstance(img_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_source)).convert("RGB")
    else:
        img = image.load_img(img_source, target_size=(224, 224))

    # Ensure correct size
    img = img.resize((224, 224))
    
    # Convert the image to a NumPy array
    img = image.img_to_array(img)

    # Add a dimension for the batch size (VGG16 expects a batch of images)
    img = np.expand_dims(img, axis=0)

    # Apply VGG16-specific preprocessing (BGR conversion and mean subtraction)
    img = preprocess_input(img)
    
    return img

def predict
