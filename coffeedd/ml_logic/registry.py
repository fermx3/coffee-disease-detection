"""
This module will be responsible for:
- buil model architecture
- loading the trained model
- preprocessing incoming images
- running predictions

Step 1: for now we only implement model loading.
"""
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, Sequential


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
    Try to load the full model from .keras (new format).
    If that fails (serialization mismatch), rebuild and load weights from .h5.
    """
    global model
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


def preprocess_image(image_bytes: bytes):
    """
    Placeholder. We'll implement this in the next step.
    For now it just raises to remind us it's not done.
    """
    raise NotImplementedError("preprocess_image() not implemented yet")

def predict