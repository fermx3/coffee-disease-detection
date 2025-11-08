import tensorflow as tf
from PIL import Image
import numpy as np
import io
from keras.utils import load_img, img_to_array


def decode_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(
        img_bytes, channels=0, expand_animations=False
    )  # autodetect
    # Asegurar 3 canales (RGB):
    ch = tf.shape(img)[-1]
    img = tf.cond(
        tf.equal(ch, 1),
        lambda: tf.image.grayscale_to_rgb(img),
        lambda: tf.cond(tf.equal(ch, 4), lambda: img[..., :3], lambda: img),
    )
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


def _random_resized_crop(img, target, seed):
    """Aproximación de RandomResizedCrop: escalar por lado corto y cropear aleatorio con rango de AR."""
    # Rango tipo ImageNet:
    min_scale, max_scale = 0.6, 1.0
    min_ratio, max_ratio = 0.8, 1.25

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    # Muestreamos AR y escala
    rnd = tf.random.stateless_uniform([2], seed)
    ratio = tf.exp(
        tf.math.log(min_ratio) + (tf.math.log(max_ratio / min_ratio)) * rnd[0]
    )
    scale = min_scale + (max_scale - min_scale) * rnd[1]

    # Tamaño objetivo del crop en “pix originales”
    area = h * w
    target_area = scale * area
    crop_h = tf.cast(tf.round(tf.sqrt(target_area / ratio)), tf.int32)
    crop_w = tf.cast(tf.round(tf.sqrt(target_area * ratio)), tf.int32)

    # Si excede, fallback a central crop sobre lado corto
    def do_random_crop():
        max_y = tf.maximum(1, tf.shape(img)[0] - crop_h + 1)
        max_x = tf.maximum(1, tf.shape(img)[1] - crop_w + 1)
        rnd2 = tf.random.stateless_uniform([2], seed + 1)
        y = tf.cast(rnd2[0] * tf.cast(max_y, tf.float32), tf.int32)
        x = tf.cast(rnd2[1] * tf.cast(max_x, tf.float32), tf.int32)
        return img[y : y + crop_h, x : x + crop_w, :]

    def do_center_crop_from_short():
        # resize por lado corto y center crop a target
        short = tf.minimum(h, w)
        scale2 = tf.cast(target, tf.float32) / short
        nh = tf.cast(tf.round(h * scale2), tf.int32)
        nw = tf.cast(tf.round(w * scale2), tf.int32)
        img2 = tf.image.resize(img, (nh, nw), method="bicubic")
        off_y = (nh - target) // 2
        off_x = (nw - target) // 2
        return img2[off_y : off_y + target, off_x : off_x + target, :]

    ok = tf.logical_and(crop_h <= tf.shape(img)[0], crop_w <= tf.shape(img)[1])
    cropped = tf.cond(ok, do_random_crop, do_center_crop_from_short)
    return tf.image.resize(cropped, (target, target), method="bicubic")


def normalize_imagenet(img):
    IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], tf.float32)
    IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], tf.float32)
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def _letterbox_to_square(img, target, pad_value=0.5):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    scale = tf.minimum(target / h, target / w)
    nh = tf.cast(tf.round(h * scale), tf.int32)
    nw = tf.cast(tf.round(w * scale), tf.int32)
    img = tf.image.resize(img, (nh, nw), method="bicubic")
    dh = target - nh
    dw = target - nw
    top = dh // 2
    left = dw // 2
    pad = [[top, dh - top], [left, dw - left], [0, 0]]
    img = tf.pad(img, pad, constant_values=pad_value)
    return img


def build_tf_dataset(
    paths,
    labels,
    target=224,
    batch_size=32,
    training=True,
    policy="rrc",  # "rrc" (random-resized-crop) | "letterbox"
    seed=42,
    shuffle_buffer=2048,
    num_parallel=tf.data.AUTOTUNE,
    cache=False,
):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )

    def _map(path, y):
        img = decode_image(path)
        if training and policy == "rrc":
            img = _random_resized_crop(
                img, target, seed=tf.constant([seed, 0], tf.int32)
            )
        else:
            img = _letterbox_to_square(img, target, pad_value=0.5)
        img = normalize_imagenet(img)
        return img, y

    ds = ds.map(_map, num_parallel_calls=num_parallel)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def parse_image(filename, label):
    """Lee y decodifica una imagen desde archivo"""
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # Normalizar a [0, 1]
    return image, label


def augment_image(image, label):
    """
    Data augmentation MÍNIMA para preservar características de enfermedad
    """
    # Solo flip horizontal (las hojas NO crecen al revés)
    image = tf.image.random_flip_left_right(image)

    # Ajustes MUY sutiles de iluminación
    image = tf.image.random_brightness(image, max_delta=0.05)  # Muy sutil
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)  # Muy sutil

    # NO cambiar saturación ni hue - son críticos para detectar enfermedades

    # Asegurar rango [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


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
        img = load_img(img_source, target_size=(224, 224))

    # Ensure correct size
    img = img.resize((224, 224))

    # Convert the image to a NumPy array
    img = img_to_array(img)

    # Add a dimension for the batch size (VGG16 expects a batch of images)
    img = np.expand_dims(img, axis=0)

    return img
