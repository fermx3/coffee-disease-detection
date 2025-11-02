import os
from coffeedd.utilities.params_helpers import auto_type, get_epochs_for_sample_size, get_sample_name, get_model_name

##################  VARIABLES  ##################
SAMPLE_SIZE = os.environ.get("SAMPLE_SIZE")
IMG_SIZE = int(os.environ.get("IMG_SIZE"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
FINE_TUNE = bool(os.environ.get("FINE_TUNE", "True") == "True")
MODEL_TARGET = os.environ.get("MODEL_TARGET", "local")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "efficientnet")
PRODUCTION_MODEL = os.environ.get("PRODUCTION_MODEL", None)
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "imagenet")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(
    os.path.expanduser("~"),
    "code",
    "fermx3",
    "coffee-disease-detection",
    "data",
    "processed_data_full",
)

LOCAL_RAW_DATA_PATH = os.path.join(
    os.path.expanduser("~"),
    "code",
    "fermx3",
    "coffee-disease-detection",
    "data",
    "raw_data",
)

# Carpeta para guardar modelos
MODELS_PATH = os.path.join(
    os.path.expanduser("~"),
    "code",
    "fermx3",
    "coffee-disease-detection",
    "data",
    "models",
)

# Carpeta para guardar registros
LOCAL_REGISTRY_PATH = os.path.join(
    os.path.expanduser("~"),
    ".coffeedd",
    "mlops",
    "training_outputs",
)

IMG_PATTERNS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.gif",
    "*.tif",
    "*.tiff",
    "*.webp",
)

# Ruta a dataset
# DATA_PATH = '../data/raw_data_224'


CLASS_NAMES = ['healthy', 'cerscospora', 'leaf_rust', 'miner', 'phoma']
NUM_CLASSES = len(CLASS_NAMES)

SUPPORTED_FORMATS = ("image/jpeg", "image/jpg", "image/png")

##################  VARIABLES CALCULATIONS  #####################
# Sample size con conversión automática de tipo
SAMPLE_SIZE = auto_type(os.environ.get("SAMPLE_SIZE", "1.0"))

# Epochs adaptativo según sample size
EPOCHS = get_epochs_for_sample_size(SAMPLE_SIZE)

# Nombre descriptivo para archivos
SAMPLE_NAME = get_sample_name(SAMPLE_SIZE)

# Nombre del modelo
MODEL_NAME = get_model_name(MODEL_ARCHITECTURE, SAMPLE_NAME, EPOCHS)
