import os

##################  VARIABLES  ##################
SAMPLE_SIZE = float(os.environ.get("SAMPLE_SIZE"))
IMG_SIZE = int(os.environ.get("IMG_SIZE"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
FINE_TUNE = bool(os.environ.get("FINE_TUNE", "True") == "True")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(
    os.path.expanduser("~"),
    "code",
    "fermx3",
    "coffee-disease-detection",
    "data",
    "processed_data",
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

##################  VARIABLES CALCULATIONS  #####################
# Determinar nombre descriptivo para archivos
if SAMPLE_SIZE is None or SAMPLE_SIZE == 'full':
    SAMPLE_NAME = 'full'
elif SAMPLE_SIZE == 'half':
    SAMPLE_NAME = 'half'
elif isinstance(SAMPLE_SIZE, float):
    SAMPLE_NAME = f'{int(SAMPLE_SIZE*100)}pct'
else:
    SAMPLE_NAME = f'{SAMPLE_SIZE}'

# Configuración adaptativa de epochs según tamaño de muestra
if SAMPLE_SIZE == 100:
    EPOCHS = 30  # Más epochs para muestras pequeñas
elif SAMPLE_SIZE == 1000:
    EPOCHS = 40
elif isinstance(SAMPLE_SIZE, float) and SAMPLE_SIZE <= 0.5:
    EPOCHS = 50
else:
    EPOCHS = 60  # Más epochs para dataset completo
