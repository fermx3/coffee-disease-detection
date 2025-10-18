import os

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

labels = ["healthy", "unhealthy"]
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
