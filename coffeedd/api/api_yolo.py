# coffeedd/api/yolo.py

import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

from coffeedd.params import SUPPORTED_FORMATS

YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "data/models/yolo/best.pt")

app = FastAPI(
    title="Coffee YOLO Detection API",
    description="YOLOv8 object detection for coffee leaves"
)

# CORS, same as before
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once
try:
    print(f"🔄 Loading YOLO model from {YOLO_WEIGHTS_PATH} ...")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    print("✅ YOLO model loaded")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    yolo_model = None


@app.get("/")
def root():
    return {"status": "ok", "message": "YOLO Coffee Detection API"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if yolo_model is not None else "unhealthy",
        "model_loaded": yolo_model is not None,
        "weights_path": YOLO_WEIGHTS_PATH,
    }


@app.post("/predict")
async def yolo_predict(file: UploadFile = File(...)):
    # 1) Validate format
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    if yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    # 2) Read image
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    # 3) Run YOLO (classification)
    try:
        result = yolo_model(img)[0]   # first (and only) image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {e}")

    if result.probs is None:
        raise HTTPException(status_code=500, detail="Model did not return classification probabilities")

    # probs is a tensor with one score per class
    probs = result.probs.data.tolist()
    names = result.names  # id -> class name dict

    top_idx = int(result.probs.top1)
    top_conf = float(result.probs.top1conf)

    return {
        "top_class_name": names.get(top_idx, str(top_idx)),
        "top_confidence": top_conf,
        "probabilities": {
            names.get(i, str(i)): float(probs[i]) for i in range(len(probs))
        },
    }

