"""
To run the API, you can use uvicorn. Here's how to start it:

run in the terminal root of the project:
uvicorn coffeedd.api.fast:app --reload

Once the server is running, you can:

Access the API documentation at http://localhost:8000/docs
Test the API by:
Opening the docs page
Click on the POST /predict endpoint
Click "Try it out"
Upload an image file
Click "Execute"


to stop running the api, run in the terminal:
ctrl + c
"""



from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from coffeedd.interface.main import pred
from coffeedd.params import SUPPORTED_FORMATS

app = FastAPI(
    title="Coffee Disease Detection API",
    description="API for detecting diseases in coffee plants using computer vision"
)

# Allow Streamlit (or any localhost) to call us while developing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    """
    Root endpoint - can be used to check if API is running
    """
    return {
        "status": "ok",
        "message": "Welcome to the Coffee Disease Detection API!"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)")
):
    """
    Make a prediction on an uploaded image.

    Supports multiple input formats:
    - Direct file upload
    - Works with any model that accepts image bytes or base64
    """

    # 1. Validar formato
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # 2. Leer imagen
    try:
        image_bytes = await file.read()

        # Validar que no esté vacío
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # 3. Hacer predicción (maneja ambos formatos)
    try:
        # La función pred() debe aceptar bytes O base64
        # Opción 1: Enviar bytes directamente
        result = pred(img_source=image_bytes)

        # Opción 2: Si tu modelo necesita base64, convertir aquí
        # img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        # result = pred(img_source=img_base64)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Check if the model is loaded and ready"""
    try:
        from coffeedd.interface.main import load_model
        model = load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
