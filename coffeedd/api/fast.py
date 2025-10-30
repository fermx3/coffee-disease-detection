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
from coffeedd.ml_logic.registry import predict, load_model

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
async def create_prediction(file: UploadFile = File(...)):
    """
    Make a prediction on an uploaded image
    """
    # basic validation
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPG and PNG are supported.")
    
    # read bytes
    image_bytes = await file.read()
    
    # Get prediction using the model
    try:
        result = predict(image_bytes)  # calls your registry.predict(...)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return result