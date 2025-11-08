FROM python:3.10-slim

WORKDIR /app

# Install system dependencies with proper error handling
RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create required directories (where your code expects the model)
RUN mkdir -p /app/.coffeedd/mlops/training_outputs/models/vgg16

# Copy the application code
COPY coffeedd ./coffeedd

# Copy model from the repo (data/models) into the registry path
COPY data/models/model_VGG16_20251102-073551.keras \
     /app/.coffeedd/mlops/training_outputs/models/vgg16/

# Set environment variables
ENV MODEL_TARGET=local \
    MODEL_ARCHITECTURE=vgg16 \
    IMG_SIZE=224 \
    BATCH_SIZE=16 \
    LEARNING_RATE=0.00005 \
    FINE_TUNE=False \
    PRODUCTION_MODEL=model_VGG16_20251102-073551.keras \
    HOME=/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "coffeedd.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
