import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import ANN


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="ANN MNIST API",
    description="API for MNIST digit classification",
    version="1.0.0"
)


# ============================================================
# Load model
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ann_mnist.pth"

model = ANN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# Request schema
# ============================================================

class PredictionRequest(BaseModel):
    pixels: list[float]


# ============================================================
# Health check
# ============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "model": "ANN-MNIST"
    }


# ============================================================
# Model information
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    # MNIST image must contain 28 x 28 = 784 pixels
    if len(request.pixels) != 784:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 784 pixels, got {len(request.pixels)}"
        )

    # Convert input to tensor
    image = torch.tensor(
        request.pixels,
        dtype=torch.float32
    )

    # Reshape:
    # [784]
    #   ↓
    # [1, 1, 28, 28]
    image = image.reshape(1, 1, 28, 28)

    # Inference
    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    return {
        "prediction": int(predicted.item()),
        "confidence": float(confidence.item())
    }