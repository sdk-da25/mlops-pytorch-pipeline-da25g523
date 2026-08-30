"""FastAPI inference service for the CIFAR-10 classifier.

Loads the trained model once at startup and serves predictions using the
exact same eval transform as training (``dataset.get_transforms``), so
serving and training preprocessing never drift apart.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from dataset import CIFAR10_CLASSES, get_transforms
from model import get_model

CONTAINER_CONFIG_PATH = "/app/configs/training_config.yaml"
LOCAL_CONFIG_PATH = "configs/training_config.yaml"
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")

app = FastAPI(title="CIFAR-10 Classifier")

MODEL_READY = False
_model: torch.nn.Module | None = None
_transform = get_transforms(train=False)


def load_config() -> dict:
    config_path = CONTAINER_CONFIG_PATH if Path(CONTAINER_CONFIG_PATH).exists() else LOCAL_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def load_model() -> None:
    global MODEL_READY, _model

    try:
        config = load_config()
        model_config = config["model"]

        model = get_model(model_config["architecture"], model_config["num_classes"])
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to("cpu")
        model.eval()

        _model = model
        MODEL_READY = True
    except Exception:
        # Leave MODEL_READY False (e.g. checkpoint not yet mounted) instead
        # of crashing the process — /health reports this via HTTP 503.
        MODEL_READY = False


@app.get("/health")
def health() -> dict[str, str]:
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    pil_image = pil_image.resize((32, 32))
    tensor = _transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = _model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()

    predictions = dict(zip(CIFAR10_CLASSES, probabilities))
    top_class = max(predictions, key=predictions.get)

    return {"predictions": predictions, "top_class": top_class}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
