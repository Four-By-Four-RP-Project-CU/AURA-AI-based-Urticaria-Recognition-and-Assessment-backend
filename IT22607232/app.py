import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import Predictor


WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "weights")
CONFIG_PATH = os.path.join(WEIGHTS_DIR, "config.json")
PREPROCESS_PATH = os.path.join(WEIGHTS_DIR, "preprocess.joblib")

WEIGHTS_FILE = os.getenv("WEIGHTS_FILE", "best_model.pt")  # or "model.pt"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)

DEVICE = os.getenv("DEVICE", None)  # e.g. "cpu" or "cuda"


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    weights_file: str


app = FastAPI(title="ClinicalSafe CU MTL API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[Predictor] = None


@app.on_event("startup")
def _load():
    global predictor
    predictor = Predictor(
        config_path=CONFIG_PATH,
        preprocess_path=PREPROCESS_PATH,
        weights_path=WEIGHTS_PATH,
        device=DEVICE,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=(predictor is not None),
        weights_file=WEIGHTS_FILE,
    )


@app.post("/predict")
def predict(req: PredictRequest):
    assert predictor is not None, "Model not loaded"
    return predictor.predict_one(req.features)
