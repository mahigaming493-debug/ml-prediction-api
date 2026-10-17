import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
model = joblib.load(BASE / "model" / "model.pkl")
meta = json.loads((BASE / "model" / "metadata.json").read_text())
FEATURES = meta["features"]          # e.g. "worst perimeter"
RANGES = meta["feature_ranges"]

app = FastAPI(title="Breast Cancer Prediction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Input(BaseModel):
    worst_perimeter: float = Field(..., example=107.0)
    mean_concave_points: float = Field(..., example=0.05)
    area_error: float = Field(..., example=24.5)
    worst_concavity: float = Field(..., example=0.27)
    worst_compactness: float = Field(..., example=0.21)
    mean_compactness: float = Field(..., example=0.10)


@app.get("/")
def home():
    return {"message": "ML API is running", "docs": "/docs", "metrics": meta["metrics"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Input):
    row = {f: getattr(data, f.replace(" ", "_")) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    warnings = [
        f"{f} ({v}) is outside training range {RANGES[f]}"
        for f, v in row.items() if not RANGES[f][0] <= v <= RANGES[f][1]
    ]
    return {
        "prediction": meta["target_mapping"][str(label)],
        "confidence": round(float(proba[label]), 4),
        "probability_malignant": round(float(proba[0]), 4),
        "probability_benign": round(float(proba[1]), 4),
        "warnings": warnings,
    }
