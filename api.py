
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AI4I Predictive Maintenance API",
    version="1.0.0",
    description="API de prediction des defaillances industrielles avec XGBoost"
)

model = joblib.load(BASE_DIR / "models" / "xgboost.joblib")


class MachineInput(BaseModel):
    type: str = Field(pattern="^(L|M|H)$")
    air_temperature: float
    process_temperature: float
    rotational_speed: float = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "AI4I Predictive Maintenance API",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost"
    }


@app.post("/predict")
def predict(machine: MachineInput):
    data = pd.DataFrame([{
        "Type": machine.type,
        "Air temperature": machine.air_temperature,
        "Process temperature": machine.process_temperature,
        "Rotational speed": machine.rotational_speed,
        "Torque": machine.torque,
        "Tool wear": machine.tool_wear
    }])

    try:
        probability = float(model.predict_proba(data)[0, 1])
        threshold = 0.5
        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "threshold": threshold,
            "status": (
                "Panne detectee"
                if prediction == 1
                else "Fonctionnement normal"
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la prediction"
        ) from exc


@app.get("/metrics")
def metrics():
    path = BASE_DIR / "resultats_optimisation.csv"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Fichier des resultats introuvable"
        )

    df = pd.read_csv(path)
    return df.to_dict(orient="records")
