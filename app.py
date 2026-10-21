"""Customer Churn Prediction API with a lightweight browser UI."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_pipeline.pkl"
UI_PATH = BASE_DIR / "index.html"

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "A customer churn prediction service with a browser-based form and REST API. "
        "It returns a prediction, churn probability, and risk band from a trained scikit-learn pipeline."
    ),
    version="1.1.0",
)


class CustomerInput(BaseModel):
    """Input fields expected by the saved churn prediction pipeline."""

    age: int = Field(ge=18, le=100, examples=[35])
    gender: Literal["Female", "Male"] = "Female"
    region: Literal["East", "West", "North", "South"] = "East"
    tenure_months: int = Field(ge=0, le=72, examples=[12])
    monthly_charges: float = Field(ge=0, le=500, examples=[70.0])
    total_charges: float = Field(ge=0, le=20000, examples=[800.0])
    contract_type: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    internet_service: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    tech_support: Literal["Yes", "No", "No internet service"] = "No"
    online_security: Literal["Yes", "No", "No internet service"] = "No"
    paperless_billing: Literal["Yes", "No"] = "Yes"
    payment_method: Literal[
        "Electronic check", "Bank transfer", "Credit card", "Mailed check"
    ] = "Electronic check"
    num_support_calls: int = Field(ge=0, le=15, examples=[1])
    late_payments_last_year: int = Field(ge=0, le=12, examples=[0])
    avg_monthly_usage_gb: float = Field(ge=0, le=1000, examples=[150.0])


class PredictionResponse(BaseModel):
    prediction: Literal["Churn", "Stay"]
    churn_probability: float = Field(ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    message: str


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model once and keep it in memory."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Copy churn_pipeline.pkl "
            "from your customer-prediction repository into this project folder."
        )
    return joblib.load(MODEL_PATH)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """Serve the custom user interface; API docs remain available at /docs."""
    if not UI_PATH.exists():
        return HTMLResponse(
            "<h1>Customer Churn Prediction</h1><p>UI file missing. Restore index.html.</p>",
            status_code=500,
        )
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/health", tags=["General"])
def health_check():
    model_file_exists = MODEL_PATH.exists()
    return {
        "status": "ok" if model_file_exists else "model_missing",
        "model_file_found": model_file_exists,
        "model_file": MODEL_PATH.name,
        "note": "If model_file_found is false, copy churn_pipeline.pkl into this folder.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerInput):
    """Predict whether the supplied customer is likely to churn."""
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "The model could not be loaded. Check that the scikit-learn version "
                "matches the one used to train the saved model."
            ),
        ) from exc

    input_df = pd.DataFrame([customer.model_dump()])

    try:
        predicted_class = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
        churn_probability = float(probabilities[positive_index])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check that the input fields match the trained model.",
        ) from exc

    if churn_probability < 0.35:
        risk_level = "Low"
    elif churn_probability < 0.65:
        risk_level = "Medium"
    else:
        risk_level = "High"

    prediction = "Churn" if predicted_class == 1 else "Stay"
    return PredictionResponse(
        prediction=prediction,
        churn_probability=round(churn_probability, 4),
        risk_level=risk_level,
        message=(
            "The model estimates that this customer may be at risk of leaving."
            if prediction == "Churn"
            else "The model estimates that this customer is more likely to stay."
        ),
    )
