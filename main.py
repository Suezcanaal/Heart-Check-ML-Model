from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 1. Initialize App
app = FastAPI(title="Heart Disease Risk Predictor with SHAP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# app = FastAPI(title="Heart Disease Risk Predictor with SHAP")
# 2. Load Model & Explainer
# We load the trained brain once when the server starts
try:
    model = joblib.load('heart_model.pkl')
    # TreeExplainer is specifically optimized for XGBoost trees
    explainer = shap.TreeExplainer(model)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")

# 3. Define Input Schema
# This forces the user to send data that matches your CSV columns exactly
class PatientData(BaseModel):
    age: int        # Age in years
    sex: int        # 1 = Male, 0 = Female
    cp: int         # Chest Pain Type (0-3)
    trestbps: int   # Resting Blood Pressure
    chol: int       # Cholesterol
    fbs: int        # Fasting Blood Sugar > 120 (1=True, 0=False)
    restecg: int    # Resting ECG results (0-2)
    thalach: int    # Max Heart Rate achieved
    exang: int      # Exercise Induced Angina (1=Yes, 0=No)
    oldpeak: float  # ST depression induced by exercise
    slope: int      # Slope of the peak exercise ST segment
    ca: int         # Number of major vessels (0-3) colored by flourosopy
    thal: int       # Thalassemia (0-3)

# 4. Prediction Endpoint
@app.post("/predict_risk")
def predict_heart_disease(patient: PatientData):
    # Convert input data to DataFrame (XGBoost expects a DataFrame with column names)
    input_data = patient.dict()
    df = pd.DataFrame([input_data])
    
    # A. Predict
    # returns array like [0] or [1]
    prediction = model.predict(df)[0]
    # returns array like [[0.1, 0.9]] (probability of class 0 vs 1)
    probability = model.predict_proba(df)[0][1] 

    # B. Explain (SHAP)
    # shap_values_array has shape (1, num_features)
    shap_values = explainer.shap_values(df)
    
    # Map feature names to their SHAP impact
    # Positive SHAP = Pushes risk HIGHER
    # Negative SHAP = Pushes risk LOWER
    explanation = dict(zip(df.columns, shap_values[0].tolist()))

    return {
        "prediction": int(prediction), # 1 = Disease, 0 = Healthy
        "risk_probability": float(probability), # e.g., 0.85 (85% risk)
        "shap_explanation": explanation
    }

@app.get("/")
def home():
    # This serves your HTML file when someone visits the root URL
    return FileResponse("index.html")