# ===============================
# Path fix
# ===============================
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ===============================
# Imports
# ===============================
import joblib
import pandas as pd
import numpy as np
import logging

from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PassengerInput, PredictionResponse

# ===============================
# Config
# ===============================
API_KEY = os.getenv("API_KEY", "super-secret-key")

logging.basicConfig(level=logging.INFO)

# ===============================
# App
# ===============================
app = FastAPI(title="Airline Satisfaction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Load model & pipeline ONCE
# ===============================
model = joblib.load("models/xgboost_best_model.joblib")
pipeline = joblib.load("models/feature_pipeline.joblib")

SAT_CLASS_IDX = list(model.classes_).index(1)

# ===============================
# Auth dependency
# ===============================
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ===============================
# Health check
# ===============================
@app.get("/")
def health():
    return {"status": "running"}

# ===============================
# Single prediction
# ===============================
@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict(data: PassengerInput):

    df = pd.DataFrame([{
        "Gender": data.Gender,
        "Customer Type": data.Customer_Type,
        "Age": data.Age,
        "Type of Travel": data.Type_of_Travel,
        "Class": data.Class,
        "Flight Distance": data.Flight_Distance,
        "Departure Delay in Minutes": data.Departure_Delay_in_Minutes,
        "Arrival Delay in Minutes": data.Arrival_Delay_in_Minutes,

        "Seat comfort": data.Seat_comfort,
        "Food and drink": data.Food_and_drink,
        "Inflight wifi service": data.Inflight_wifi_service,
        "Inflight entertainment": data.Inflight_entertainment,
        "Online support": data.Online_support,
        "Ease of Online booking": data.Ease_of_Online_booking,
        "On-board service": data.On_board_service,
        "Leg room service": data.Leg_room_service,
        "Baggage handling": data.Baggage_handling,
        "Checkin service": data.Checkin_service,
        "Cleanliness": data.Cleanliness,
        "Online boarding": data.Online_boarding,
        "Gate location": data.Gate_location,
        "Departure/Arrival time convenient": data.Departure_Arrival_time_convenient,
    }])

    X = pipeline.transform(df)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][SAT_CLASS_IDX]

    return {
        "prediction": "Satisfied" if pred == 1 else "Dissatisfied",
        "probability": round(float(prob), 4),
    }

# ===============================
# Batch prediction
# ===============================
@app.post(
    "/predict-batch",
    dependencies=[Depends(verify_api_key)],
)
async def predict_batch(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)
    X = pipeline.transform(df)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, SAT_CLASS_IDX]

    df["prediction"] = np.where(preds == 1, "Satisfied", "Dissatisfied")
    df["probability"] = probs

    return df.to_dict(orient="records")
