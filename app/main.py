# Part 9 - FastAPI App

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("models/fraud_model.pkl")

# Create app
app = FastAPI(title="Credit Card Fraud Detection API")


# Input format
class Transaction(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.features])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }