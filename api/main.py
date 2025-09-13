# filename: main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "crop_yield_model.pkl")
model = joblib.load(model_path)

scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = joblib.load(scaler_path)


# Define Input Schema
class CropData(BaseModel):
    State: str
    District: str
    Year: int
    Season: str
    Crop: str
    Area: float
    Production: float
    Temperature: float
    Dew: float
    Humidity: float
    Percipitation: float
    WindSpeed: float
    Pressure: float
    Rainfall_mm: float
    Soil_Type: str
    SoilFertility_General: str

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running!"}

@app.post("/predict")
def predict(data: CropData):
    try:
        
        df = pd.DataFrame([data.dict()])

        # Preprocessing (same as training)

        # Map Soil Fertility ordinal values
        soil_map = {"Moderate": 1, "High": 2}
        df["SoilFertility_General"] = df["SoilFertility_General"].map(soil_map)

        # Label encode State (must match training mapping)
        state_map = {"Andhra Pradesh": 0, "Telangana": 1}  
        df["State"] = df["State"].map(state_map)

        # One-hot encode other categorical columns
        df = pd.get_dummies(df, columns=["District", "Season", "Crop", "Soil_Type"], drop_first=True)

        # Ensure all expected columns exist (fill missing with 0)
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[expected_cols]

        # Scale numeric columns using trained scaler
        numeric_cols = ['Year', 'Area', 'Production', 'Temperature', 'Dew', 'Humidity',
                        'Percipitation', 'WindSpeed', 'Pressure', 'Rainfall_mm']
        df[numeric_cols] = scaler.transform(df[numeric_cols])


        prediction = model.predict(df)[0]

        return {"predicted_yield": prediction}

    except Exception as e:
        return {"error": str(e)}
