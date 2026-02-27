from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/spam_classifier_v2.pkl")

# Global state to hold the model
model_state = {}

class SMSRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Load model artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    artifacts = joblib.load(MODEL_PATH)
    model_state["pipeline"] = artifacts["pipeline"]
    model_state["threshold"] = artifacts["threshold"]
    
    classes = list(artifacts["classes"])
    model_state["spam_idx"] = classes.index("spam")
    
    yield
    # Shutdown logic (if any)
    model_state.clear()

app = FastAPI(title="Spam Detection API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {
        "message": "Spam Classifier API is running", 
        "threshold": model_state.get("threshold")
    }

@app.post("/predict")
def predict_sms(request: SMSRequest):
    pipeline = model_state["pipeline"]
    threshold = model_state["threshold"]
    spam_idx = model_state["spam_idx"]

    # Convert input to format pipeline expects
    X = pd.Series([request.text])
    
    # Get probability
    spam_probability = pipeline.predict_proba(X)[0][spam_idx]
    
    # Determine label based on saved threshold
    label = "spam" if spam_probability >= threshold else "ham"
    
    return {
        "text": request.text,
        "prediction": label,
        "spam_probability": round(float(spam_probability), 4),
        "threshold_used": round(float(threshold), 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
