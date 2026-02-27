from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/spam_classifier_v2.pkl")

model_state = {}

class SMSRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    try:
        artifacts = joblib.load(MODEL_PATH)
        model_state["pipeline"] = artifacts["pipeline"]
        model_state["threshold"] = artifacts["threshold"]
        
        classes = list(artifacts["classes"])
        model_state["spam_idx"] = classes.index("spam")
        
        logger.info(f"Model loaded successfully. Threshold: {artifacts['threshold']}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    logger.info("Shutting down API")
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
    try:
        pipeline = model_state["pipeline"]
        threshold = model_state["threshold"]
        spam_idx = model_state["spam_idx"]

        X = pd.Series([request.text])
        
        spam_probability = pipeline.predict_proba(X)[0][spam_idx]
        
        label = "spam" if spam_probability >= threshold else "ham"
        
        text_preview = request.text[:50] + "..." if len(request.text) > 50 else request.text
        logger.info(f"Request - text: '{text_preview}', prediction: {label}, probability: {round(spam_probability, 4)}")
        
        return {
            "text": request.text,
            "prediction": label,
            "spam_probability": round(float(spam_probability), 4),
            "threshold_used": round(float(threshold), 4)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
