from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# 1. Define the Request Structure
class SMSRequest(BaseModel):
    text: str

# 2. Initialize FastAPI and Load Model
app = FastAPI(title="Spam Detection API")

# Load artifacts once at startup
model_artifacts = joblib.load("models/spam_classifier_v2.pkl")
pipeline = model_artifacts["pipeline"]
best_threshold = model_artifacts["threshold"]
classes = list(model_artifacts["classes"])
spam_idx = classes.index("spam")

@app.get("/")
def home():
    return {"message": "Spam Classifier API is running", "threshold": best_threshold}

@app.post("/predict")
def predict_sms(request: SMSRequest):
    # Convert input to format pipeline expects
    X = pd.Series([request.text])
    
    # Get probability
    spam_probability = pipeline.predict_proba(X)[0][spam_idx]
    
    # Determine label based on saved threshold
    label = "spam" if spam_probability >= best_threshold else "ham"
    
    return {
        "text": request.text,
        "prediction": label,
        "spam_probability": round(float(spam_probability), 4),
        "threshold_used": round(float(best_threshold), 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)