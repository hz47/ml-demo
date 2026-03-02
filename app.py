from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os

# Preprocessing: cleans raw SMS text into normalized tokens
# Required because model was trained on cleaned text (e.g., "walmart" -> "walmart", "$" -> "moneysymb")
from data.clean import clean_text_v3

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/svm_spam_model.pkl")

# Global model state - populated at startup, cleared on shutdown
model = {"pipeline": None, "threshold": 0.0, "spam_idx": 0}


def load_model():
    """Load trained model and artifacts from pickle file."""
    artifacts = joblib.load(MODEL_PATH)
    model["pipeline"] = artifacts["model"]
    model["threshold"] = artifacts["threshold"]
    # Find index of "spam" class in model's classes list
    # This is needed because predict_proba returns probabilities in class order
    model["spam_idx"] = list(model["pipeline"].classes_).index("spam")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifecycle handler - loads model on startup, clears on shutdown."""
    load_model()
    yield


app = FastAPI(title="Spam Detection API", lifespan=lifespan)


class SMSRequest(BaseModel):
    """Request body for /predict endpoint."""
    text: str


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: SMSRequest):
    """
    Predict if SMS is spam or ham.
    
    Flow:
    1. Preprocess: clean raw text using clean_text_v3 (critical for accuracy)
    2. Create DataFrame with 'text' (for metadata features) and 'clean_light' (for TF-IDF)
    3. Get probability from model
    4. Apply threshold to determine label
    """
    pipeline = model["pipeline"]
    spam_idx = model["spam_idx"]
    threshold = model["threshold"]

    # Step 1: Preprocess text - this is critical!
    # Model expects cleaned tokens (e.g., "URGENT: Walmart $1000" -> "urgent walmart moneysymb")
    cleaned = clean_text_v3(request.text)
    
    # Step 2: Match training DataFrame structure
    # The pipeline uses two separate branches (see ml/train_svm.py):
    #   - 'text_pipe' (TF-IDF): runs on 'clean_light' - needs normalized tokens
    #   - 'meta_pipe' (SMSMetadataExtractor): runs on 'text' - needs raw text for features
    # Without both columns, the model won't work correctly.
    X = pd.DataFrame({"text": [request.text], "clean_light": [cleaned]})
    
    # Step 3: Get probability of being spam
    prob = pipeline.predict_proba(X)[0][spam_idx]
    
    # Step 4: Apply threshold (default 0.5, but model uses optimized 0.56 for high precision)
    label = "spam" if prob >= threshold else "ham"

    return {
        "prediction": label,
        "spam_probability": round(float(prob), 4)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
