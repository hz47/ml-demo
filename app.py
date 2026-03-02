import os
import uvicorn
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient

# Import your custom cleaning logic
from data.clean import clean_text_v3

# Environment Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/svm_spam_model.pkl")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "sms_collection"

# Global state to hold models and clients
state = {
    "svm_pipeline": None,
    "svm_threshold": 0.0,
    "svm_spam_idx": 0,
    "openai": None,
    "qdrant": None
}

def load_svm():
    """Load the local SVM spam model artifacts."""
    if os.path.exists(MODEL_PATH):
        artifacts = joblib.load(MODEL_PATH)
        state["svm_pipeline"] = artifacts["model"]
        state["svm_threshold"] = artifacts["threshold"]
        state["svm_spam_idx"] = list(state["svm_pipeline"].classes_).index("spam")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown of models and database connections."""
    # 1. Load Local Model
    load_svm()
    
    # 2. Initialize Cloud Clients
    state["openai"] = OpenAI(api_key=OPENAI_KEY)
    state["qdrant"] = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, check_compatibility=False)
    
    yield
    # Clean up on shutdown if necessary
    state.clear()

app = FastAPI(title="SMS Intelligence API", lifespan=lifespan)

class SMSRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "svm_loaded": state.get("svm_pipeline") is not None}

# --- ENDPOINT 1: FAST BINARY CLASSIFICATION ---
@app.post("/predict")
def predict(request: SMSRequest):
    """Predicts if an SMS is Spam or Ham using the local SVM model."""
    if not state["svm_pipeline"]:
        return {"error": "SVM model not loaded"}

    # Preprocess
    cleaned = clean_text_v3(request.text)
    
    # Format for pipeline (text for metadata, clean_light for TF-IDF)
    X = pd.DataFrame({"text": [request.text], "clean_light": [cleaned]})
    
    # Probability and Labeling
    prob = state["svm_pipeline"].predict_proba(X)[0][state["svm_spam_idx"]]
    label = "spam" if prob >= state["svm_threshold"] else "ham"

    return {
        "prediction": label,
        "spam_probability": round(float(prob), 4)
    }

# --- ENDPOINT 2: SEMANTIC CLUSTERING (NEAR-REAL-TIME) ---
@app.post("/cluster")
def cluster(request: SMSRequest):
    """Finds the semantic category of an SMS using Embeddings and Qdrant."""
    if not state["openai"] or not state["qdrant"]:
        return {"error": "AI/Vector clients not initialized"}

    # 1. Preprocess
    cleaned = clean_text_v3(request.text)
    
    # 2. Generate Embedding
    response = state["openai"].embeddings.create(
        model="text-embedding-3-small",
        input=cleaned
    )
    vector = response.data[0].embedding

    # 3. Vector Search in Qdrant
    # This finds the 'Nearest Neighbor' among your pre-clustered messages
    search_result = state["qdrant"].query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using="sms_embedding", # Matches your previous Qdrant setup
        limit=1,
        with_payload=True
    ).points

    if not search_result:
        return {"category": "uncategorized", "confidence_score": 0.0}

    match = search_result[0]
    return {
        "semantic_category": match.payload.get("cluster_label", "unknown"),
        "similarity_score": round(match.score, 4),
        "cluster_id": match.payload.get("cluster_id")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)