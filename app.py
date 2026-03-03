import os
import uvicorn
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from data.clean import clean_text_v3

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/svm_spam_model.pkl")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "sms_collection"

state = {
    "svm_pipeline": None,
    "svm_threshold": 0.0,
    "svm_spam_idx": 0,
    "openai": None,
    "qdrant": None
}

def load_svm():
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
    load_svm()
    
    state["openai"] = OpenAI(api_key=OPENAI_KEY)
    state["qdrant"] = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, check_compatibility=False)
    
    yield
    # clean up on shutdown
    state.clear()

app = FastAPI(title="SMS Intelligence API", lifespan=lifespan)

class SMSRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "svm_loaded": state.get("svm_pipeline") is not None}

@app.post("/predict")
def predict(request: SMSRequest):
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

@app.post("/cluster")
def cluster(request: SMSRequest):
    """Finds the semantic category using Embeddings and Qdrant."""
    if not state["openai"] or not state["qdrant"]:
        return {"error": "AI/Vector clients not initialized"}

    # Generate Embedding using RAW text (to match our Sync script logic)
    response = state["openai"].embeddings.create(
        model="text-embedding-3-small",
        input=request.text 
    )
    vector = response.data[0].embedding

    #Vector Search in Qdrant
    search_result = state["qdrant"].query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using="sms_embedding",
        limit=1,
        with_payload=True
    ).points

    if not search_result:
        return {"semantic_category": "unknown", "similarity_score": 0.0}

    match = search_result[0]
    
    # AAdjust threshold. 
    if match.score < 0.60: 
        return {
            "semantic_category": "Uncertain / New Topic",
            "similarity_score": round(match.score, 4),
            "nearest_cluster_id": match.payload.get("cluster_id"),
            "note": "Low similarity to existing database need llm feedback or human."
        }

    return {
        "semantic_category": match.payload.get("cluster_label", "uncategorized"),
        "similarity_score": round(match.score, 4),
        "cluster_id": match.payload.get("cluster_id"),
        "cluster_confidence_at_indexing": match.payload.get("cluster_confidence")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)