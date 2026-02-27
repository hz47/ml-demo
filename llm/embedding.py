import uuid
import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-rosssw0o8o0gwwck0c0o0484.116.203.135.75.sslip.io")

COLLECTION_NAME = "sms_collection"
CSV_PATH = "data/processed/sms_clean.csv"

def get_clients():
    client = OpenAI(api_key=OPENAI_KEY)
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_KEY,
        timeout=60,
        check_compatibility=False
    )
    return client, qdrant

def ensure_collection(qdrant, name):
    if not qdrant.collection_exists(name):
        print(f"Creating new collection: {name}")
        qdrant.create_collection(
            collection_name=name,
            vectors_config={
                "sms_embedding": models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            }
        )

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def sync_sms_to_qdrant(client, qdrant, records, batch_size=100):
    existing_points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=10000)

    existing_clean = set(
        p.payload.get("clean")
        for p in existing_points
        if p.payload.get("clean") is not None
    )

    new_records = [r for r in records if r["clean"] not in existing_clean]

    if not new_records:
        print("No new SMS found to add.")
        return

    for batch_records in chunked(new_records, batch_size):
        points_to_upsert = []
        for r in batch_records:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=r["clean"]
            )
            embedding = response.data[0].embedding

            points_to_upsert.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"sms_embedding": embedding},
                    payload={
                        "label": r["label"],
                        "text": r["text"],
                        "clean": r["clean"]
                    }
                )
            )

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)
        print(f"Upserted batch of {len(points_to_upsert)}")

def search_similar_sms(client, qdrant, query_text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )
    query_vector = response.data[0].embedding

    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="sms_embedding",
        limit=3
    ).points

    print(f"\nSearching for: '{query_text}'")
    for res in search_result:
        print(
            f" -> Found: '{res.payload['clean']}' "
            f"(Label: {res.payload['label']}, Score: {res.score:.3f})"
        )

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
    else:
        client, qdrant = get_clients()
        ensure_collection(qdrant, COLLECTION_NAME)
        
        df = pd.read_csv(CSV_PATH)
        df = df.dropna(subset=["clean"])
        records = df[["label", "text", "clean"]].to_dict(orient="records")
        
        sync_sms_to_qdrant(client, qdrant, records)
        search_similar_sms(client, qdrant, "I still need to go to the supermarket")
