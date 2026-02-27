import uuid
import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv
load_dotenv()

# Access keys using os.getenv
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")

# -------------------------------
# 1. Setup (OpenAI & Qdrant)
# -------------------------------
client = OpenAI(api_key=OPENAI_KEY)

qdrant = QdrantClient(
    url="http://qdrant-rosssw0o8o0gwwck0c0o0484.116.203.135.75.sslip.io",
    # If HTTPS is available, use:
    # url="https://qdrant-rosssw0o8o0gwwck0c0o0484.116.203.135.75.sslip.io",
    api_key=QDRANT_KEY,
    timeout=60,
    check_compatibility=False
)

collection_name = "sms_collection"

# -------------------------------
# 2. Collection Management
# -------------------------------
if not qdrant.collection_exists(collection_name):
    print(f"Erstelle neue Collection: {collection_name}")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "sms_embedding": models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            )
        }
    )

# -------------------------------
# 3. Load CSV (label, text, clean)
# -------------------------------
csv_path = r"C:\Users\uvyh73u\Documents\PROJECTSA1GROUP\scripts\a1_ai\data\sms_clean.csv"
df = pd.read_csv(csv_path)

# Keep only rows with clean text
df = df.dropna(subset=["clean"])

# Convert to list of dicts
records = df[["label", "text", "clean"]].to_dict(orient="records")

# -------------------------------
# 4. Sync to Qdrant (batched)
# -------------------------------
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def sync_sms_to_qdrant(records, batch_size=100):
    existing_points, _ = qdrant.scroll(collection_name=collection_name, limit=10000)

    existing_clean = set(
        p.payload.get("clean")
        for p in existing_points
        if p.payload.get("clean") is not None
    )

    new_records = [r for r in records if r["clean"] not in existing_clean]

    if not new_records:
        print("Keine neuen SMS zum Hinzufügen gefunden.")
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

        qdrant.upsert(collection_name=collection_name, points=points_to_upsert)
        print(f"Upserted batch of {len(points_to_upsert)}")

# Run sync
sync_sms_to_qdrant(records, batch_size=100)

# -------------------------------
# 5. Search Function
# -------------------------------
def search_similar_sms(query_text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )
    query_vector = response.data[0].embedding

    search_result = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="sms_embedding",
        limit=3
    ).points

    print(f"\nSuche nach: '{query_text}'")
    for res in search_result:
        print(
            f" -> Gefunden: '{res.payload['clean']}' "
            f"(Label: {res.payload['label']}, Score: {res.score:.3f})"
        )

# Test search
search_similar_sms("Ich muss noch in den Supermarkt")