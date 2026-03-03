from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

points, _ = client.scroll(collection_name="sms_collection", limit=2, with_payload=True)

for p in points:
    print(f"id: {p.id}")
    print(f"payload: {p.payload}")
    print("-" * 30)