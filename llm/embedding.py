import os
import logging
import hashlib
import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from config import SMS_CLEAN_PATH

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

COLLECTION_NAME = "sms_collection"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50
SEARCH_LIMIT = 3


def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )


def ensure_collection_exists(qdrant_client, collection_name):
    if qdrant_client.collection_exists(collection_name):
        return

    logging.info(f"Creating collection: {collection_name}")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "sms_embedding": models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE
            )
        }
    )


def generate_deterministic_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def split_into_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_sms_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clean_light"])
    return df[["label", "text", "clean_light"]].to_dict("records")  # type: ignore[attr-defined]


def get_embeddings(client, texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def build_points(records, embeddings):
    points = []
    for record, embedding in zip(records, embeddings):
        point_id = generate_deterministic_id(record["clean_light"])
        points.append(models.PointStruct(
            id=point_id,
            vector={"sms_embedding": embedding},
            payload={
                "label": record["label"],
                "text": record["text"],
                "clean_light": record["clean_light"]
            }
        ))
    return points


def sync_sms_to_vector_store(openai_client, qdrant_client, records):
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, batch in enumerate(split_into_batches(records, BATCH_SIZE), 1):
        try:
            texts = [r["text"] for r in batch]
            embeddings = get_embeddings(openai_client, texts)
            points = build_points(batch, embeddings)

            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            logging.info(f"Batch {batch_idx}/{total_batches}: upserted {len(points)} records")

        except Exception as e:
            logging.error(f"Batch {batch_idx}/{total_batches} failed: {e}")


def get_query_embedding(client, query_text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text
    )
    return response.data[0].embedding


def search_similar_sms(openai_client, qdrant_client, query_text):
    query_vector = get_query_embedding(openai_client, query_text)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="sms_embedding",
        limit=SEARCH_LIMIT
    ).points

    logging.info(f"Search Results for: '{query_text}' ---")
    for result in results:
        logging.info(
            f"Match: '{result.payload['text']}' | "
            f"Label: {result.payload['label']} | Score: {result.score:.3f}"
        )
    return results


def main():
    records = load_sms_data(SMS_CLEAN_PATH)

    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()

    ensure_collection_exists(qdrant_client, COLLECTION_NAME)
    sync_sms_to_vector_store(openai_client, qdrant_client, records)


if __name__ == "__main__":
    main()
