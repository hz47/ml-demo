import os
import sys
import logging
import uuid
from pathlib import Path‚
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
BATCH_SIZE = 100
SEARCH_LIMIT = 3


def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        check_compatibility=False
    )


def ensure_collection_exists(qdrant_client, collection_name):
    if not qdrant_client.collection_exists(collection_name):
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


def split_into_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def get_existing_clean_texts(qdrant_client, collection_name):
    points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10000)
    return {
        point.payload.get("clean_light")
        for point in points
        if point.payload.get("clean_light") is not None
    }


def create_embedding(openai_client, text):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def build_point(record, embedding):
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector={"sms_embedding": embedding},
        payload={
            "label": record["label"],
            "text": record["text"],
            "clean_light": record["clean_light"]
        }
    )


def load_sms_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clean_light"])
    return df[["label", "text", "clean_light"]].to_dict(orient="records")


def sync_sms_to_vector_store(openai_client, qdrant_client, records):
    existing_clean_texts = get_existing_clean_texts(qdrant_client, COLLECTION_NAME)
    new_records = [r for r in records if r["clean_light"] not in existing_clean_texts]

    if not new_records:
        logging.info("No new SMS records to add")
        return

    for batch in split_into_batches(new_records, BATCH_SIZE):
        points = []
        for record in batch:
            embedding = create_embedding(openai_client, record["clean_light"])
            points.append(build_point(record, embedding))

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        logging.info(f"Added batch of {len(points)} records")


def search_similar_sms(openai_client, qdrant_client, query_text):
    query_embedding = create_embedding(openai_client, query_text)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        using="sms_embedding",
        limit=SEARCH_LIMIT
    ).points

    logging.info(f"Search query: '{query_text}'")
    for result in results:
        logging.info(
            f"Found: '{result.payload['clean_light']}' "
            f"(Label: {result.payload['label']}, Score: {result.score:.3f})"
        )

    return results


def main():
    records = load_sms_data(SMS_CLEAN_PATH)

    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()

    ensure_collection_exists(qdrant_client, COLLECTION_NAME)

    sync_sms_to_vector_store(openai_client, qdrant_client, records)

    search_similar_sms(
        openai_client,
        qdrant_client,
        "I still need to go to the supermarket"
    )


if __name__ == "__main__":
    main()
