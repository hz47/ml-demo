import os
import logging
<<<<<<< HEAD
import uuid
import hashlib
from pathlib import Path
=======
import hashlib
>>>>>>> 49c38dd062e56690fa2dc74ece3adcf68b13335b
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
<<<<<<< HEAD
BATCH_SIZE = 50  
SEARCH_LIMIT = 3

def get_openai_client():
    """Initializes the OpenAI client using the environment API key."""
    return OpenAI(api_key=OPENAI_API_KEY)

def get_qdrant_client():
    """Initializes the Qdrant client for vector storage."""
=======
BATCH_SIZE = 50
SEARCH_LIMIT = 3


def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


def get_qdrant_client():
>>>>>>> 49c38dd062e56690fa2dc74ece3adcf68b13335b
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
<<<<<<< HEAD
    )

def ensure_collection_exists(qdrant_client, collection_name):
    """Checks if the collection exists; if not, creates it with Cosine distance."""
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
=======
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
>>>>>>> 49c38dd062e56690fa2dc74ece3adcf68b13335b
        )
    return results

<<<<<<< HEAD
def generate_deterministic_id(text):
    """
    unique hex ID based on the text content 
    This prevents duplicate entries in Qdrant even if you run the script twice.
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def split_into_batches(items, batch_size):
    """Generator to yield chunks of data for batch processing."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def load_sms_data(csv_path):
    """Loads CSV, drops empty rows, and converts to a list of dictionaries."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clean_light"])
    # Keeping 'label', 'text' (raw), and 'clean_light' (processed)
    return df[["label", "text", "clean_light"]].to_dict(orient="records")

# 5. Core Logic: Embedding and Syncing
def sync_sms_to_vector_store(openai_client, qdrant_client, records):
    """
    Processes records in batches, gets embeddings from OpenAI in one call per batch,
    and upserts them into Qdrant.
    """
    logging.info(f"Starting sync for {len(records)} records...")

    for batch in split_into_batches(records, BATCH_SIZE):
        try:
            # xtract all texts in  batch to embed in one call api
            texts_to_embed = [r["text"] for r in batch]
            
            # Batch embedding call
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts_to_embed
            )
            embeddings = [item.embedding for item in response.data]

            points = []
            for record, vector in zip(batch, embeddings):
                # use a hash of the cleaned text as the ID to avoid duplicates
                point_id = generate_deterministic_id(record["clean_light"])
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector={"sms_embedding": vector},
                    payload={
                        "label": record["label"],
                        "text": record["text"],
                        "clean_light": record["clean_light"]
                    }
                ))

            # batch upload to Qdrant
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            logging.info(f"Successfully upserted batch of {len(points)} records.")

        except Exception as e:
            logging.error(f"Failed to process batch: {e}")

def search_similar_sms(openai_client, qdrant_client, query_text):
    """Converts a user query into a vector and finds the top matches in Qdrant."""
    # Embed the query
    query_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text
    )
    query_vector = query_response.data[0].embedding

    # Search Qdrant
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="sms_embedding",
        limit=SEARCH_LIMIT
    ).points

    logging.info(f"--- Search Results for: '{query_text}' ---")
    for result in results:
        logging.info(
            f"Match: '{result.payload['text']}' | "
            f"Label: {result.payload['label']} | Score: {result.score:.3f}"
        )
    return results
=======
>>>>>>> 49c38dd062e56690fa2dc74ece3adcf68b13335b

def main():
    records = load_sms_data(SMS_CLEAN_PATH)

    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()

    ensure_collection_exists(qdrant_client, COLLECTION_NAME)
<<<<<<< HEAD

    sync_sms_to_vector_store(openai_client, qdrant_client, records)

    # Test 
    # search_similar_sms(
    #     openai_client,
    #     qdrant_client,
    #     "Winner! You’ve been chosen for a brand new iPhone 15! Click http://claim-now-prize.com to confirm delivery."
    # )

if __name__ == "__main__":
    main()
=======
    sync_sms_to_vector_store(openai_client, qdrant_client, records)


if __name__ == "__main__":
    main()
>>>>>>> 49c38dd062e56690fa2dc74ece3adcf68b13335b
