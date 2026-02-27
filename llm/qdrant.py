from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv()

# Access keys using os.getenv
QDRANT_KEY = os.getenv("QDRANT_API_KEY")

qdrant = QdrantClient(
    url="http://qdrant-rosssw0o8o0gwwck0c0o0484.116.203.135.75.sslip.io",
    api_key=QDRANT_KEY,  # optional, falls aktiviert
    check_compatibility=False

)# qdrant = QdrantClient(url="https://<dein-remote-qdrant>", api_key="<API_KEY>")  # Cloud


# Alle Collections abrufen
collections = qdrant.get_collections()
print("Collections in Qdrant:")
print(collections)