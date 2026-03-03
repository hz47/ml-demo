import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SMSSpamDetector:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY"),
            check_compatibility=False
        )
        self.collection_name = "sms_collection"
        self.vector_name = "sms_embedding"
        self.payload_key = "clean_light" 
        
        self.data = None
        self.embeddings = None

    def fetch_data(self, max_points=10000):
        logging.info(f"Fetching data from {self.collection_name}...")
        
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=max_points,
            with_vectors=True,
            with_payload=True,
        )

        if not points:
            logging.error("No data found in Qdrant.")
            return False

        records = []
        vectors = []
        for p in points:
            if isinstance(p.vector, dict) and self.vector_name in p.vector:
                vec = p.vector[self.vector_name]
            else:
                vec = p.vector
            
            if vec:
                records.append({
                    "point_id": p.id,
                    "SMS": p.payload.get(self.payload_key, "N/A")
                })
                vectors.append(vec)

        self.data = pd.DataFrame(records)
        self.embeddings = normalize(np.array(vectors))
        
        logging.info(f"Dataset size: {len(self.data)}")
        return True

    def run_spam_clustering(self):
        logging.info("Running Enhanced K-Means (k=2)...")
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.data["Cluster"] = kmeans.fit_predict(self.embeddings)
        
        score = silhouette_score(self.embeddings, self.data["Cluster"])
        logging.info(f"Cluster Quality (Silhouette Score): {score:.4f}")

        distances = kmeans.transform(self.embeddings)
        self.data["Confidence"] = distances.min(axis=1)

        counts = self.data["Cluster"].value_counts()
        spam_id = counts.idxmin()
        
        self.data["Label"] = self.data["Cluster"].map({spam_id: "spam", 1-spam_id: "ham"})
        logging.info(f"Clustering complete. Group {spam_id} identified as Spam.")

    def sync_to_qdrant(self):
        
        logging.info(f"Syncing {len(self.data)} labels to Qdrant...")
        
        success_count = 0
        for _, row in self.data.iterrows():
            try:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={
                        "cluster_id": int(row["Cluster"]),
                        "cluster_label": row["Label"],
                        "cluster_confidence": round(float(row["Confidence"]), 4)
                    },
                    points=[row["point_id"]],
                    wait=False  # Setting wait=False makes it much faster (asynchronous)
                )
                success_count += 1
                if success_count % 500 == 0:
                    logging.info(f"Progress: {success_count}/{len(self.data)} synced...")
            except Exception as e:
                logging.error(f"Failed to sync point {row['point_id']}: {e}")

        logging.info(f"Sync complete. Successfully updated {success_count} points.")
        
    def visualize(self):
        logging.info("Generating t-SNE plot...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(self.embeddings)

        plt.figure(figsize=(10, 6))
        for label in ["ham", "spam"]:
            mask = self.data["Label"] == label
            plt.scatter(reduced[mask, 0], reduced[mask, 1], label=label, alpha=0.6, s=15)
        
        plt.title("Upgraded SMS Clustering (Unique Messages)")
        plt.legend()
        plt.show()

def main():
    detector = SMSSpamDetector()
    if detector.fetch_data():
        detector.run_spam_clustering()
        #detector.sync_to_qdrant() use just when u need resync
        detector.visualize()

if __name__ == "__main__":
    main()