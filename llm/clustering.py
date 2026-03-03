import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from openai import OpenAI
from pydantic import BaseModel, Field

from config import CLUSTER_DIR, CLUSTER_IMG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
IMG_DIR = CLUSTER_IMG.parent / "img"


class ClusterLabel(BaseModel):
    category_name: str = Field(description="A unique 2-3 word category name for the cluster.")
    primary_tone: str = Field(description="The dominant tone (e.g., Emotional, Logistical, Urgent).")
    justification: str = Field(description="A brief explanation of why this label fits.")


class LargeSMSClusterer:
    def __init__(
        self,
        url=QDRANT_URL,
        collection_name="sms_collection",
        vector_name="sms_embedding",
        payload_key="clean",
        api_key=QDRANT_KEY,
        random_state=42,
    ):
        self.client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.payload_key = payload_key
        self.random_state = random_state

        self.data = None
        self.embeddings_norm = None
        self.model = None
        self.pca_model = None

    def fetch_data(self, max_points=10000, batch_size=512):
        """Fetch embeddings and texts from Qdrant."""
        points_all = []
        offset = None

        logging.info(f"Fetching from Qdrant: '{self.collection_name}'")

        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            if not points:
                break

            points_all.extend(points)
            if len(points_all) >= max_points:
                points_all = points_all[:max_points]
                break

            offset = next_page
            if offset is None:
                break

        if not points_all:
            logging.warning("No data found in collection.")
            return False

        texts, vectors = [], []
        for p in points_all:
            if self.payload_key not in p.payload:
                continue

            vec = p.vector.get(self.vector_name) if isinstance(p.vector, dict) else p.vector
            if vec is None:
                continue

            texts.append(p.payload[self.payload_key])
            vectors.append(vec)

        if not texts:
            logging.warning(f"No valid data for key: '{self.payload_key}'")
            return False

        self.data = pd.DataFrame({"SMS": texts})
        self.embeddings_norm = normalize(np.array(vectors))
        logging.info(f"Loaded {len(self.data)} records.")
        return True

    def apply_pca(self, n_components=50):
        """Reduce dimensionality with PCA."""
        n_samples, n_features = self.embeddings_norm.shape
        max_components = min(n_samples, n_features)

        if max_components <= 1:
            logging.warning("Skipping PCA (not enough data).")
            return

        n_components = min(n_components, max_components)
        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        self.embeddings_norm = self.pca_model.fit_transform(self.embeddings_norm)
        logging.info(f"PCA applied: {n_features} -> {n_components} dims")

    def find_optimal_k(self, k_min=3, k_max=12, sample_size=2000):
        """Find best cluster count using silhouette score."""
        n_samples = len(self.data)
        k_max = min(k_max, max(k_min + 1, n_samples // 2))

        rng = np.random.default_rng(self.random_state)
        sample_idx = rng.choice(n_samples, size=min(sample_size, n_samples), replace=False)
        X_sample = self.embeddings_norm[sample_idx]

        best_score, best_k = -1, k_min

        for k in range(k_min, k_max + 1):
            labels = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, n_init="auto").fit_predict(self.embeddings_norm)
            score = silhouette_score(X_sample, labels[sample_idx])
            logging.info(f"k={k}: silhouette={score:.3f}")

            if score > best_score:
                best_score, best_k = score, k

        logging.info(f"Selected k={best_k} (score={best_score:.3f})")
        return best_k

    def run_clustering(self, k):
        """Apply KMeans with given k."""
        self.model = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        self.data["Cluster"] = self.model.fit_predict(self.embeddings_norm)
        logging.info(f"Clustering complete: {k} clusters")

    def generate_labels(self, openai_client, model_name="gpt-4o-mini"):
        """Generate descriptive labels for each cluster using OpenAI."""
        cluster_info = {}

        for cluster_id in sorted(self.data["Cluster"].unique()):
            samples = self.data[self.data["Cluster"] == cluster_id]["SMS"].sample(min(15, len(self.data))).tolist()
            samples_str = "\n- ".join(samples)

            completion = openai_client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a linguistic expert specializing in SMS intent classification."},
                    {"role": "user", "content": f"Analyze these SMS samples and provide a unique category:\n\n{samples_str}"},
                ],
                response_format=ClusterLabel,
            )

            result = completion.choices[0].message.parsed
            cluster_info[cluster_id] = result.category_name
            logging.info(f"Cluster {cluster_id}: {result.category_name} | Tone: {result.primary_tone}")

        self.data["Cluster_Label"] = self.data["Cluster"].map(cluster_info)
        return cluster_info

    def get_insights(self, top_n=5):
        """Print representative samples from each cluster."""
        logging.info("CLUSTER SUMMARY")
        for cluster_id in sorted(self.data["Cluster"].unique()):
            cluster_data = self.data[self.data["Cluster"] == cluster_id]
            logging.info(f"Cluster {cluster_id} ({len(cluster_data)} messages):")
            for s in cluster_data["SMS"].head(top_n):
                logging.info(f"  - {s.replace(chr(10), ' ')[:100]}...")

    def log_summary(self, silhouette_score=None):
        """Log clustering metrics summary."""
        k = self.data["Cluster"].nunique()
        sizes = self.data["Cluster"].value_counts().sort_index().to_dict()
        
        logging.info("=" * 50)
        logging.info(f"CLUSTERING SUMMARY")
        logging.info(f"  Clusters: {k}")
        logging.info(f"  Total points: {len(self.data)}")
        
        if silhouette_score is not None:
            logging.info(f"  Silhouette score: {silhouette_score:.3f}")
        
        logging.info(f"  Cluster sizes: {sizes}")
        
        if "Cluster_Label" in self.data.columns:
            label_dist = self.data["Cluster_Label"].value_counts().to_dict()
            logging.info(f"  Labels: {label_dist}")
        
        logging.info("=" * 50)

    def export_csv(self, filename):
        """Export clustered data to CSV."""
        sort_col = "Cluster_Label" if "Cluster_Label" in self.data.columns else "Cluster"
        self.data.sort_values(sort_col).to_csv(CLUSTER_DIR / filename, index=False, encoding="utf-8-sig")
        logging.info(f"Exported to {filename}")

    def visualize(self, filename="clusters_tsne.png", max_points=3000):
        """Generate and save t-SNE visualization."""
        n_samples = len(self.data)
        X = self.embeddings_norm
        labels = self.data["Cluster"].values

        if n_samples > max_points:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n_samples, size=max_points, replace=False)
            X, labels = X[idx], labels[idx]

        logging.info("Generating t-SNE plot...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=self.random_state, init="pca")
        reduced = tsne.fit_transform(X)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
        plt.title(f"t-SNE Clustering (k={self.data['Cluster'].nunique()})")
        plt.colorbar(scatter, label="Cluster ID")

        IMG_DIR.mkdir(parents=True, exist_ok=True)
        filepath = IMG_DIR / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        logging.info(f"Visualization saved to {filepath}")

    def sync_to_qdrant(self):
        """Sync cluster labels back to Qdrant."""
        logging.info("Syncing labels to Qdrant...")

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=len(self.data),
            with_payload=True,
        )

        text_to_id = {p.payload["clean"]: p.id for p in points if "clean" in p.payload}

        success_count = 0
        for _, row in self.data.iterrows():
            p_id = text_to_id.get(row["SMS"])
            if p_id:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"cluster_id": int(row["Cluster"]), "cluster_label": row["Cluster_Label"]},
                    points=[p_id],
                )
                success_count += 1

        logging.info(f"Qdrant sync complete: {success_count} points updated.")


def main():
    clusterer = LargeSMSClusterer()

    if not clusterer.fetch_data(max_points=10000):
        return

    clusterer.apply_pca(n_components=50)

    optimal_k = clusterer.find_optimal_k(k_min=3, k_max=10)
    clusterer.run_clustering(optimal_k)
    clusterer.log_summary()
    clusterer.get_insights(top_n=5)

    if OPENAI_KEY:
        client = OpenAI(api_key=OPENAI_KEY)
        clusterer.generate_labels(client)
        #clusterer.sync_to_qdrant()

    clusterer.export_csv("clustered_sms_with_labels.csv")
    clusterer.visualize()


if __name__ == "__main__":
    main()
