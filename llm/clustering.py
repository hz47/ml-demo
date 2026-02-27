import os
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

# Load environment variables
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-rosssw0o8o0gwwck0c0o0484.116.203.135.75.sslip.io")

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
        random_state=42
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
        """Fetch data with paging until max_points or no more points."""
        points_all = []
        offset = None

        print(f"📡 Connecting to Qdrant collection: '{self.collection_name}'...")

        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True
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
            print("❌ No data found in the collection.")
            return False

        texts = []
        vectors = []

        for p in points_all:
            if self.payload_key not in p.payload:
                continue
            
            if isinstance(p.vector, dict):
                if self.vector_name not in p.vector:
                    continue
                vec = p.vector[self.vector_name]
            else:
                vec = p.vector

            texts.append(p.payload[self.payload_key])
            vectors.append(vec)

        if not texts:
            print(f"❌ No valid pairs found. Checked for key: '{self.payload_key}'")
            return False

        self.data = pd.DataFrame({"SMS": texts})
        self.embeddings_norm = normalize(np.array(vectors))
        print(f"✅ Loaded {len(self.data)} valid records.")
        return True

    def maybe_pca(self, n_components=50):
        """Optional PCA with safe component count."""
        n_samples, n_features = self.embeddings_norm.shape
        max_components = min(n_samples, n_features)

        if max_components <= 1:
            print("⚠️ Skipping PCA (not enough samples/features).")
            return

        safe_components = min(n_components, max_components)
        self.pca_model = PCA(n_components=safe_components, random_state=self.random_state)
        self.embeddings_norm = self.pca_model.fit_transform(self.embeddings_norm)
        print(f"✅ Applied PCA: new dims = {self.embeddings_norm.shape[1]}")

    def run_clustering(self, k_min=3, k_max=12, sample_size=2000):
        """Finds optimal k using sampled silhouette score."""
        n_samples = len(self.data)
        actual_k_max = min(k_max, max(k_min + 1, n_samples // 2))

        best_score = -1
        best_k = k_min

        print(f"🔎 Analyzing {n_samples} records for optimal k...")

        rng = np.random.default_rng(self.random_state)
        if n_samples > sample_size:
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
            X_sample = self.embeddings_norm[sample_idx]
        else:
            sample_idx = np.arange(n_samples)
            X_sample = self.embeddings_norm

        for k in range(k_min, actual_k_max + 1):
            km = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = km.fit_predict(self.embeddings_norm)
            score = silhouette_score(X_sample, labels[sample_idx])
            print(f"  > k={k}: Silhouette = {score:.3f}")

            if score > best_score:
                best_score = score
                best_k = k

        print(f"✅ Selected k={best_k} (Score: {best_score:.3f})")
        self.model = MiniBatchKMeans(n_clusters=best_k, random_state=self.random_state, n_init="auto")
        self.data["Cluster"] = self.model.fit_predict(self.embeddings_norm)
        return best_score

    
    def generate_cluster_labels(self, openai_client, model_name="gpt-4o-mini"):
        """Uses OpenAI with Pydantic to assign structured names to each cluster."""
        cluster_info = {}
        print("\n🏷️  Generating Structured AI labels...")

        for i in sorted(self.data["Cluster"].unique()):
            cluster_subset = self.data[self.data["Cluster"] == i]
            sample_size = min(15, len(cluster_subset))
            samples = cluster_subset["SMS"].sample(sample_size).tolist()
            samples_str = "\n- ".join(samples)

            completion = openai_client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a linguistic expert specializing in SMS intent classification."},
                    {"role": "user", "content": f"Analyze these SMS samples and provide a unique category:\n\n{samples_str}"}
                ],
                response_format=ClusterLabel,
            )

            result = completion.choices[0].message.parsed
            cluster_info[i] = result.category_name
            print(f" > Cluster {i}: {result.category_name} | Tone: {result.primary_tone}")
            print(f"   Reason: {result.justification}")

        self.data["Cluster_Label"] = self.data["Cluster"].map(cluster_info)
        return cluster_info

    def get_cluster_insights(self, top_n=5):
        """Prints representative samples from each cluster."""
        print("\n" + "=" * 30 + "\n📊 CLUSTER SUMMARY\n" + "=" * 30)
        for i in sorted(self.data["Cluster"].unique()):
            cluster_data = self.data[self.data["Cluster"] == i]
            print(f"\nCluster {i} ({len(cluster_data)} messages):")
            for s in cluster_data["SMS"].head(top_n).tolist():
                print(f" - {s.replace('\\n', ' ')[:100]}...")

    def export_results(self, filename="clustered_results.csv"):
        """Saves categorized data to CSV."""
        sort_col = "Cluster_Label" if "Cluster_Label" in self.data.columns else "Cluster"
        self.data.sort_values(sort_col).to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"\n💾 Data exported to {filename}")

    def visualize_tsne(self, max_points=3000):
        """Generates a t-SNE plot."""
        n_samples = len(self.data)
        X = self.embeddings_norm
        labels = self.data["Cluster"].values

        if n_samples > max_points:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n_samples, size=max_points, replace=False)
            X, labels = X[idx], labels[idx]

        print(f"🎨 Generating t-SNE plot...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=self.random_state, init="pca")
        reduced = tsne.fit_transform(X)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
        plt.title(f"t-SNE Clustering (k={self.data['Cluster'].nunique()})")
        plt.colorbar(scatter, label="Cluster ID")
        plt.show()

def main():
    clusterer = LargeSMSClusterer()

    if clusterer.fetch_data(max_points=10000):
        clusterer.maybe_pca(n_components=50)
        clusterer.run_clustering(k_min=3, k_max=10)
        clusterer.get_cluster_insights(top_n=5)
        
        if OPENAI_KEY:
            client = OpenAI(api_key=OPENAI_KEY)
            clusterer.generate_cluster_labels(client) 
        
        clusterer.export_results("clustered_sms_with_labels.csv")
        # clusterer.visualize_tsne(max_points=3000) # Optional

if __name__ == "__main__":
    main()
