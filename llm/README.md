# SMS Embedding Clustering (Qdrant + KMeans)

This project fetches SMS embeddings from a Qdrant collection, normalizes them, optionally reduces dimensionality with PCA, and clusters them using MiniBatch K-Means. It exports a CSV of clustered results and provides a t-SNE visualization.

---

## ✅ Features

- Fetches SMS embeddings directly from Qdrant
- Normalizes vectors for better clustering performance
- Optional PCA for faster clustering
- Automatically selects best $k$ using silhouette score
- Exports clustered results to CSV
- Visualizes clusters using t-SNE

---

## ✅ How It Works

1. **Fetch Data**  
   Loads SMS texts and their vector embeddings from Qdrant.

2. **Normalize**  
   Applies $L_2$ normalization so $||x||_2 = 1$.

3. **(Optional) PCA**  
   Reduces embedding dimensions to speed up clustering.

4. **Find Best $k$**  
   Runs K-Means with multiple $k$ values and picks the best based on silhouette score.

5. **Cluster & Export**  
   Saves results to CSV and prints sample messages per cluster.

6. **Visualize**  
   t-SNE plot shows cluster separation.

---

## ✅ Project Structure

```
.
├── clustering.py
├── clustered_results.csv
└── README.md
```

---

## ✅ Installation

```bash
pip install numpy pandas matplotlib scikit-learn qdrant-client
```

---

## ✅ Usage

Edit the config values in `clustering.py`:

```python
URL = "http://your-qdrant-url"
COLLECTION = "sms_collection"
```

Run:

```bash
python clustering.py
```

---

## ✅ Output

### 1. CSV Export

`clustered_results.csv`

Example:

| SMS | Cluster |
|---|---|
| "Payment received" | 0 |
| "Your OTP is 1234" | 1 |

---

### 2. Console Summary

```
✅ Loaded 5000 valid records.
🔎 Analyzing 5000 records for optimal k...
  > k=3: Silhouette = 0.241
  > k=4: Silhouette = 0.278
✅ Selected k=4 (Score: 0.278)
```

---

### 3. t-SNE Plot

A 2D scatter plot of embeddings colored by cluster ID.

---

## ✅ Configuration Options

You can customize:

- `max_points` — max SMS pulled from Qdrant
- `batch_size` — Qdrant scroll batch size
- `n_components` — PCA components
- `k_min`, `k_max` — KMeans range
- `sample_size` — silhouette sample size

---

## ✅ Troubleshooting

### PCA error: `n_components` too large  
PCA components must satisfy:

$$
n\_{components} \le \min(n\_{samples}, n\_{features})
$$

The script now auto-adjusts this for small datasets.

---

## ✅ Example Command

```bash
python clustering.py
```
