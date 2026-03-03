# ML Demo - SMS Spam Intelligence

FastAPI SMS spam detection, combining ML classification + LLM semantic clustering

## Module Documentation

- [ML Models](./ml/README.md) - Training, performance rankings, features
- [LLM Embeddings](./llm/README.md) - Semantic search, clustering insights
- [Data Analysis](./analysis/README.md) - Dataset stats, visualizations

## Tech Stack

| Category | Technology |
|----------|------------|
| API | FastAPI, Pydantic, Uvicorn |
| ML | scikit-learn (SVM, LR, RF, NB), TF-IDF |
| LLM | OpenAI text-embedding-3-small |
| Vector DB | Qdrant |
| Data | pandas, numpy, NLTK |

## Project Structure

```
ml-demo/
├── app.py              # FastAPI application (main entry point)
├── config.py           # Project paths configuration
├── Makefile            # CLI shortcuts for all tasks
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
├── ml/                 # ML training module
│   ├── train_*.py      # Model trainers (NB, LR, RF, SVM)
│   └── predict.py      # CLI prediction tool
├── llm/                # LLM embeddings & clustering
│   ├── embedding.py    # Generate OpenAI embeddings
│   └── clustering.py   # Create clustering using embeddings
├── analysis/           # Data analysis 
│   ├── *.py            # scripts for analysis 
│   └── img/            # Generated charts
├── models/             # Trained model files (.pkl)
├── data/               # Raw & processed datasets
└── tests/              # Test suite
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Classify SMS as spam/ham using SVM |
| POST | `/cluster` | Find semantic category using embeddings + Qdrant |

### Request/Response Examples

**POST /predict**
```json
// Request
{
  "text": "FREE entry into our £250 weekly competition! Text WIN to 80082 now! Std txt rates apply. 16+ only."
}

// Response
{
  "prediction": "spam",
  "spam_probability": 0.9997
}
```

**POST /cluster**
```json
// Request
{
  "text": "Congratulations! You have WON a £500 Amazon Gift Card! Claim now by calling 09061701461. T&Cs apply. Reply STOP to opt out."
}
// Response
{
  "semantic_category": "spam",
  "similarity_score": 0.7275,
  "cluster_id": 1,
  "cluster_confidence_at_indexing": 0.704
}
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Add OPENAI_API_KEY and QDRANT_API_KEY to .env

# Train SVM model (required for /predict)
make train-svm

# Generate Embeddings (required for /cluster)
make make embedding

# Generate clusters (required for /cluster)
make cluster
# Start API server

make api
# Open http://localhost:8000/docs for interactive Swagger UI


```

## Makefile Commands

### Training
```bash
make train-svm      # Train SVM model (recommended)
make train-all     # Train all 4 models
make predict       # Interactive CLI prediction
```

### Analysis
```bash
make distribution     # Label distribution
make word-analysis   # Top words (spam vs ham)
make ngrams          # Bigrams/trigrams analysis
make report-analysis # Generate visual report
```

### LLM
```bash
make embedding  # Generate embeddings via OpenAI
make cluster    # Run K-Means clustering
```

## Running app.py Directly

```bash
python app.py
```

This runs the FastAPI server on `http://0.0.0.0:8000`. Requires:
- Trained model at `models/svm_spam_model.pkl`
- Valid OpenAI and Qdrant credentials in `.env`
