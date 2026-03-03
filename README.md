# ML Demo - SMS Spam Intelligence

FastAPI-powered SMS spam detection combining ML classification + LLM semantic clustering.

## Tech Stack

| Category | Technology |
|----------|------------|
| API | FastAPI, Pydantic, Uvicorn |
| ML | scikit-learn (SVM, LR, RF, NB), TF-IDF |
| LLM | OpenAI text-embedding-3-small |
| Vector DB | Qdrant |
| Testing | pytest, pytest-cov |
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
│   └── clustering.py   # K-Means clustering on embeddings
├── analysis/           # Data analysis & visualization
│   ├── *.py            # Various analysis scripts
│   └── img/            # Generated charts
├── models/             # Trained model files (.pkl)
├── data/               # Raw & processed datasets
└── tests/              # Test suite
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check, returns SVM model status |
| POST | `/predict` | Classify SMS as spam/ham using SVM |
| POST | `/cluster` | Find semantic category using embeddings + Qdrant |

### Request/Response Examples

**POST /predict**
```json
// Request
{"text": "Win $1000 now! Click here."}

// Response
{"prediction": "spam", "spam_probability": 0.9823}
```

**POST /cluster**
```json
// Request
{"text": "Your prescription is ready for pickup"}

// Response
{
  "semantic_category": "Medical/Health",
  "similarity_score": 0.8472,
  "cluster_id": 3,
  "cluster_confidence_at_indexing": 0.89
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

### API & Testing
```bash
make api        # Start FastAPI server (port 8000)
make test       # Run all tests
make test-cov   # Run tests with coverage
```

## Testing

```bash
# Run all tests
make test

# Run with coverage report
make test-cov
```

## Module Documentation

- [ML Models](./ml/README.md) - Training, performance rankings, features
- [LLM Embeddings](./llm/README.md) - Semantic search, clustering insights
- [Data Analysis](./analysis/README.md) - Dataset stats, visualizations

## Running app.py Directly

```bash
python app.py
```

This runs the FastAPI server on `http://0.0.0.0:8000`. Requires:
- Trained model at `models/svm_spam_model.pkl`
- Valid OpenAI and Qdrant credentials in `.env`
