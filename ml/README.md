# ML Training Module

SMS Spam Classification models with hybrid TF-IDF + metadata features.

## Models Ranking

| Rank | Model | Threshold | Accuracy | Spam Recall | False Positives |
|:----:|-------|-----------|----------|-------------|:---------------:|
| 1 | **SVM** | 0.4901 | **99%** | **95%** | **1** |
| 2 | **Logistic Regression** | 0.7259 | 99% | 91% | **2** |
| 3 | **Random Forest** | 0.5600 | 98% | 88% | 2 |
| 4 | **Naive Bayes** | 0.9617 | 98% | 85% | **0** |

## Key Findings

- **Best Overall**: SVM (#1) achieves the highest accuracy (**99%**) with only **1 false positive** and the highest spam recall (**95%**).

- **Zero False Positives**: Naive Bayes (#4) flagged **no ham messages incorrectly** (100% precision on ham), but spam recall is lower (**85%**).

- **Trade-off Observed**: Models with **higher decision thresholds** (e.g., Naive Bayes 0.9617, Logistic Regression 0.7259) tend to have **fewer false positives** but **lower spam recall**. Lower thresholds (e.g., SVM 0.4901) increase spam detection at the cost of minimal false positives.

- **Balanced Performance**: Random Forest (#3) shows a **compromise** between spam recall (**88%**) and false positives (**2**) with a moderate threshold (**0.5600**).

## Features

### Text Features
- TF-IDF word n-grams (1-3)
- TF-IDF character n-grams (3-5)

### Metadata Features (per model)

**Naive Bayes:**
- Character count
- Average word length
- Sentence count
- Trigger keyword density

**Logistic Regression / SVM / Random Forest:**
- Character count
- Uppercase ratio
- Trigger keyword count
- Exclamation count

## Usage

### Train Models

```bash
# Train individual models
python -m ml.train_nb      # Naive Bayes
python -m ml.train_lr      # Logistic Regression
python -m ml.train_rf      # Random Forest
python -m ml.train_svm     # SVM

# Or use Makefile shortcuts
make train-nb
make train-lr
make train-rf
make train-svm
make train-all        # Train all 4 models

# Run predictions
make predict
# Or directly
python -m ml.predict
```

### Run Predictions

```bash
# Interactive command-line prediction
python -m ml.predict
```

The prediction module loads the trained model and lets you enter SMS messages interactively. Enter any text and press Enter to see if it's classified as **spam** or **ham**. Press Ctrl+C to exit.

Example:
```
Enter SMS (or Ctrl+C to exit): You won a free prize! Click here to claim.
2026-03-02 16:39:27,681 - INFO - Prediction: spam

Enter SMS (or Ctrl+C to exit): Hey, are we still meeting tonight?
2026-03-02 16:39:27,681 - INFO - Prediction: ham
```

## Models Location

Trained models are saved in `models/`:
- `final_spam_model.pkl` - Naive Bayes
- `logreg_spam_model.pkl` - Logistic Regression
- `rf_spam_model.pkl` - Random Forest
- `svm_spam_model.pkl` - SVM
