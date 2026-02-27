import logging
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_curve, recall_score

PROCESSED_DATA_PATH = "data/processed/sms_clean.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier_v2.pkl")


def calculate_best_threshold(y_true, probs):
    prec, rec, thresholds = precision_recall_curve(y_true, probs, pos_label="spam")
    beta_sq = 2**2 
    f2_scores = (1 + beta_sq) * (prec * rec) / ((beta_sq * prec) + rec + 1e-9)
    best_idx = np.argmax(f2_scores[:-1])
    return thresholds[best_idx]


def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["clean_light"])
    df = df[df["clean_light"].astype(str).str.strip() != ""]
    X, y = df["clean_light"].astype(str), df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_pipeline():
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')),
            ("chars", TfidfVectorizer(analyzer="char", max_features=3000, ngram_range=(2, 5)))
        ])),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])


def evaluate_model(pipeline, X_test, y_test, threshold):
    spam_idx = list(pipeline.classes_).index("spam")
    test_probs = pipeline.predict_proba(X_test)[:, spam_idx]
    y_pred = np.where(test_probs >= threshold, "spam", "ham")
    
    print("\n--- FINAL TEST REPORT ---")
    print(classification_report(y_test, y_pred))


def run_model_training():
    logging.info("STEP 4: MODEL TRAINING STARTED")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(PROCESSED_DATA_PATH)
    
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    spam_idx = list(pipeline.classes_).index("spam")
    val_probs = pipeline.predict_proba(X_val)[:, spam_idx]
    best_threshold = calculate_best_threshold(y_val, val_probs)
    
    logging.info(f"Optimized Threshold: {best_threshold:.4f}")
    evaluate_model(pipeline, X_test, y_test, best_threshold)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline, 
        "threshold": best_threshold, 
        "classes": pipeline.classes_
    }, MODEL_PATH)
    
    logging.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_model_training()
