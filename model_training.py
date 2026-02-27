import logging, os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_curve, make_scorer, recall_score

def calculate_best_threshold(y_true, probs):
    """Finds the threshold that maximizes F2-Score (Prioritizes Recall)"""
    prec, rec, thresholds = precision_recall_curve(y_true, probs, pos_label="spam")
    
    # F2-Score Math: (1 + 2^2) * (Prec * Rec) / (2^2 * Prec + Rec)
    beta_sq = 2**2 
    f2_scores = (1 + beta_sq) * (prec * rec) / ((beta_sq * prec) + rec + 1e-9)
    
    best_idx = np.argmax(f2_scores[:-1])
    return thresholds[best_idx]

def run_model_training():
    logging.info("STEP 4: MODEL TRAINING STARTED")

    # --- CHAPTER 1: DATA PREPARATION ---
    df = pd.read_csv("data/sms_clean.csv")
    X, y = df["clean_light"].astype(str), df["label"]

    # Split into 70% Train, 15% Validation (to tune), 15% Test (to check)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # --- CHAPTER 2: THE BRAIN (PIPELINE) ---
    # Combine Word patterns + Character patterns (to catch misspellings)
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')),
            ("chars", TfidfVectorizer(analyzer="char", max_features=3000, ngram_range=(2, 5)))
        ])),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    # --- CHAPTER 3: TRAINING & OPTIMIZATION ---
    pipeline.fit(X_train, y_train)

    # Use the Validation set to find the best 'Sensitivity Dial' (Threshold)
    spam_idx = list(pipeline.classes_).index("spam")
    val_probs = pipeline.predict_proba(X_val)[:, spam_idx]
    best_threshold = calculate_best_threshold(y_val, val_probs)
    
    logging.info(f"Optimized Threshold: {best_threshold:.4f}")

    # --- CHAPTER 4: EVALUATION & SAVING ---
    # Test on completely unseen data using our custom threshold
    test_probs = pipeline.predict_proba(X_test)[:, spam_idx]
    y_pred = np.where(test_probs >= best_threshold, "spam", "ham")

    print("\n--- FINAL TEST REPORT ---")
    print(classification_report(y_test, y_pred))

    # Overfitting Check
    train_rec = recall_score(y_train, pipeline.predict(X_train), pos_label="spam")
    val_rec = recall_score(y_val, pipeline.predict(X_val), pos_label="spam")
    print(f"Train Recall: {train_rec:.2f} | Val Recall: {val_rec:.2f}")

    # Save everything in one toolbox
    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": best_threshold, "classes": pipeline.classes_}, 
                "models/spam_classifier_v2.pkl")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_model_training()