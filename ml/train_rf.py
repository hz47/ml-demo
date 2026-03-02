import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # New Estimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# --- Keep Metadata Extractor as is ---
class SMSMetadataExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        features = []
        triggers = ['win', 'won', 'prize', 'cash', 'claim', 'urgent', 'free', 'tone', 'voucher']
        for text in X:
            text = str(text).lower()
            char_count = len(text)
            upper_ratio = sum(1 for c in text if c.isupper()) / (char_count + 1)
            trigger_count = sum(1 for t in triggers if t in text)
            excl_count = text.count('!')
            features.append([char_count, upper_ratio, trigger_count, excl_count])
        return np.array(features)

def find_high_precision_threshold(y_true, probs, target_precision=1.0):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs, pos_label="spam")
    valid_idx = np.where(precisions >= target_precision)[0]
    return thresholds[valid_idx[0]] if len(valid_idx) > 0 else 0.95

def run_training_rf():
    df = pd.read_csv("data/processed/sms_clean.csv").dropna()
    X = df[['text', 'clean_light']]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # Features
    text_branch = FeatureUnion([
        ("words", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("chars", TfidfVectorizer(analyzer="char", max_features=1000, ngram_range=(3, 5)))
    ])

    meta_branch = Pipeline([
        ("ext", SMSMetadataExtractor()),
        ("scaler", MinMaxScaler())
    ])

    processor = ColumnTransformer([
        ("text_pipe", text_branch, "clean_light"),
        ("meta_pipe", meta_branch, "text")
    ])

    # --- RANDOM FOREST SETTINGS ---
    # n_estimators=200: Use 200 trees to vote
    # max_depth=None: Allow trees to grow until they are pure (or use a limit like 30)
    # class_weight="balanced": Helps because we have more Ham than Spam
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=30, 
        class_weight="balanced", 
        random_state=42,
        n_jobs=-1 # Use all CPU cores
    )

    pipeline = Pipeline([
        ("features", processor),
        ("clf", CalibratedClassifierCV(rf_model, method='sigmoid', cv=5))
    ])

    pipeline.fit(X_train, y_train)

    # Threshold Optimization
    spam_idx = list(pipeline.classes_).index("spam")
    val_probs = pipeline.predict_proba(X_val)[:, spam_idx]
    best_threshold = find_high_precision_threshold(y_val, val_probs)

    # Testing
    test_probs = pipeline.predict_proba(X_test)[:, spam_idx]
    y_pred = np.where(test_probs >= best_threshold, "spam", "ham")

    print(f"\n[RANDOM FOREST] Decision Threshold: {best_threshold:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump({"model": pipeline, "threshold": best_threshold}, "models/rf_spam_model.pkl")

if __name__ == "__main__":
    run_training_rf()