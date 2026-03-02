import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # New Estimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# --- Metadata Extractor (Keeping your proven logic) ---
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
    return thresholds[valid_idx[0]] if len(valid_idx) > 0 else 0.98

def run_training_logreg():
    df = pd.read_csv("data/processed/sms_clean.csv").dropna()
    X = df[['text', 'clean_light']]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # Features
    text_branch = FeatureUnion([
        ("words", TfidfVectorizer(max_features=4000, ngram_range=(1, 3))),
        ("chars", TfidfVectorizer(analyzer="char", max_features=2000, ngram_range=(3, 5)))
    ])

    meta_branch = Pipeline([
        ("ext", SMSMetadataExtractor()),
        ("scaler", MinMaxScaler())
    ])

    processor = ColumnTransformer([
        ("text_pipe", text_branch, "clean_light"),
        ("meta_pipe", meta_branch, "text")
    ])

    # --- LOGISTIC REGRESSION SETTINGS ---
    # C=1.0 is the inverse of regularization strength (lower = more regularization)
    # class_weight='balanced' helps with our imbalanced dataset
    base_lr = LogisticRegression(class_weight='balanced', solver='liblinear', C=1.0, random_state=42)
    
    pipeline = Pipeline([
        ("features", processor),
        ("clf", CalibratedClassifierCV(base_lr, method='sigmoid', cv=5))
    ])

    pipeline.fit(X_train, y_train)

    # Thresholding
    spam_idx = list(pipeline.classes_).index("spam")
    val_probs = pipeline.predict_proba(X_val)[:, spam_idx]
    best_threshold = find_high_precision_threshold(y_val, val_probs)

    # Final Test
    test_probs = pipeline.predict_proba(X_test)[:, spam_idx]
    y_pred = np.where(test_probs >= best_threshold, "spam", "ham")

    print(f"\n[LOGISTIC REGRESSION] Decision Threshold: {best_threshold:.4f}")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print(f"False Positives: {cm[0][1]}")

    joblib.dump({"model": pipeline, "threshold": best_threshold}, "models/logreg_spam_model.pkl")

if __name__ == "__main__":
    run_training_logreg()