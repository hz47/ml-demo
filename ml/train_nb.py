import logging
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from config import MODELS_DIR
from ml.utils import load_data, find_high_precision_threshold, check_overfitting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class SMSMetadataExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        triggers = ['win', 'won', 'prize', 'cash', 'claim', 'urgent', 'free', 'tone', 'voucher']
        for text in X:
            text = str(text)
            char_count = len(text)
            words = text.split()
            word_count = len(words) + 1

            avg_word_len = char_count / word_count
            sent_count = text.count('.') + text.count('!') + text.count('?')
            trigger_count = sum(1 for t in triggers if t in text.lower())

            features.append([
                char_count,
                avg_word_len,
                sent_count,
                trigger_count
            ])
        return np.array(features)


def run_training():
    logger.info("Starting Naive Bayes training...")

    df = load_data()
    X = df[['text', 'clean_light']]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

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

    pipeline = Pipeline([
        ("features", processor),
        ("clf", CalibratedClassifierCV(MultinomialNB(alpha=0.1), method='sigmoid', cv=5))
    ])

    logger.info("Training pipeline...")
    logger.info("Checking for overfitting...")
    overfit_result = check_overfitting(pipeline, X_train, y_train, cv=5)
    logger.info(f"Train Score: {overfit_result['train_score']:.4f}")
    logger.info(f"CV Score: {overfit_result['cv_score_mean']:.4f} (+/- {overfit_result['cv_score_std']:.4f})")
    logger.info(f"Overfitting Gap: {overfit_result['overfitting_gap']:.4f}")
    if overfit_result['is_overfitting']:
        logger.warning("OVERFITTING DETECTED - Consider tuning hyperparameters")
    
    pipeline.fit(X_train, y_train)

    spam_idx = list(pipeline.classes_).index("spam")
    val_probs = pipeline.predict_proba(X_val)[:, spam_idx]
    best_threshold = find_high_precision_threshold(y_val, val_probs)

    test_probs = pipeline.predict_proba(X_test)[:, spam_idx]
    y_pred = np.where(test_probs >= best_threshold, "spam", "ham")

    logger.info(f"[Naive Bayes]] Decision Threshold: {best_threshold:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = MODELS_DIR / "final_spam_model.pkl"
    joblib.dump({"model": pipeline, "threshold": best_threshold}, model_path)
    logger.info(f"Model saved to {model_path}")

    return {
        "model": "Naive Bayes",
        "threshold": best_threshold,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    run_training()
