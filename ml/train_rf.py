import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from config import MODELS_DIR
from ml.utils import load_data, find_high_precision_threshold, SMSMetadataExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_training():
    logger.info("Starting Random Forest training...")

    df = load_data()
    X = df[['text', 'clean_light']]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

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

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("features", processor),
        ("clf", CalibratedClassifierCV(rf_model, method='sigmoid', cv=5))
    ])

    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)

    spam_idx = list(pipeline.classes_).index("spam")
    val_probs = pipeline.predict_proba(X_val)[:, spam_idx]
    best_threshold = find_high_precision_threshold(y_val, val_probs, default=0.95)

    test_probs = pipeline.predict_proba(X_test)[:, spam_idx]
    y_pred = np.where(test_probs >= best_threshold, "spam", "ham")

    logger.info(f"[RANDOM FOREST] Decision Threshold: {best_threshold:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    model_path = MODELS_DIR / "rf_spam_model.pkl"
    joblib.dump({"model": pipeline, "threshold": best_threshold}, model_path)
    logger.info(f"Model saved to {model_path}")

    return {
        "model": "Random Forest",
        "threshold": best_threshold,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


if __name__ == "__main__":
    run_training()
