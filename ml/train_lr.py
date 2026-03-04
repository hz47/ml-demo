import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from config import MODELS_DIR
from ml.utils import load_data, find_high_precision_threshold, SMSMetadataExtractor, check_overfitting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_training():
    logger.info("Starting Logistic Regression training...")

    df = load_data()
    X = df[['text', 'clean_light']]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    text_branch = FeatureUnion([
        ("words", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ("chars", TfidfVectorizer(analyzer="char", max_features=1000, ngram_range=(3, 4)))
    ])

    meta_branch = Pipeline([
        ("ext", SMSMetadataExtractor()),
        ("scaler", MinMaxScaler())
    ])

    processor = ColumnTransformer([
        ("text_pipe", text_branch, "clean_light"),
        ("meta_pipe", meta_branch, "text")
    ])

    base_lr = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, random_state=42)

    pipeline = Pipeline([
        ("features", processor),
        ("clf", CalibratedClassifierCV(base_lr, method='sigmoid', cv=5))
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

    logger.info(f"[LOGISTIC REGRESSION] Decision Threshold: {best_threshold:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"False Positives: {cm[0][1]}")

    model_path = MODELS_DIR / "logreg_spam_model.pkl"
    joblib.dump({"model": pipeline, "threshold": best_threshold}, model_path)
    logger.info(f"Model saved to {model_path}")

    return {
        "model": "Logistic Regression",
        "threshold": best_threshold,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    run_training()
