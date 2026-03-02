import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from config import SMS_CLEAN_PATH, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROCESSED_DATA_PATH = SMS_CLEAN_PATH
MODEL_V1_PATH = MODELS_DIR / "spam_classifier_v2.pkl"
MODEL_NB_PATH = MODELS_DIR / "spam_classifier_nb.pkl"


def load_test_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df.dropna(subset=["clean_light"])
    df = df[df["clean_light"].astype(str).str.strip() != ""]
    X, y = df["clean_light"].astype(str), df["label"]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    return X_test, y_test


def load_model(model_path):
    artifacts = joblib.load(model_path)
    return artifacts["pipeline"], artifacts["threshold"], list(artifacts["classes"])


def predict_with_threshold(pipeline, X, threshold):
    spam_idx = list(pipeline.classes_).index("spam")
    probs = pipeline.predict_proba(X)[:, spam_idx]
    return np.where(probs >= threshold, "spam", "ham")


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_ham": precision_score(y_true, y_pred, pos_label="ham"),
        "precision_spam": precision_score(y_true, y_pred, pos_label="spam"),
        "recall_ham": recall_score(y_true, y_pred, pos_label="ham"),
        "recall_spam": recall_score(y_true, y_pred, pos_label="spam"),
        "f1_ham": f1_score(y_true, y_pred, pos_label="ham"),
        "f1_spam": f1_score(y_true, y_pred, pos_label="spam"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def measure_inference_time(pipeline, X, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        pipeline.predict(X)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def print_table(results):
    headers = ["Metric", "Logistic Regression", "Naive Bayes", "Winner"]
    rows = [
        ["Accuracy", f"{results['lr']['accuracy']:.4f}", f"{results['nb']['accuracy']:.4f}", results['winners']['accuracy']],
        ["Precision (Ham)", f"{results['lr']['precision_ham']:.4f}", f"{results['nb']['precision_ham']:.4f}", results['winners']['precision_ham']],
        ["Precision (Spam)", f"{results['lr']['precision_spam']:.4f}", f"{results['nb']['precision_spam']:.4f}", results['winners']['precision_spam']],
        ["Recall (Ham)", f"{results['lr']['recall_ham']:.4f}", f"{results['nb']['recall_ham']:.4f}", results['winners']['recall_ham']],
        ["Recall (Spam)", f"{results['lr']['recall_spam']:.4f}", f"{results['nb']['recall_spam']:.4f}", results['winners']['recall_spam']],
        ["F1 (Ham)", f"{results['lr']['f1_ham']:.4f}", f"{results['nb']['f1_ham']:.4f}", results['winners']['f1_ham']],
        ["F1 (Spam)", f"{results['lr']['f1_spam']:.4f}", f"{results['nb']['f1_spam']:.4f}", results['winners']['f1_spam']],
        ["F1 (Macro)", f"{results['lr']['f1_macro']:.4f}", f"{results['nb']['f1_macro']:.4f}", results['winners']['f1_macro']],
        ["F1 (Weighted)", f"{results['lr']['f1_weighted']:.4f}", f"{results['nb']['f1_weighted']:.4f}", results['winners']['f1_weighted']],
        ["Inference Time (ms)", f"{results['lr']['inference_time']:.4f}", f"{results['nb']['inference_time']:.4f}", results['winners']['inference_time']],
    ]
    
    col_widths = [max(len(row[i]) for row in rows + [headers]) for i in range(4)]
    
    def print_row(row):
        return f"│ {row[0]:<{col_widths[0]}} │ {row[1]:<{col_widths[1]}} │ {row[2]:<{col_widths[2]}} │ {row[3]:<{col_widths[3]}} │"
    
    separator = "─" * (sum(col_widths) + 13)
    
    print(f"\n{'═' * (sum(col_widths) + 13)}")
    print(f"{'MODEL BENCHMARK COMPARISON':^{sum(col_widths) + 12}}")
    print(f"{'═' * (sum(col_widths) + 13)}")
    print(print_row(headers))
    print(separator)
    
    for row in rows:
        print(print_row(row))
    
    print(separator)


def print_confusion_matrices(results, y_test):
    print("\n" + "═" * 50)
    print("CONFUSION MATRICES")
    print("═" * 50)
    
    for model_name, cm in [("Logistic Regression", results['lr']['confusion_matrix']), 
                           ("Naive Bayes", results['nb']['confusion_matrix'])]:
        print(f"\n{model_name}:")
        print(f"                 Predicted")
        print(f"              ┌─────────┬─────────┐")
        print(f"              │   Ham   │  Spam   │")
        print(f"──────────────┼─────────┼─────────┤")
        print(f"Actual Ham    │   {cm[0,0]:4d}  │  {cm[0,1]:4d}  │")
        print(f"──────────────┼─────────┼─────────┤")
        print(f"Actual Spam   │   {cm[1,0]:4d}  │  {cm[1,1]:4d}  │")
        print(f"──────────────┴─────────┴─────────┘")


def determine_winner(metric, lr_val, nb_val, lower_is_better=False):
    if lower_is_better:
        return "LR" if lr_val < nb_val else ("NB" if nb_val < lr_val else "TIE")
    return "LR" if lr_val > nb_val else ("NB" if nb_val > lr_val else "TIE")


def run_benchmark():
    logging.info("Loading models and test data")
    
    X_test, y_test = load_test_data()
    logging.info(f"Test set size: {len(X_test)}")
    
    lr_pipeline, lr_threshold, lr_classes = load_model(MODEL_V1_PATH)
    nb_pipeline, nb_threshold, nb_classes = load_model(MODEL_NB_PATH)
    
    logging.info(f"Logistic Regression - Threshold: {lr_threshold:.4f}")
    logging.info(f"Naive Bayes - Threshold: {nb_threshold:.4f}")
    
    logging.info("Making predictions")
    
    y_pred_lr = predict_with_threshold(lr_pipeline, X_test, lr_threshold)
    y_pred_nb = predict_with_threshold(nb_pipeline, X_test, nb_threshold)
    
    lr_metrics = calculate_metrics(y_test, y_pred_lr)
    nb_metrics = calculate_metrics(y_test, y_pred_nb)
    
    lr_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred_lr, labels=["ham", "spam"])
    nb_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred_nb, labels=["ham", "spam"])
    
    logging.info("Measuring inference time")
    
    lr_mean, lr_std = measure_inference_time(lr_pipeline, X_test)
    nb_mean, nb_std = measure_inference_time(nb_pipeline, X_test)
    
    lr_metrics["inference_time"] = lr_mean * 1000
    nb_metrics["inference_time"] = nb_mean * 1000
    
    logging.info(f"Logistic Regression: {lr_mean*1000:.4f}ms (±{lr_std*1000:.4f}ms)")
    logging.info(f"Naive Bayes: {nb_mean*1000:.4f}ms (±{nb_std*1000:.4f}ms)")
    
    winners = {
        metric: determine_winner(metric, lr_metrics[metric], nb_metrics[metric])
        for metric in lr_metrics.keys()
        if metric != "confusion_matrix"
    }
    
    results = {
        "lr": lr_metrics,
        "nb": nb_metrics,
        "winners": winners
    }
    
    print_table(results)
    print_confusion_matrices(results, y_test)
    
    lr_score = sum(1 for w in winners.values() if w == "LR")
    nb_score = sum(1 for w in winners.values() if w == "NB")
    
    logging.info(f"Logistic Regression wins: {lr_score}/{len(winners)} metrics")
    logging.info(f"Naive Bayes wins: {nb_score}/{len(winners)} metrics")
    
    if lr_score > nb_score:
        logging.info("RECOMMENDED MODEL: Logistic Regression")
    elif nb_score > lr_score:
        logging.info("RECOMMENDED MODEL: Naive Bayes")
    else:
        logging.info("RECOMMENDED MODEL: Both models are equivalent")
    
    logging.info("For spam detection, prioritize: High Recall (Spam) and High F1 (Spam)")


if __name__ == "__main__":
    run_benchmark()
