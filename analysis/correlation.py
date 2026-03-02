import logging
import pandas as pd
from .utils import load_data, count_sentences


def compute_correlation_matrix():
    df = load_data()
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    df["sentence_count"] = df["text"].astype(str).apply(count_sentences)

    corr_matrix = df[["text_length", "word_count", "sentence_count"]].corr()
    return corr_matrix.round(3)


def compute_correlation_by_label():
    df = load_data()
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    df["sentence_count"] = df["text"].astype(str).apply(count_sentences)

    result = {}
    for label in df["label"].unique():
        label_df = df[df["label"] == label]
        corr_matrix = label_df[["text_length", "word_count", "sentence_count"]].corr()
        result[label] = corr_matrix.round(3)
    return result


def run_correlation():
    logging.info("CORRELATION MATRIX (Overall):")
    corr_matrix = compute_correlation_matrix()
    logging.info(f"\n{corr_matrix}")

    logging.info("\nCORRELATION MATRIX BY LABEL:")
    by_label = compute_correlation_by_label()
    for label, corr_matrix in by_label.items():
        logging.info(f"\n{label}:")
        logging.info(f"\n{corr_matrix}")

    return corr_matrix, by_label


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_correlation()
