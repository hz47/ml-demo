import logging
import pandas as pd
from .utils import load_data


def get_avg_word_count_by_label():
    df = load_data()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    return df.groupby("label")["word_count"].median() # maybe median comunciates better 


def run_sms_word_count():
    logging.info("Median WORD COUNT BY LABEL")
    avg_counts = get_avg_word_count_by_label()
    for label, avg_count in avg_counts.items():
        logging.info(f"  {label}: {avg_count:.1f} words")
    return avg_counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_sms_word_count()
