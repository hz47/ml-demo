import logging
import pandas as pd
from .utils import load_data, count_sentences


def get_avg_sentence_count_by_label():
    df = load_data()
    df["sentence_count"] = df["text"].astype(str).apply(count_sentences)
    return df.groupby("label")["sentence_count"].median() # use median instead of mean()


def run_sms_sentence_count():
    logging.info("MEDIAN SENTENCE COUNT BY LABEL")
    avg_counts = get_avg_sentence_count_by_label()
    for label, avg_count in avg_counts.items():
        logging.info(f"  {label}: {avg_count:.1f} sentences")
    return avg_counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_sms_sentence_count()
