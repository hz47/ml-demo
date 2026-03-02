import logging
import pandas as pd
from .utils import load_data


def get_label_distribution():
    df = load_data()
    return df["label"].value_counts()


def run_distribution():
    logging.info("LABEL DISTRIBUTION")
    dist = get_label_distribution()
    for label, count in dist.items():
        pct = count / dist.sum() * 100
        logging.info(f"  {label}: {count} ({pct:.1f}%)")
    return dist


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_distribution()
