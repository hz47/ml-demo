import logging
import pandas as pd
from .utils import load_data

def get_length_stats_by_label():
    df = load_data()

    df["text_length"] = df["text"].astype(str).str.len()
    
    grouped = df.groupby("label")["text_length"]
    stats = pd.DataFrame({
        "mean": grouped.mean(),
        "median": grouped.median()
    })
    return stats

def run_sms_length():
    logging.info("SMS LENGTH BY LABEL (Mean & Median)")
    stats = get_length_stats_by_label()
    
    for label, row in stats.iterrows():
        logging.info(f"  {label}: mean = {row['mean']:.1f} chars, median = {row['median']:.1f} chars")
    
    return stats

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_sms_length()