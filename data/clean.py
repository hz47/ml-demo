"""
Preprocessing Module
--------------------
- Loads SMS dataset
- Dual-cleaning: Strict (for meaning) & Light (for patterns)
- Saves cleaned CSV
"""

import os
import logging
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

RAW_DATA_PATH = "data/raw/SMSSpamCollection.txt"
PROCESSED_DATA_PATH = "data/processed/sms_clean.csv"

STRICT_REGEX = re.compile(r"[^a-zA-Z0-9\s]")
LIGHT_REGEX = re.compile(r"[^a-zA-Z0-9\s!?$%]")

def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

def clean_text_strict(text: str, stop_words: set) -> str:
    if not isinstance(text, str): return ""
    
    text = text.lower()
    text = STRICT_REGEX.sub(" ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return " ".join(words)

def clean_text_light(text: str) -> str:
    if not isinstance(text, str): return ""
    
    text = " ".join(text.split())
    text = LIGHT_REGEX.sub(" ", text)
    
    return text.strip()

def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw data file: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        names=["label", "text"],
        header=None,
        dtype={"label": "category", "text": "string"}
    )

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Applying dual-cleaning strategy...")
    stop_words = get_stopwords()
    
    df["clean_strict"] = df["text"].map(lambda x: clean_text_strict(x, stop_words))
    df["clean_light"] = df["text"].map(clean_text_light)
    return df

def run_preprocessing():
    logging.info("STEP 1: PREPROCESSING STARTED")

    try:
        df = load_raw_data()
        df = process_dataframe(df)
        
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        
        logging.info(f"Preprocessing finished. Saved to {PROCESSED_DATA_PATH}")
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_preprocessing()
