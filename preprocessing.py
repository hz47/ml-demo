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

# ---------------------------------------
# STOPWORDS SETUP
# ---------------------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ---------------------------------------
# REGEX PRECOMPILATION
# ---------------------------------------
# Strict: Letters and numbers only
STRICT_REGEX = re.compile(r"[^a-zA-Z0-9\s]")
# Light: Keep important symbols used in spam
LIGHT_REGEX = re.compile(r"[^a-zA-Z0-9\s!?$%]")

# ---------------------------------------
# TEXT CLEANING FUNCTIONS
# ---------------------------------------

def clean_text_strict(text: str) -> str:
    """
    Standard cleaning: lowercase, no symbols, no stopwords.
    Best for finding the 'topic' of the message.
    """
    if not isinstance(text, str): return ""
    
    text = text.lower()
    text = STRICT_REGEX.sub(" ", text)
    words = text.split()
    # Remove stopwords and single characters
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return " ".join(words)

def clean_text_light(text: str) -> str:
    """
    Preserves signals: Keeps case (URGENT), numbers, and symbols ($ ! ? %).
    Best for character-level patterns and catching 'spammy' formatting.
    """
    if not isinstance(text, str): return ""
    
    # Normalize whitespace
    text = " ".join(text.split())
    # Keep alpha-numeric plus specific spam symbols
    text = LIGHT_REGEX.sub(" ", text)
    
    return text.strip()

# ---------------------------------------
# DATASET LOADING FUNCTION
# ---------------------------------------

def load_dataset(path="data/SMSSpamCollection.txt"):
    """Loads dataset and applies both cleaning methods."""
    logging.info(f"Reading dataset from {path}...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw data file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        names=["label", "text"],
        header=None,
        dtype={"label": "category", "text": "string"}
    )

    logging.info("Applying dual-cleaning strategy...")
    df["clean_strict"] = df["text"].map(clean_text_strict)
    df["clean_light"] = df["text"].map(clean_text_light)

    return df

# ---------------------------------------
# MAIN PREPROCESSING FUNCTION
# ---------------------------------------

def run_preprocessing():
    """Main entry point for step 1."""
    logging.info("STEP 1: PREPROCESSING STARTED")

    try:
        df = load_dataset()
        
        os.makedirs("data", exist_ok=True)
        output_path = "data/sms_clean.csv"
        
        df.to_csv(output_path, index=False)
        
        logging.info(f"Preprocessing finished. Saved to {output_path}")
        logging.info(f"Sample Light: {df['clean_light'].iloc[0]}")
        logging.info(f"Sample Strict: {df['clean_strict'].iloc[0]}")
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_preprocessing()