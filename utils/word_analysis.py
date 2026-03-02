import sys
import logging
import pandas as pd
from collections import Counter
from config import SMS_CLEAN_PATH
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_top_words(texts, n=20):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    return counter.most_common(n)

def run_word_analysis(top_n=20):
    logging.info("STEP 2: WORD ANALYSIS STARTED")

    df = pd.read_csv(SMS_CLEAN_PATH)
    df = df.dropna(subset=["clean_light"])
    df["clean_light"] = df["clean_light"].astype(str).str.strip()

    logging.info(f"Rows loaded: {len(df)}")

    # Top words overall
    top_all = get_top_words(df["clean_light"], top_n)
    top_words_all = [w for w, _ in top_all]

    # Log
    logging.info("Top words: %s", top_words_all)

    logging.info("Word analysis completed.")
    return top_words_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_word_analysis()