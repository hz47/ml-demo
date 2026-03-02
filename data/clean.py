import re
import pandas as pd
import os

RAW_DATA_PATH = "data/raw/SMSSpamCollection.txt"
PROCESSED_DATA_PATH = "data/processed/sms_clean.csv"

def clean_text_v3(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    
    # 1. Standardize common spam shorthand
    text = re.sub(r'\b2\b', ' to ', text)
    text = re.sub(r'\b4\b', ' for ', text)
    text = re.sub(r'\bu\b', ' you ', text)
    
    # 2. Fix currency symbols (including the '?' misencoding found in leaked spam)
    # If a '?' is next to a number, it's almost certainly a currency symbol
    text = re.sub(r'\?\d', ' moneysymb ', text) 
    text = re.sub(r'[$£€]', ' moneysymb ', text)
    
    # 3. Normalize specific spam entities
    text = re.sub(r'http\S+|www\S+', ' urladdr ', text)
    text = re.sub(r'\b\d{10,13}\b', ' longphonenum ', text)
    text = re.sub(r'\b\d{4,6}\b', ' shortcode ', text)
    
    # 4. Keep exclamation marks (Spam uses them 3x more than Ham)
    text = re.sub(r"[^a-zA-Z0-9\s!]", " ", text)
    return " ".join(text.split())

def load_and_process():
    if not os.path.exists(RAW_DATA_PATH):
        print("Raw data file not found.")
        return
    df = pd.read_csv(RAW_DATA_PATH, sep="\t", names=["label", "text"], header=None)
    df["clean_light"] = df["text"].apply(clean_text_v3)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("V3 Preprocessing Complete.")

if __name__ == "__main__":
    load_and_process()