import pandas as pd
from config import SMS_CLEAN_PATH
import re
import nltk
from nltk.tokenize import sent_tokenize


def load_data():
    return pd.read_csv(SMS_CLEAN_PATH)


def count_sentences(text):
    if not text or not text.strip():
        return 0
    
    # Split by sentence that end 
    sentences = re.split(r'[.!?]+', text.strip())
    
    # r empty strings
    sentences = [s for s in sentences if s.strip()]
    
    return len(sentences)

def count_sentences_nltk(text):
    if not text or not text.strip():
        return 0
    
    sentences = sent_tokenize(text)
    return len(sentences)