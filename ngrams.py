import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def top_ngrams(texts, ngram_range, top_n):
    vectorizer = CountVectorizer(ngram_range=ngram_range,
                                 stop_words="english",
                                 min_df=2)
    X = vectorizer.fit_transform(texts)
    counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()

    df = pd.DataFrame({"ngram": vocab, "count": counts})
    df = df.sort_values("count", ascending=False)
    return df["ngram"].head(top_n).tolist()

def run_ngrams(top_n=20):
    logging.info("STEP 3: NGRAM ANALYSIS STARTED")

    df = pd.read_csv("data/sms_clean.csv")
    df["clean_strict"] = df["clean_strict"].astype(str).str.strip()

    texts = df["clean_strict"].tolist()

    top_bigrams = top_ngrams(texts, (2,2), top_n)
    top_trigrams = top_ngrams(texts, (3,3), top_n)

    logging.info("Top bigrams: %s", top_bigrams)
    logging.info("Top trigrams: %s", top_trigrams)

    return top_bigrams, top_trigrams