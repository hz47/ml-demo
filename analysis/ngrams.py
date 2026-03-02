import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from .utils import load_data


def top_ngrams(texts, ngram_range, top_n, min_df=1):
    if not texts:
        return []
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english", min_df=min_df)
    X = vectorizer.fit_transform(texts)
    counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    df = pd.DataFrame({"ngram": vocab, "count": counts})
    df = df.sort_values("count", ascending=False)
    return df["ngram"].head(top_n).tolist()


def run_ngrams(top_n=20):
    logging.info("NGRAM ANALYSIS")

    df = load_data()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()

    texts_all = df["text"].tolist()
    texts_ham = df[df["label"] == "ham"]["text"].tolist()
    texts_spam = df[df["label"] == "spam"]["text"].tolist()

    total_bigrams = top_ngrams(texts_all, (2, 2), top_n)
    total_trigrams = top_ngrams(texts_all, (3, 3), top_n)

    ham_bigrams = top_ngrams(texts_ham, (2, 2), top_n)
    ham_trigrams = top_ngrams(texts_ham, (3, 3), top_n)

    spam_bigrams = top_ngrams(texts_spam, (2, 2), top_n)
    spam_trigrams = top_ngrams(texts_spam, (3, 3), top_n)

    logging.info("Top ngrams (total - bigrams): %s", total_bigrams)
    logging.info("Top ngrams (total - trigrams): %s", total_trigrams)
    logging.info("Top ngrams (ham - bigrams): %s", ham_bigrams)
    logging.info("Top ngrams (ham - trigrams): %s", ham_trigrams)
    logging.info("Top ngrams (spam - bigrams): %s", spam_bigrams)
    logging.info("Top ngrams (spam - trigrams): %s", spam_trigrams)

    logging.info("Ngram analysis completed.")

    return {
        "total_bigrams": total_bigrams,
        "total_trigrams": total_trigrams,
        "total_ngrams": total_bigrams + total_trigrams,
        "ham_bigrams": ham_bigrams,
        "ham_trigrams": ham_trigrams,
        "spam_bigrams": spam_bigrams,
        "spam_trigrams": spam_trigrams,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ngrams()
