import logging
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from .utils import load_data


def get_top_words(texts, n=20):
    stop_words = set(stopwords.words('english'))
    counter = Counter()
    for text in texts:
        words = [w for w in text.lower().split() if w not in stop_words]
        counter.update(words)
    return counter.most_common(n)


def run_word_analysis(top_n=20):
    logging.info("WORD ANALYSIS")

    df = load_data()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()

    logging.info(f"Rows loaded: {len(df)}")

    top_all = get_top_words(df["text"], top_n)
    top_words_all = [w for w, _ in top_all]

    spam_df = df[df["label"] == "spam"]
    ham_df = df[df["label"] == "ham"]

    top_spam = get_top_words(spam_df["text"], top_n)
    top_ham = get_top_words(ham_df["text"], top_n)

    top_words_spam = [w for w, _ in top_spam]
    top_words_ham = [w for w, _ in top_ham]

    logging.info("Top words (all): %s", top_words_all)
    logging.info("Top words (spam): %s", top_words_spam)
    logging.info("Top words (ham): %s", top_words_ham)

    logging.info("Word analysis completed.")
    return {
        "all": top_words_all,
        "spam": top_words_spam,
        "ham": top_words_ham
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_word_analysis()
