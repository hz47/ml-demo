import logging
import pandas as pd
from collections import Counter
from .utils import load_data


def get_top_words(texts, n=20):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    return counter.most_common(n)


def run_word_stop_analysis(top_n=20):
    logging.info("WORD (without stopwords) ANALYSIS")

    df = load_data()
    df = df.dropna(subset=["clean_light"])
    df["clean_light"] = df["clean_light"].astype(str).str.strip()

    logging.info(f"Rows loaded: {len(df)}")

    top_all = get_top_words(df["clean_light"], top_n)
    top_words_all = [w for w, _ in top_all]

    spam_df = df[df["label"] == "spam"]
    ham_df = df[df["label"] == "ham"]

    top_spam = get_top_words(spam_df["clean_light"], top_n)
    top_ham = get_top_words(ham_df["clean_light"], top_n)

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
    run_word_stop_analysis()
