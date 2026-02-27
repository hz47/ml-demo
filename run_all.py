import logging
import sys

from preprocessing import run_preprocessing
from word_analysis import run_word_analysis
from ngrams import run_ngrams
from model_training import run_model_training

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
        ]
    )

def main():
    logging.info("Pipeline started")

    try:
        logging.info("STEP 1: PREPROCESSING")
        run_preprocessing()

        logging.info("STEP 2: WORD ANALYSIS")
        top_words = run_word_analysis()

        logging.info("STEP 3: NGRAM ANALYSIS")
        top_bigrams, top_trigrams = run_ngrams()
        top_ngrams = top_bigrams + top_trigrams

        logging.info("STEP 4: MODEL TRAINING")
        run_model_training()

        logging.info("Pipeline completed successfully")

    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()