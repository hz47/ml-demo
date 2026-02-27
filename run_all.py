import argparse
import logging
import sys

from steps.clean import run_preprocessing
from utils.word_analysis import run_word_analysis
from utils.ngrams import run_ngrams
from steps.train import run_model_training
from steps.train_nb import run_model_training as run_nb_training


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
        ]
    )


def run_clean():
    logging.info("STEP 1: PREPROCESSING")
    run_preprocessing()


def run_analysis():
    logging.info("STEP 2: WORD ANALYSIS")
    top_words = run_word_analysis()

    logging.info("STEP 3: NGRAM ANALYSIS")
    top_bigrams, top_trigrams = run_ngrams()
    top_ngrams = top_bigrams + top_trigrams


def run_train():
    logging.info("STEP 4: MODEL TRAINING (Logistic Regression)")
    run_model_training()


def run_train_nb():
    logging.info("STEP 5: MODEL TRAINING (Naive Bayes)")
    run_nb_training()


def run_benchmark():
    logging.info("STEP 6: MODEL BENCHMARK")
    from steps.benchmark import run_benchmark as run_bm
    run_bm()


def run_all_steps():
    try:
        run_clean()
        run_analysis()
        run_train()
        run_train_nb()
        run_benchmark()
        logging.info("Pipeline completed successfully")
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SMS Spam Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                     Run the full pipeline
  python run_all.py --step clean        Run only data preprocessing
  python run_all.py --step analysis     Run word and ngram analysis
  python run_all.py --step train        Train Logistic Regression model
  python run_all.py --step train_nb     Train Naive Bayes model
  python run_all.py --step benchmark    Compare both models
  python run_all.py --step all          Run full pipeline (default)
        """
    )

    parser.add_argument(
        "--step",
        choices=["clean", "analysis", "train", "train_nb", "benchmark", "all"],
        default="all",
        help="Which pipeline step to run (default: all)"
    )

    args = parser.parse_args()

    setup_logging()
    logging.info(f"Running pipeline step: {args.step}")

    if args.step == "clean":
        run_clean()
    elif args.step == "analysis":
        run_analysis()
    elif args.step == "train":
        run_train()
    elif args.step == "train_nb":
        run_train_nb()
    elif args.step == "benchmark":
        run_benchmark()
    elif args.step == "all":
        run_all_steps()


if __name__ == "__main__":
    main()
