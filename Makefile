.PHONY: help clean train train_nb predict benchmark cluster api test test-cov all analysis word-analysis word-analysis-stop ngrams distribution sms-length word-count sentence-count correlation report-analysis

help:
	@echo "Available commands:"
	@echo "  make clean          - Run data preprocessing (clean SMS data)"
	@echo "  make word-analysis - Run word frequency analysis (spam vs ham)"
	@echo "  make word-analysis-stop - Run word frequency analysis without stopwords (spam vs ham)"
	@echo "  make ngrams         - Run ngram analysis (bigrams/trigrams)"
	@echo "  make distribution  - Run label distribution analysis"
	@echo "  make sms-length     - Run SMS length analysis by label"
	@echo "  make word-count    - Run word count analysis by label"
	@echo "  make sentence-count - Run sentence count analysis by label"
	@echo "  make correlation  - Run correlation matrix analysis"
	@echo "  make report-analysis - Generate visual analysis report (all charts)"
	@echo "  make train          - Train Logistic Regression model"
	@echo "  make train_nb       - Train Naive Bayes model"
	@echo "  make predict        - Run interactive prediction"
	@echo "  make benchmark      - Compare both models"
	@echo "  make cluster       - Run the LLM SMS clustering script"
	@echo "  make api           - Start the FastAPI server"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run all tests with coverage report"
	@echo "  make all           - Run the entire end-to-step pipeline"

clean-data:
	python -m ml.clean

word-analysis:
	python -m analysis.word_analysis

word-analysis-stop:
	python -m analysis.word_analysis_stop

ngrams:
	python -m analysis.ngrams

distribution:
	python -m analysis.distribution

sms-length:
	python -m analysis.sms_length

word-count:
	python -m analysis.sms_word_count

sentence-count:
	python -m analysis.sms_sentence_count

correlation:
	python -m analysis.correlation

report-analysis:
	python -m analysis.report_analysis

train:
	python -m ml.train

train_nb:
	python -m ml.train_nb

predict:
	python -m ml.predict

benchmark:
	python -m ml.benchmark

cluster:
	python -m llm.clustering

api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=. --cov-report=term-missing -v

all:
	python run_all.py
