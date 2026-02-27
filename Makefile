.PHONY: help clean train train_nb predict benchmark cluster api test test-cov all

help:
	@echo "Available commands:"
	@echo "  make clean       - Run data preprocessing (clean SMS data)"
	@echo "  make train       - Train Logistic Regression model"
	@echo "  make train_nb    - Train Naive Bayes model"
	@echo "  make predict     - Run interactive prediction"
	@echo "  make benchmark   - Compare both models"
	@echo "  make cluster    - Run the LLM SMS clustering script"
	@echo "  make api         - Start the FastAPI server"
	@echo "  make test        - Run all tests"
	@echo "  make test-cov    - Run all tests with coverage report"
	@echo "  make all         - Run the entire end-to-step pipeline"

clean-data:
	python -m steps.clean

train:
	python -m steps.train

train_nb:
	python -m steps.train_nb

predict:
	python -m steps.predict

benchmark:
	python -m steps.benchmark

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
