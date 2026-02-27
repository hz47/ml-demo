.PHONY: help clean-data train cluster api test test-cov all

help:
	@echo "Available commands:"
	@echo "  make clean-data - Run only the data preprocessing step"
	@echo "  make train      - Train the ML model"
	@echo "  make cluster    - Run the LLM SMS clustering script"
	@echo "  make api        - Start the FastAPI server"
	@echo "  make test       - Run all tests"
	@echo "  make test-cov  - Run all tests with coverage report"
	@echo "  make all        - Run the entire end-to-step pipeline"

clean-data:
	python -m steps.clean

train:
	python -m steps.train

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
