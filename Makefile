.PHONY: help install test clean run lint format setup-dev

help:
	@echo "Available commands:"
	@echo "  install     Install the package and dependencies"
	@echo "  test        Run tests"
	@echo "  clean       Clean up generated files"
	@echo "  run         Run the main analysis"
	@echo "  lint        Run code linting"
	@echo "  format      Format code"
	@echo "  setup-dev   Setup development environment"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf results/*.png
	rm -rf results/*.html

run:
	python main.py

lint:
	flake8 src/ tests/ main.py --max-line-length=100 --ignore=E203,W503

format:
	black src/ tests/ main.py --line-length=100

setup-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy
	pip install -e .
	mkdir -p data results results/plots
