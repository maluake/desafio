.PHONY: install index index-crawl run test lint eval eval-llm docker-build docker-up docker-test demo

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

index:
	python -m app.rag.ingest --mode seed

index-crawl:
	python -m app.rag.ingest --mode crawl

run:
	uvicorn app.main:app --reload --port 8000

test:
	python -m pytest

lint:
	ruff check app tests evals

eval:
	python -m evals.run_eval

eval-llm:
	python -m evals.run_eval --llm --judge --real-web

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-test:
	docker compose run --rm api python -m pytest -p no:cacheprovider

demo:
	bash scripts/demo.sh
