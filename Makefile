# Common tasks. `make help` lists them. Each target does one thing.
.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## install python workspace + pre-commit hooks
	uv sync --all-packages
	uv run pre-commit install

infra-up: ## start postgres, redis, minio, ollama (dev)
	docker compose -f infra/docker-compose.yml up -d

infra-down: ## stop dev infrastructure
	docker compose -f infra/docker-compose.yml down

lint: ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## mypy strict on clinical + inference, basic elsewhere
	uv run mypy packages/clinical/src apps/inference

test: ## unit tests (no docker needed)
	uv run pytest -m "not integration and not golden"

test-all: ## everything, including integration + golden set
	uv run pytest

contracts: ## regenerate OpenAPI specs and the web client
	@echo "TODO(phase 1): api/inference/llm_gateway -> packages/contracts/openapi/*.yaml -> apps/web/src/api"

doc: ## rebuild docs/MediScan_Engineering_Baseline.pdf
	uv run --with reportlab python docs/src/build_baseline_pdf.py

.PHONY: help setup infra-up infra-down lint typecheck test test-all contracts doc
