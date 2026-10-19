# VoxDesk — developer entry points
#   make help     list targets
PYTHON_VERSION ?= 3.12
HOST           ?= 127.0.0.1
PORT           ?= 8000
TENANT         ?= acme-telecom
WHISPER        ?= base.en

.DEFAULT_GOAL := help
.PHONY: help setup run dev test smoke call fake-llm stop reset-db clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python 3.12 + dependencies (first run)
	uv python install $(PYTHON_VERSION)
	uv sync

run:  ## Start the server (TENANT=acme-telecom PORT=8000)
	AGENT_TENANT=$(TENANT) WHISPER_MODEL=$(WHISPER) uv run uvicorn src.main:app --host $(HOST) --port $(PORT)

dev:  ## Start the server with auto-reload
	AGENT_TENANT=$(TENANT) WHISPER_MODEL=$(WHISPER) uv run uvicorn src.main:app --host $(HOST) --port $(PORT) --reload

test:  ## Run the unit test suite
	uv run pytest -q

smoke:  ## Transport smoke test against a running server
	uv run python tools/smoke_ws.py

call:  ## Simulated call, e.g. make call TEXT="how much is Smart Plus?"
	@test -n "$(TEXT)" || { echo 'usage: make call TEXT="your question"'; exit 1; }
	uv run python tools/simulate_call.py "$(TEXT)"

fake-llm:  ## Offline dev LLM on :8123 (then run with OPENROUTER_BASE_URL=http://127.0.0.1:8123/api/v1)
	uv run python tools/fake_openrouter.py

stop:  ## Stop a running server
	-pkill -f "[u]vicorn src.main:app"

reset-db:  ## Delete call history (data/calls.db)
	rm -f data/calls.db

clean:  ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
