# ─────────────────────────────────────────────────────────────────
#  LiteAgent — Makefile
# ─────────────────────────────────────────────────────────────────
.PHONY: help install install-full install-dev run api telegram test \
        test-cov lint clean docker docker-up docker-down

PYTHON ?= python3.11
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

# Default target
help: ## Show this help
	@echo ""
	@echo "  LiteAgent — available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Installation ─────────────────────────────────────────────────

venv: ## Create virtual environment
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP) install --quiet --upgrade pip wheel setuptools
	@echo "Virtual environment ready: $(VENV)"

install: venv ## Install (standard: API + embeddings)
	$(PIP) install --quiet -e ".[api,embeddings]"
	@echo "Installed. Run: source $(VENV)/bin/activate && liteagent"

install-full: venv ## Install all components (API + Telegram + embeddings + dev)
	$(PIP) install --quiet -e ".[all]"
	@echo "Full install complete."

install-dev: venv ## Install with dev tools only
	$(PIP) install --quiet -e ".[dev]"
	@echo "Dev install complete."

install-min: venv ## Install minimal (CLI only)
	$(PIP) install --quiet -e .
	@echo "Minimal install complete."

# ── Running ──────────────────────────────────────────────────────

run: ## Start interactive CLI
	$(PY) -m liteagent

run-one: ## One-shot query (usage: make run-one Q="your question")
	$(PY) -m liteagent -1 "$(Q)"

api: ## Start API server + dashboard on :8080
	$(PY) -m liteagent --channel api

telegram: ## Start Telegram bot
	$(PY) -m liteagent --channel telegram

# ── Testing ──────────────────────────────────────────────────────

test: ## Run all tests
	PYTHONPATH=. $(PY) -m pytest tests/ -v

test-cov: ## Run tests with coverage
	PYTHONPATH=. $(PY) -m pytest tests/ -v --cov=liteagent --cov-report=term-missing

test-fast: ## Run tests without slow markers
	PYTHONPATH=. $(PY) -m pytest tests/ -v -x

# ── Docker ───────────────────────────────────────────────────────

docker: ## Build Docker image
	docker build -t liteagent:latest .

docker-up: ## Start with docker-compose (API mode)
	docker compose up -d api

docker-up-tg: ## Start with docker-compose (Telegram mode)
	docker compose --profile telegram up -d

docker-down: ## Stop docker-compose
	docker compose down

docker-logs: ## Show docker logs
	docker compose logs -f

# ── Maintenance ──────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

reset-db: ## Delete memory database (careful!)
	@echo "This will delete ALL memories. Press Ctrl+C to cancel."
	@sleep 3
	rm -f ~/.liteagent/memory.db
	@echo "Database reset."

check: ## Verify installation
	@$(PY) -c "from liteagent import __version__; print(f'LiteAgent v{__version__} OK')"
	@$(PY) -c "import anthropic; print(f'anthropic SDK OK')"
	@echo "All checks passed."
