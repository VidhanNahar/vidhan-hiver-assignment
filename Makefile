PYTHON := $(shell [ -f .venv/bin/python3 ] && echo .venv/bin/python3 || echo python3)

.PHONY: setup preprocess explore sample auto-label eval-main eval-trivial eval-simple eval-all eval-nojudge calibrate demo clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and set up environment
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "\n✅ Setup complete. Activate with: source .venv/bin/activate"
	@echo "📝 Copy example.env to .env and add your OpenAI API key:"
	@echo "   cp example.env .env"

preprocess: ## Run data preprocessing (filter AmazonHelp, clean, reconstruct threads)
	$(PYTHON) -m src.data.preprocess

explore: ## Run exploratory data analysis
	$(PYTHON) -m src.data.explore

sample: ## Sample golden evaluation set (200 examples)
	$(PYTHON) -m src.data.sample

auto-label: ## Auto-label golden candidates with LLM (requires OPENAI_API_KEY)
	$(PYTHON) -m src.data.auto_label

eval-main: ## Evaluate the main system against golden set
	$(PYTHON) -m src.eval.run_eval --system main

eval-trivial: ## Evaluate the trivial baseline
	$(PYTHON) -m src.eval.run_eval --system baseline_trivial --skip-judge

eval-simple: ## Evaluate the simple (zero-shot) baseline
	$(PYTHON) -m src.eval.run_eval --system baseline_simple

eval-all: ## Evaluate all systems and print comparison
	$(PYTHON) -m src.eval.run_eval --system all

eval-nojudge: ## Evaluate all systems without LLM judge (saves API cost)
	$(PYTHON) -m src.eval.run_eval --system all --skip-judge

calibrate: ## Run human vs LLM judge agreement calibration
	$(PYTHON) -m src.eval.judge_calibration

demo: ## Run interactive demo
	$(PYTHON) -m src.pipeline.agent

clean: ## Remove generated files (keeps raw data and golden set)
	rm -rf data/chroma/ data/processed/ results/ __pycache__/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
