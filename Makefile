# ControlPlane Checker — developer & judge entry points.
#
# On Windows without `make`, run the equivalent commands shown for each target,
# or just use:  python scripts/demo.py
#
# Requires Python 3.11 and a populated .env (copy from .env.example).

PYTHON ?= python

.PHONY: help setup up console eval demo test clean

help:  ## Show available targets
	@echo "setup    - create venv deps + download spaCy model"
	@echo "up       - run the FastAPI governance proxy (OpenAI-compatible)"
	@echo "console  - launch the Streamlit operator console"
	@echo "eval     - run the eval harness (PR curves, thresholds, latency)"
	@echo "demo     - single-command end-to-end demo for judges"
	@echo "test     - run pytest"
	@echo "clean    - remove caches and local db/data artifacts"

setup:  ## Install dependencies and the spaCy model
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

up:  ## Start the governance proxy
	$(PYTHON) -m uvicorn controlplane.proxy.app:app --host 127.0.0.1 --port 8000 --reload

console:  ## Start the Streamlit operator console
	$(PYTHON) -m streamlit run console/app.py

eval:  ## Run the offline eval harness
	$(PYTHON) -m controlplane.eval.harness

demo:  ## Single-command demo for judges
	$(PYTHON) scripts/demo.py

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

clean:  ## Remove caches and local artifacts
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"
