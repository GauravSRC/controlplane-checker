# ControlPlane Checker — developer & judge entry points.
#
# Make is OPTIONAL - these targets are convenience aliases only, and `make` is not
# installed on many Windows machines. The documented path is the plain python command
# shown under each target, e.g.:  python scripts/demo.py --offline
#
# Requires Python 3.11-3.12 and a populated .env (copy from .env.example).

PYTHON ?= python

.PHONY: help setup up console dataset eval eval-live tune demo test clean

help:  ## Show available targets
	@echo "setup    - create venv deps + download spaCy model"
	@echo "up       - run the FastAPI governance proxy (OpenAI-compatible)"
	@echo "console  - launch the Streamlit operator console"
	@echo "dataset  - generate the synthetic labeled eval set"
	@echo "eval     - run the eval harness OFFLINE (instant, no API calls)"
	@echo "eval-live- run the eval harness against live guard models"
	@echo "tune     - threshold optimizer + PR curves + pack writeback"
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

dataset:  ## Generate the 300-case synthetic labeled eval set
	$(PYTHON) -m controlplane.eval.generate_dataset

eval:  ## Run the eval harness offline (instant, zero API calls)
	$(PYTHON) -m controlplane.eval.harness --offline

eval-live:  ## Run the eval harness against live guard models (populates the cache)
	$(PYTHON) -m controlplane.eval.harness

tune:  ## Tune thresholds, render plots, write the operating point into the pack
	$(PYTHON) -m controlplane.eval.optimizer --write-pack

demo:  ## Single-command demo for judges
	$(PYTHON) scripts/demo.py --offline

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

clean:  ## Remove caches and local artifacts
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"
