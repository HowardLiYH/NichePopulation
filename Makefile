# Makefile for Emergent Specialization experiments
# NeurIPS 2025 Reproducibility

.PHONY: all install test quick full figures tables clean help

# Python interpreter
PYTHON ?= python3

# Default number of trials
TRIALS ?= 100

# Random seed for reproducibility
SEED ?= 42

help:
	@echo "Emergent Specialization in Multi-Agent Trading"
	@echo "=============================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install dependencies"
	@echo "  test      Run unit tests"
	@echo "  quick     Quick experiment run (10 trials)"
	@echo "  full      Full experiment suite (100 trials)"
	@echo "  exp1      Run Experiment 1 only"
	@echo "  exp2      Run Experiment 2 only"
	@echo "  exp3      Run Experiment 3 only"
	@echo "  exp4      Run Experiment 4 only"
	@echo "  exp5      Run Experiment 5 only"
	@echo "  exp6      Run Experiment 6 only"
	@echo "  figures   Generate paper figures"
	@echo "  tables    Generate paper tables"
	@echo "  clean     Remove generated files"
	@echo ""
	@echo "Variables:"
	@echo "  TRIALS    Number of trials (default: 100)"
	@echo "  SEED      Random seed (default: 42)"
	@echo ""
	@echo "Examples:"
	@echo "  make quick           # Quick test with 10 trials"
	@echo "  make full TRIALS=50  # Full run with 50 trials"
	@echo "  make exp1 TRIALS=20  # Run exp1 with 20 trials"

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

quick:
	$(PYTHON) -m experiments.runner -e exp1 exp2 --trials 10 --seed $(SEED)

full:
	$(PYTHON) -m experiments.runner --all --trials $(TRIALS) --seed $(SEED)

exp1:
	$(PYTHON) experiments/exp1_emergence.py --trials $(TRIALS) --seed $(SEED)

exp2:
	$(PYTHON) experiments/exp2_diversity_value.py --trials $(TRIALS) --seed $(SEED)

exp3:
	$(PYTHON) experiments/exp3_population_size.py --trials 50 --seed $(SEED)

exp4:
	$(PYTHON) experiments/exp4_transfer_frequency.py --trials 50 --seed $(SEED)

exp5:
	$(PYTHON) experiments/exp5_regime_transitions.py --trials 50 --seed $(SEED)

exp6:
	$(PYTHON) experiments/exp6_real_data.py --seed $(SEED)

figures:
	$(PYTHON) -c "from src.analysis.figures import generate_all_figures; generate_all_figures()"

tables:
	$(PYTHON) -c "from src.analysis.cross_experiment import run_cross_analysis; run_cross_analysis()"

clean:
	rm -rf results/
	rm -rf paper/figures/*.pdf
	rm -rf paper/tables/*.tex
	rm -rf paper/tables/*.md
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
