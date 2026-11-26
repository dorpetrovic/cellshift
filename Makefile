.PHONY: install test lint smoke module1 module2 module3 module4 reproduce

install:      ## dev install with every extra
	pip install -e ".[data,models,track,demo,dev]"
test:
	pytest -q
lint:
	ruff check src tests scripts app

CFG = --config configs/default.yaml

smoke:        ## module 1, first task: K562 end to end
	python scripts/week1_k562.py

module1:      ## data and features: y (pseudobulk) and X (ESM-2, random)
	python scripts/01_build_pseudobulk.py $(CFG)
	python scripts/02_embed_genes.py $(CFG)
module2:      ## models and in-distribution evaluation
	python scripts/03_run_baselines.py $(CFG)
	python scripts/04_train_models.py $(CFG)
module3:      ## shift and trust: Q1 accuracy drop, Q2 coverage, Q3 warning score, Q4 fixes
	python scripts/05_accuracy_drop.py $(CFG)
	python scripts/06_coverage.py $(CFG)
	python scripts/07_warning_score.py $(CFG)
	python scripts/08_cheap_fixes.py $(CFG)
module4:      ## report, figures (demo: python app/app.py)
	python scripts/09_make_report.py $(CFG)

reproduce: module1 module2 module3 module4
