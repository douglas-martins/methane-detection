ENV_BASELINE_PYTHON := vendor/starcop/.venv/bin/python
ENV_BASELINE_GENBADGE := vendor/starcop/.venv/bin/genbadge
ENV_BASELINE_MUTMUT := vendor/starcop/.venv/bin/mutmut
ENV_BASELINE_TEST_PATHS := src/data/download/__tests__ tests/vendor_starcop src/training/__tests__ src/baselines/starcop/training/__tests__ src/registry/__tests__ src/baselines/starcop/registry/__tests__ src/baselines/starcop/evaluation/__tests__
ENV_BASELINE_COV_PATHS := --cov=src/data/download --cov=vendor/starcop/scripts/preprocessing --cov=src/training --cov=src/baselines/starcop/training

ENV_RESEARCH_PYTHON := .venv/bin/python
ENV_RESEARCH_GENBADGE := .venv/bin/genbadge
ENV_RESEARCH_INTERROGATE := .venv/bin/interrogate
ENV_RESEARCH_MUTMUT := .venv/bin/mutmut
ENV_RESEARCH_RUFF := .venv/bin/ruff
ENV_RESEARCH_TEST_PATHS := src/data/preprocessing/__tests__ src/training/__tests__ src/baselines/starcop/training/__tests__ src/registry/__tests__ src/baselines/starcop/registry/__tests__ src/serving/__tests__ src/baselines/starcop/serving/__tests__ src/baselines/starcop/evaluation/__tests__ flows/__tests__ tests/__tests__
ENV_RESEARCH_COV_PATHS := --cov=src/data/preprocessing --cov=src/training --cov=src/baselines/starcop/training --cov=src/registry --cov=src/baselines/starcop/registry --cov=src/serving --cov=src/baselines/starcop/serving --cov=src/baselines/starcop/evaluation --cov=flows

BATS_IMAGE := bats/bats:latest
SCRIPTS_TEST_PATHS := scripts/__tests__

# --- Coursework (DL course final project) -----------------------------
# Isolated targets for coursework/dl-final-project/ (see the untracked
# dl-course-final-project-plan.md, Section 0/15). Deliberately NOT wired
# into `test`, `test-research`, `lint`, or `docstring-coverage` -- the
# coursework grade doesn't need this repo's thesis-scale gates, and this
# folder is deleted (along with this whole block) once the course is
# graded and any reusable pieces have been promoted into src/ per the
# plan's Section 15.
ENV_COURSEWORK_PYTHON := .venv/bin/python
COURSEWORK_PATH := coursework/dl-final-project

.PHONY: coursework-test coursework-lint coursework-train

coursework-test:
	$(ENV_COURSEWORK_PYTHON) -m pytest $(COURSEWORK_PATH) -v

coursework-lint:
	$(ENV_RESEARCH_RUFF) check $(COURSEWORK_PATH)
	$(ENV_RESEARCH_RUFF) format --check $(COURSEWORK_PATH)

coursework-train:
	$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/train.py

.PHONY: test-baseline coverage test-research coverage-research badges badges-research test docstring-coverage test-scripts lint docs-serve docs-build mutation-research mutation-baseline mutation-gate

mutation-research:
	$(ENV_RESEARCH_MUTMUT) run \
		'registry.promotion_criteria.*' \
		'serving.band_statistics.*' \
		'serving.drift.*' \
		'baselines.starcop.training.validation_metrics.*' \
		'baselines.starcop.training.launch_profiles.*' \
		'baselines.starcop.training.accelerator_check.*' \
		'baselines.starcop.training.colab_bootstrap.*' \
		'training.mlflow_log_model_compat.*' \
		'training.dvc_dataset_version.*' \
		'training.mlflow_utils.*' \
		'baselines.starcop.training.settings_overlay.*' \
		'data.preprocessing.split.*' \
		'baselines.starcop.serving.band_baseline.*' \
		'baselines.starcop.evaluation.select_docs_examples.*' \
		'baselines.starcop.registry.hf_baseline_import.*'

mutation-baseline:
	$(ENV_BASELINE_MUTMUT) run 'data.download.download_mini_dataset.*'

mutation-gate:
	@if [ -x "$(ENV_RESEARCH_MUTMUT)" ]; then \
		$(ENV_RESEARCH_MUTMUT) export-cicd-stats; \
	else \
		$(ENV_BASELINE_MUTMUT) export-cicd-stats; \
	fi
	jq -e \
		'.survived == 0 and .suspicious == 0 and .segfault == 0 and .total > 0' \
		mutants/mutmut-cicd-stats.json

test-baseline:
	$(ENV_BASELINE_PYTHON) -m pytest $(ENV_BASELINE_TEST_PATHS) -v

coverage:
	$(ENV_BASELINE_PYTHON) -m pytest $(ENV_BASELINE_TEST_PATHS) \
		$(ENV_BASELINE_COV_PATHS) \
		--cov-report=term-missing --cov-report=xml --junitxml=junit.xml

test-research:
	$(ENV_RESEARCH_PYTHON) -m pytest $(ENV_RESEARCH_TEST_PATHS) -v

coverage-research:
	$(ENV_RESEARCH_PYTHON) -m pytest $(ENV_RESEARCH_TEST_PATHS) \
		$(ENV_RESEARCH_COV_PATHS) \
		--cov-report=term-missing --cov-report=xml:coverage-research.xml --junitxml=junit-research.xml

badges: coverage
	mkdir -p docs/badges
	$(ENV_BASELINE_GENBADGE) tests -i junit.xml -o docs/badges/tests-baseline.svg -n "tests (baseline)"
	$(ENV_BASELINE_GENBADGE) coverage -i coverage.xml -o docs/badges/coverage-baseline.svg -n "coverage (baseline)"

badges-research: coverage-research
	mkdir -p docs/badges
	$(ENV_RESEARCH_GENBADGE) tests -i junit-research.xml -o docs/badges/tests-research.svg -n "tests (research)"
	$(ENV_RESEARCH_GENBADGE) coverage -i coverage-research.xml -o docs/badges/coverage-research.svg -n "coverage (research)"

docstring-coverage:
	$(ENV_RESEARCH_INTERROGATE) -v .

lint:
	$(ENV_RESEARCH_RUFF) check .
	$(ENV_RESEARCH_RUFF) format --check .

test-scripts:
	docker run --rm -v "$$PWD":/code -w /code $(BATS_IMAGE) $(SCRIPTS_TEST_PATHS)

docs-serve:
	$(ENV_RESEARCH_PYTHON) -m mkdocs serve

docs-build:
	$(ENV_RESEARCH_PYTHON) -m mkdocs build --strict

test: test-baseline test-research
