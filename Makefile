ENV_BASELINE_PYTHON := vendor/starcop/.venv/bin/python
ENV_BASELINE_GENBADGE := vendor/starcop/.venv/bin/genbadge
ENV_BASELINE_TEST_PATHS := src/data/download/__tests__ tests/vendor_starcop src/training/__tests__ src/registry/__tests__ src/evaluation/__tests__
ENV_BASELINE_COV_PATHS := --cov=src/data/download --cov=vendor/starcop/scripts/preprocessing --cov=src/training

ENV_RESEARCH_PYTHON := .venv/bin/python
ENV_RESEARCH_GENBADGE := .venv/bin/genbadge
ENV_RESEARCH_INTERROGATE := .venv/bin/interrogate
ENV_RESEARCH_RUFF := .venv/bin/ruff
ENV_RESEARCH_TEST_PATHS := src/data/preprocessing/__tests__ src/training/__tests__ src/registry/__tests__ src/serving/__tests__ src/evaluation/__tests__ flows/__tests__
ENV_RESEARCH_COV_PATHS := --cov=src/data/preprocessing --cov=src/training --cov=src/registry --cov=src/serving --cov=src/evaluation --cov=flows

BATS_IMAGE := bats/bats:latest
SCRIPTS_TEST_PATHS := scripts/__tests__

.PHONY: test-baseline coverage test-research coverage-research badges badges-research test docstring-coverage test-scripts lint docs-serve docs-build

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
