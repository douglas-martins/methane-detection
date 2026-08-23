ENV_A_PYTHON := vendor/starcop/.venv/bin/python
ENV_A_GENBADGE := vendor/starcop/.venv/bin/genbadge
ENV_A_TEST_PATHS := src/data/download/__tests__ tests/vendor_starcop src/training/__tests__ src/registry/__tests__ src/evaluation/__tests__
ENV_A_COV_PATHS := --cov=src/data/download --cov=vendor/starcop/scripts/preprocessing --cov=src/training

ENV_B_PYTHON := .venv/bin/python
ENV_B_GENBADGE := .venv/bin/genbadge
ENV_B_INTERROGATE := .venv/bin/interrogate
ENV_B_RUFF := .venv/bin/ruff
ENV_B_TEST_PATHS := src/data/preprocessing/__tests__ src/training/__tests__ src/registry/__tests__ src/serving/__tests__ src/evaluation/__tests__ flows/__tests__
ENV_B_COV_PATHS := --cov=src/data/preprocessing --cov=src/training --cov=src/registry --cov=src/serving --cov=src/evaluation --cov=flows

BATS_IMAGE := bats/bats:latest
SCRIPTS_TEST_PATHS := scripts/__tests__

.PHONY: test-env-a coverage test-env-b coverage-env-b badges badges-env-b test docstring-coverage test-scripts lint docs-serve docs-build

test-env-a:
	$(ENV_A_PYTHON) -m pytest $(ENV_A_TEST_PATHS) -v

coverage:
	$(ENV_A_PYTHON) -m pytest $(ENV_A_TEST_PATHS) \
		$(ENV_A_COV_PATHS) \
		--cov-report=term-missing --cov-report=xml --junitxml=junit.xml

test-env-b:
	$(ENV_B_PYTHON) -m pytest $(ENV_B_TEST_PATHS) -v

coverage-env-b:
	$(ENV_B_PYTHON) -m pytest $(ENV_B_TEST_PATHS) \
		$(ENV_B_COV_PATHS) \
		--cov-report=term-missing --cov-report=xml:coverage-env-b.xml --junitxml=junit-env-b.xml

badges: coverage
	mkdir -p docs/badges
	$(ENV_A_GENBADGE) tests -i junit.xml -o docs/badges/tests-env-a.svg -n "tests (env A)"
	$(ENV_A_GENBADGE) coverage -i coverage.xml -o docs/badges/coverage-env-a.svg -n "coverage (env A)"

badges-env-b: coverage-env-b
	mkdir -p docs/badges
	$(ENV_B_GENBADGE) tests -i junit-env-b.xml -o docs/badges/tests-env-b.svg -n "tests (env B)"
	$(ENV_B_GENBADGE) coverage -i coverage-env-b.xml -o docs/badges/coverage-env-b.svg -n "coverage (env B)"

docstring-coverage:
	$(ENV_B_INTERROGATE) -v .

lint:
	$(ENV_B_RUFF) check .
	$(ENV_B_RUFF) format --check .

test-scripts:
	docker run --rm -v "$$PWD":/code -w /code $(BATS_IMAGE) $(SCRIPTS_TEST_PATHS)

docs-serve:
	$(ENV_B_PYTHON) -m mkdocs serve

docs-build:
	$(ENV_B_PYTHON) -m mkdocs build --strict

test: test-env-a test-env-b
