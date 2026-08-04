ENV_A_PYTHON := vendor/starcop/.venv/bin/python
ENV_A_GENBADGE := vendor/starcop/.venv/bin/genbadge
ENV_A_TEST_PATHS := src/data/__tests__ tests/vendor_starcop
ENV_A_COV_PATHS := --cov=src/data --cov=vendor/starcop/scripts/preprocessing

.PHONY: test-env-a coverage badges test

test-env-a:
	$(ENV_A_PYTHON) -m pytest $(ENV_A_TEST_PATHS) -v

coverage:
	$(ENV_A_PYTHON) -m pytest $(ENV_A_TEST_PATHS) \
		$(ENV_A_COV_PATHS) \
		--cov-report=term-missing --cov-report=xml --junitxml=junit.xml

badges: coverage
	mkdir -p docs/badges
	$(ENV_A_GENBADGE) tests -i junit.xml -o docs/badges/tests.svg
	$(ENV_A_GENBADGE) coverage -i coverage.xml -o docs/badges/coverage.svg

test: test-env-a
