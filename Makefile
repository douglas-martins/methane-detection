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

.PHONY: coursework-test coursework-lint coursework-train coursework-confirm-raw \
	coursework-precompute-cache coursework-mutation \
	coursework-evaluate-mini coursework-evaluate-cross-tier coursework-evaluate-r2 \
	coursework-evaluate-r3 coursework-evaluate coursework-pr-curves \
	coursework-eval-scenes coursework-eval-scenes-all \
	coursework-throughput coursework-throughput-scenes

coursework-test:
	$(ENV_COURSEWORK_PYTHON) -m pytest $(COURSEWORK_PATH) -v

coursework-lint:
	$(ENV_RESEARCH_RUFF) check $(COURSEWORK_PATH)
	$(ENV_RESEARCH_RUFF) format --check $(COURSEWORK_PATH)

coursework-train:
	$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/train.py dataset=starcop_mini

# Standalone mutmut run for coursework/dl-final-project (its own setup.cfg, gate
# separate from this repo's [tool.mutmut] -- see that file's own comment). ALWAYS
# use this target, never invoke mutmut directly here: train.py's fit() mutates real
# loop-control code and generates many genuinely-infinite-looping mutants, and
# without OMP_NUM_THREADS/MKL_NUM_THREADS capped to 1, mutmut's default 12 parallel
# workers x torch's default 6 threads each = 72 threads fighting over 12 real
# cores -- that oversubscription (not broken mutations) previously made ~100% of
# fit()'s mutants time out and spammed dozens of SIGXCPU crash notifications in
# under half an hour. Capping threads fixed it: a ~97-mutant fit() batch went from
# not finishing in 500s to 0 timeouts, all killed, in 26s. Pass MUTANTS="'train.x_fit*'"
# (quoted, mutmut's own glob syntax) to scope a run; omit it to run everything in
# only_mutate. See setup.cfg's own [mutmut] comment for the full story.
coursework-mutation:
	cd $(COURSEWORK_PATH) && rm -rf mutants .mutmut-cache
	cd $(COURSEWORK_PATH) && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
		$(abspath $(ENV_COURSEWORK_PYTHON)) -m mutmut run $(MUTANTS)

# One-time on-disk patch cache precompute (see patch_cache.py's own docstring for why:
# starcop_raw's real bottleneck is disk I/O, and the RAM-bound LRU cache in dataset.py
# barely helps at that scale). ~27GB total for both datasets, all splits, float16/uint8 --
# rerun any time data/processed/<dataset>/patches changes, it's a disposable, locally
# regenerable artifact (not git/DVC-tracked). fit()/evaluate.py pick it up automatically
# once built, and fall back to live reads with no error if it's stale or missing.
coursework-precompute-cache:
	$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/precompute_patch_cache.py dataset=starcop_mini
	$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/precompute_patch_cache.py dataset=starcop_raw

# R1 contract confirmation (plan Section 0.1): runs the coursework's own
# preprocessing, loader, model and metric code against real starcop_raw
# patches -- shapes, band order, dtypes, post-normalization ranges, NaN/
# nodata handling. Minutes, not hours; no training. This is the one bar
# the coursework branch's gate isolation does NOT relax, so keep it a
# single command. Requires data/processed/starcop_raw/ to exist:
#   dvc repro normalize@starcop_raw split@starcop_raw patch_extract@starcop_raw
coursework-confirm-raw:
	$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/confirm_raw.py dataset=starcop_raw

# --- Section 8 evaluation sweep -----------------------------------------
# Per-pixel Precision/Recall/F1/confusion-matrix/PR-AUC/per-patch-detection
# scoring (evaluate.py) against the checkpoints Section 7 already trained --
# see report.md's "Métricas de avaliação" for the numbers these produced.
# `-cross-tier`/`-r2`/`-r3` require data/processed/starcop_raw/ to exist
# (same prerequisite as coursework-confirm-raw, above) and their checkpoints
# to already be trained (checkpoints/<arch>-{r2,raw-full}.pt).

# mini tier: val+test for each of E1/E2/E3's own checkpoint.
coursework-evaluate-mini:
	for arch in E1 E2 E3; do \
		$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate.py architecture=$$arch dataset=starcop_mini tier=mini; \
	done

# Cross-tier: mini-trained checkpoints scored on starcop_raw's own test split
# -- does a model trained on 392 patches hold up on the real distribution?
coursework-evaluate-cross-tier:
	for arch in E1 E2 E3; do \
		$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate.py architecture=$$arch dataset=starcop_raw tier=raw-full checkpoint_tier=mini splits=test; \
	done

# R2 (raw subsample-trained) checkpoints, scored on starcop_raw's own full test split.
coursework-evaluate-r2:
	for arch in E1 E2 E3; do \
		$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate.py architecture=$$arch dataset=starcop_raw tier=r2 splits=test; \
	done

# R3 (raw-full-trained) checkpoints -- E2/E3 only, E1 optional per Section 7's
# own decision -- scored on starcop_raw's own full test split.
coursework-evaluate-r3:
	for arch in E2 E3; do \
		$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate.py architecture=$$arch dataset=starcop_raw tier=raw-full splits=test; \
	done

# Everything above, in one shot -- reproduces every evaluate.py run Section 8
# executed (11 total). Minutes, not hours, but real GPU inference over up to
# 16,758 patches per run -- expect roughly 15-20 minutes end to end.
coursework-evaluate: coursework-evaluate-mini coursework-evaluate-cross-tier coursework-evaluate-r2 coursework-evaluate-r3

# Paper-protocol (whole-scene) evaluation, one model and mode (see evaluate_scenes.py's
# docstring), e.g.:
#   make coursework-eval-scenes ARGS="architecture=E2 checkpoint_tier=raw-full mode=full_scene"
coursework-eval-scenes:
	$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate_scenes.py $(ARGS)

# The whole matrix of final models, GPU, one run at a time: R3 (E2/E3, raw-full), R2 (E1/E2/E3) and
# the mini-trained models on the raw tier (D7), each in `full_scene` (validation-picked threshold)
# and the `patches` diagnostic; plus the mini models on mini's own 9 test scenes (fixed 0.5 only).
coursework-eval-scenes-all:
	for mode in full_scene patches; do \
		for arch in E2 E3; do \
			$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate_scenes.py architecture=$$arch checkpoint_tier=raw-full eval_tier=raw-full mode=$$mode; \
		done; \
		for arch in E1 E2 E3; do \
			$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate_scenes.py architecture=$$arch checkpoint_tier=r2 eval_tier=raw-full mode=$$mode; \
			$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate_scenes.py architecture=$$arch checkpoint_tier=mini eval_tier=raw-full mode=$$mode; \
		done; \
	done
	for arch in E1 E2 E3; do \
		$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate_scenes.py architecture=$$arch checkpoint_tier=mini eval_tier=mini mode=full_scene thresholds=single; \
	done

# GPU-vs-CPU inference throughput on patches (report.md's "Throughput de inferencia" table): the
# six configurations of that table, each on cuda then cpu, splits=test. Run on an otherwise idle
# machine, one job at a time, or the timings are meaningless.
coursework-throughput:
	for device in cuda cpu; do \
		for arch in E1 E2 E3; do \
			$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate.py architecture=$$arch dataset=starcop_mini tier=mini splits=test device=$$device; \
		done; \
		$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate.py architecture=E1 dataset=starcop_raw tier=raw-full checkpoint_tier=mini splits=test device=$$device; \
		for arch in E2 E3; do \
			$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate.py architecture=$$arch dataset=starcop_raw tier=raw-full splits=test device=$$device; \
		done; \
	done

# The same comparison for whole 512x512 scenes (scenes/s), fixed 0.5 threshold only so the threshold
# sweep does not dominate the CPU time: E1 (mini-trained) and E2/E3 (R3) on the 342 raw test scenes.
coursework-throughput-scenes:
	for device in cuda cpu; do \
		$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate_scenes.py architecture=E1 checkpoint_tier=mini eval_tier=raw-full mode=full_scene thresholds=single device=$$device; \
		for arch in E2 E3; do \
			$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/evaluate_scenes.py architecture=$$arch checkpoint_tier=raw-full eval_tier=raw-full mode=full_scene thresholds=single device=$$device; \
		done; \
	done

# PR-curve comparison plots (Section 8's final Activities item) -- reads the
# MLflow artifacts the targets above log, so run coursework-evaluate first
# (or at least -mini, -cross-tier, and -r2/-r3 for the E2 configurations it
# plots) if those runs aren't already in mlflow.db.
coursework-pr-curves:
	$(ENV_COURSEWORK_PYTHON) $(COURSEWORK_PATH)/pr_curve_plots.py

.PHONY: test-baseline coverage test-research coverage-research badges badges-research test docstring-coverage test-scripts lint docs-serve docs-build mutation-research mutation-baseline mutation-gate

mutation-research:
	@rm -rf mutants
	$(ENV_RESEARCH_MUTMUT) run \
		'registry.promotion_criteria.*' \
		'registry.mlflow_registry.*' \
		'registry.promote_model.*' \
		'serving.band_statistics.*' \
		'serving.drift.*' \
		'baselines.starcop.training.validation_metrics.*' \
		'baselines.starcop.training.metrics_ext.*' \
		'baselines.starcop.training.normalizer_dtype_fix.*' \
		'baselines.starcop.training.lightning2_compat.*' \
		'baselines.starcop.training.optimizer_compat.*' \
		'baselines.starcop.training.starcop_datamodule.*' \
		'baselines.starcop.training.plot_confusion_matrix.*' \
		'baselines.starcop.training.mlflow_image_logger.*' \
		'baselines.starcop.training.launch_profiles.*' \
		'baselines.starcop.training.accelerator_check.*' \
		'baselines.starcop.training.colab_bootstrap.*' \
		'training.mlflow_log_model_compat.*' \
		'training.dvc_dataset_version.*' \
		'training.mlflow_utils.*' \
		'baselines.starcop.training.settings_overlay.*' \
		'data.preprocessing.split.*' \
		'data.preprocessing.coordinates.*' \
		'data.preprocessing.normalize.*' \
		'data.preprocessing.patch_extract.*' \
		'data.preprocessing.stats.*' \
		'baselines.starcop.serving.band_baseline.*' \
		'baselines.starcop.serving.inference.*' \
		'baselines.starcop.serving.model_loader.*' \
		'baselines.starcop.evaluation.dataset_wiring.*' \
		'baselines.starcop.evaluation.live_verify.*' \
		'baselines.starcop.evaluation.paper_eval_mlflow.*' \
		'baselines.starcop.evaluation.paper_metrics.*' \
		'baselines.starcop.evaluation.run_baseline_eval.*' \
		'flows.retrain.*' \
		'flows.eval_baseline.*' \
		'baselines.starcop.evaluation.select_docs_examples.*' \
		'baselines.starcop.registry.hf_baseline_import.*'
	# mutmut's association cache misses flow trampolines when the full union
	# suite is collected; rebuild stats against the flow tests before gating.
	@rm -f mutants/mutmut-stats.json
	MUTMUT_TEST_PATHS='flows/__tests__' \
		$(ENV_RESEARCH_MUTMUT) run \
		'flows.retrain.*' \
		'flows.eval_baseline.*'
	# The baseline-only downloader is collected by the shared config but run only
	# by mutation-baseline; remove its untested metadata before the research gate.
	@rm -rf mutants/src/data/download

mutation-baseline:
	@set -e; \
	backup=$$(mktemp); \
	cp pyproject.toml "$$backup"; \
	trap 'cp "$$backup" pyproject.toml; rm -f "$$backup"' EXIT INT TERM; \
	$(ENV_BASELINE_PYTHON) -c 'from pathlib import Path; p=Path("pyproject.toml"); s=p.read_text(); start=s.index("only_mutate = ["); end=s.index("]\npytest_add_cli_args =", start)+1; p.write_text(s[:start] + "only_mutate = [\n    \"src/data/download/download_mini_dataset.py\",\n]" + s[end:])'; \
	rm -rf mutants; \
	MUTMUT_TEST_PATHS='$(ENV_BASELINE_TEST_PATHS)' \
		$(ENV_BASELINE_MUTMUT) run 'data.download.download_mini_dataset.*'; \
	cp "$$backup" pyproject.toml; \
	rm -f "$$backup"; \
	trap - EXIT INT TERM

mutation-gate:
	@if [ -x "$(ENV_RESEARCH_MUTMUT)" ]; then \
		mutmut_bin="$(ENV_RESEARCH_MUTMUT)"; \
	else \
		mutmut_bin="$(ENV_BASELINE_MUTMUT)"; \
	fi; \
	"$$mutmut_bin" export-cicd-stats; \
	if "$$mutmut_bin" results | grep -q ': not checked$$'; then \
		echo 'Mutation gate found unchecked mutants'; \
		exit 1; \
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
