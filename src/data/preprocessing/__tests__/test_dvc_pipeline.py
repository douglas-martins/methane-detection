"""Tests for dvc.yaml's structure and stage wiring (stage *correctness* is
already covered per-file: test_normalize.py, test_split.py,
test_patch_extract.py, test_stats.py, test_coordinates.py).

Deliberately does not invoke `dvc repro` (Test Size: Large -- needs
.dvc/config + git state + real data, and is slow). Instead this loads
dvc.yaml as plain YAML and asserts the parts a typo could silently break:
every stage iterates the same `${datasets}` var, every stage's static
(non-data) deps exist on disk, every stage passes `dataset=${item}` through
to Hydra, and each stage's declared `outs` is actually listed as a `deps` of
its documented downstream consumer.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DVC_YAML_PATH = REPO_ROOT / "dvc.yaml"

# (upstream stage, downstream stage) -- the pipeline's real data-flow edges,
# per each stage module's own docstring (normalize -> split -> patch_extract
# -> stats, and normalize -> coordinates).
EXPECTED_EDGES = [
    ("normalize", "split"),
    ("split", "patch_extract"),
    ("patch_extract", "stats"),
    ("normalize", "coordinates"),
]


def _load_pipeline() -> dict:
    return yaml.safe_load(DVC_YAML_PATH.read_text())


def _datasets(pipeline: dict) -> list:
    for entry in pipeline["vars"]:
        if isinstance(entry, dict) and "datasets" in entry:
            return entry["datasets"]
    raise AssertionError("dvc.yaml declares no `datasets` var")


def test_every_declared_dataset_has_a_matching_config_file():
    pipeline = _load_pipeline()

    for dataset in _datasets(pipeline):
        assert (REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml").exists()


def test_every_stage_iterates_over_the_shared_datasets_var():
    """A stage hardcoding its own dataset list instead of `${datasets}` would
    silently drift from configs/dataset/*.yaml as datasets are added/removed."""
    pipeline = _load_pipeline()

    for stage_name, stage_def in pipeline["stages"].items():
        assert stage_def["foreach"] == "${datasets}", stage_name


def test_every_stage_cmd_passes_the_dataset_override_through():
    """Every stage's cmd must forward `dataset=${item}` -- the Hydra override
    that selects configs/dataset/<item>.yaml (see test_config.py)."""
    pipeline = _load_pipeline()

    for stage_name, stage_def in pipeline["stages"].items():
        assert "dataset=${item}" in stage_def["do"]["cmd"], stage_name


def test_every_stage_static_dependency_exists_on_disk():
    """Catches a typo'd/renamed script, vendor file, or config path in `deps` --
    skips `data/...` deps since those are raw/processed data, not present
    until `dvc pull`/`dvc repro` actually run."""
    pipeline = _load_pipeline()

    for stage_name, stage_def in pipeline["stages"].items():
        for item in _datasets(pipeline):
            for dep in stage_def["do"]["deps"]:
                resolved = dep.replace("${item}", item)
                if resolved.startswith("data/"):
                    continue
                assert (REPO_ROOT / resolved).exists(), f"{stage_name}: missing dep {resolved!r}"


def test_downstream_stage_depends_on_its_upstream_stage_outs():
    """Wires the pipeline end to end without running it: each edge in
    EXPECTED_EDGES must have the upstream stage's `outs` literally present in
    the downstream stage's `deps`, so a `dvc repro` on the downstream stage
    actually reruns when the upstream output changes."""
    pipeline = _load_pipeline()
    stages = pipeline["stages"]

    for upstream_name, downstream_name in EXPECTED_EDGES:
        upstream_outs = stages[upstream_name]["do"]["outs"]
        downstream_deps = stages[downstream_name]["do"]["deps"]
        for out in upstream_outs:
            assert out in downstream_deps, (
                f"{downstream_name} does not list {upstream_name}'s output {out!r} as a dep"
            )
