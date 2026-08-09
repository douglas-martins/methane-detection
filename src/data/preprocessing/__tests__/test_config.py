"""Tests for the Hydra config-group restructure (configs/data.yaml + configs/dataset/*.yaml).

Proves `dataset=starcop_raw` alone (one override) pulls in every raw-specific
setting consistently -- train_csv/test_csv/num_workers -- instead of relying
on 2-3 separate CLI overrides staying in sync by hand.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "configs")


def test_starcop_mini_config_resolves_expected_values():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="data", overrides=["dataset=starcop_mini"])
    assert cfg.dataset == "starcop_mini"
    assert cfg.dataset_cfg.train_csv == "train_mini10.csv"
    assert cfg.paths.raw_root == "data/starcop_mini"
    assert cfg.patch.num_workers == 1


def test_starcop_raw_config_resolves_expected_values():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="data", overrides=["dataset=starcop_raw"])
    assert cfg.dataset == "starcop_raw"
    assert cfg.dataset_cfg.train_csv == "train.csv"
    assert cfg.paths.raw_root == "data/starcop_raw"
    assert cfg.patch.num_workers == 4  # raw's override


def test_default_dataset_is_starcop_mini_when_unspecified():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="data")
    assert cfg.dataset == "starcop_mini"
