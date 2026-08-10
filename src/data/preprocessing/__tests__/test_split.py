"""Tests for src/data/preprocessing/split.py (stage 2: scene-stratified split).

STARCOP's own train.csv/test.csv already label rows `subset=train|test`
(handled by the caller passing separate train/test csvs in), but provide no
val split. This stage carves val out of train by *scene* (`name` column) so
a flightline's overlapping windows never land on both sides -- avoiding the
leakage TASK-1.2 calls out.
"""

from types import SimpleNamespace

import pandas as pd

import split


def _rows_for_scene(name: str, n_rows: int) -> list[dict]:
    """Build n_rows synthetic window rows for one scene, all sharing the same `name`."""
    return [{"id": f"{name}_w{i}", "name": name, "folder": f"/orig/{name}_w{i}"} for i in range(n_rows)]


def _train_dataframe(scene_names: list[str], rows_per_scene: int = 2) -> pd.DataFrame:
    """Build a synthetic train dataframe spanning multiple scenes, each with rows_per_scene rows."""
    rows = []
    for name in scene_names:
        rows.extend(_rows_for_scene(name, rows_per_scene))
    return pd.DataFrame(rows)


def test_no_scene_appears_in_both_train_and_val():
    """A scene's rows all go to one side of the split -- never split across train and val."""
    scenes = [f"scene{i}" for i in range(6)]
    dataframe = _train_dataframe(scenes)

    train_df, val_df = split.split_scenes(dataframe, val_fraction=0.33, seed=42, stratify_by="name")

    assert set(train_df["name"]) & set(val_df["name"]) == set()


def test_every_row_is_preserved_across_the_two_splits():
    """train + val together account for every input row -- nothing dropped."""
    scenes = [f"scene{i}" for i in range(6)]
    dataframe = _train_dataframe(scenes)

    train_df, val_df = split.split_scenes(dataframe, val_fraction=0.33, seed=42, stratify_by="name")

    assert len(train_df) + len(val_df) == len(dataframe)


def test_split_is_deterministic_for_a_fixed_seed():
    """The same seed produces the same val-scene set across repeated calls."""
    scenes = [f"scene{i}" for i in range(10)]
    dataframe = _train_dataframe(scenes)

    _, val_df_1 = split.split_scenes(dataframe, val_fraction=0.3, seed=7, stratify_by="name")
    _, val_df_2 = split.split_scenes(dataframe, val_fraction=0.3, seed=7, stratify_by="name")

    assert set(val_df_1["name"]) == set(val_df_2["name"])


def test_val_fraction_approximates_configured_value_at_scene_level():
    """val_fraction is applied to unique scene count, not row count."""
    scenes = [f"scene{i}" for i in range(20)]
    dataframe = _train_dataframe(scenes)

    _, val_df = split.split_scenes(dataframe, val_fraction=0.25, seed=1, stratify_by="name")

    val_scene_count = val_df["name"].nunique()
    assert abs(val_scene_count - 5) <= 1  # 25% of 20 scenes, tolerate rounding


def test_repoint_folder_only_changes_folder_column(tmp_path):
    """repoint_folder() rewrites `folder` to selected_root/<id> and leaves every other column alone."""
    dataframe = pd.DataFrame(
        {"id": ["ang1_w0", "ang1_w1"], "name": ["ang1", "ang1"], "folder": ["/orig/a", "/orig/b"]}
    )
    selected_root = tmp_path / "selected"

    result = split.repoint_folder(dataframe, selected_root)

    assert list(result["folder"]) == [
        str(selected_root / "ang1_w0"),
        str(selected_root / "ang1_w1"),
    ]
    assert list(result["name"]) == ["ang1", "ang1"]  # untouched


def test_run_writes_train_val_test_csvs_with_no_scene_leakage(tmp_path):
    """End-to-end: run() writes train/val/test.csv, preserves test membership, and avoids scene leakage."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True)
    train_scenes = [f"scene{i}" for i in range(8)]
    _train_dataframe(train_scenes).to_csv(raw_root / "train.csv", index=False)
    test_df = pd.DataFrame(
        {"id": ["test_scene_w0"], "name": ["test_scene"], "folder": ["/orig/test_scene_w0"]}
    )
    test_df.to_csv(raw_root / "test.csv", index=False)

    processed_root = tmp_path / "processed"
    cfg = SimpleNamespace(
        paths=SimpleNamespace(raw_root=str(raw_root), processed_root=str(processed_root)),
        dataset_cfg=SimpleNamespace(train_csv="train.csv", test_csv="test.csv"),
        split=SimpleNamespace(val_fraction=0.25, seed=42, stratify_by="name"),
    )

    split.run(cfg)

    splits_root = processed_root / "splits"
    train_out = pd.read_csv(splits_root / "train.csv")
    val_out = pd.read_csv(splits_root / "val.csv")
    test_out = pd.read_csv(splits_root / "test.csv")

    assert set(train_out["name"]) & set(val_out["name"]) == set()
    assert list(test_out["id"]) == ["test_scene_w0"]
    assert test_out["folder"].iloc[0] == str(processed_root / "selected" / "test_scene_w0")
