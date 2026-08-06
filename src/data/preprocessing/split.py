"""Stage 2 of the TASK-1.2 DVC pipeline: scene-stratified train/val/test split.

STARCOP's own train_csv/test_csv already label rows by subset, but provide
no val split -- its own DataModule just reuses test as val. This stage
carves val out of train's *scenes* (`stratify_by`, e.g. flightline `name`)
rather than rows, so one flightline's overlapping windows never land on
both sides of the split. `test` rows pass through unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def split_scenes(
    dataframe: pd.DataFrame, val_fraction: float, seed: int, stratify_by: str = "name"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve val_fraction of unique `stratify_by` values out of dataframe into val.

    Deterministic for a fixed seed. Never splits one scene's rows across
    train/val -- every row for a given scene goes entirely to one side.
    """
    scene_names = dataframe[stratify_by].unique()
    shuffled = np.random.default_rng(seed).permutation(scene_names)
    n_val = round(len(shuffled) * val_fraction)
    val_names = set(shuffled[:n_val])

    is_val = dataframe[stratify_by].isin(val_names)
    train_df = dataframe[~is_val].reset_index(drop=True)
    val_df = dataframe[is_val].reset_index(drop=True)
    return train_df, val_df


def repoint_folder(dataframe: pd.DataFrame, selected_root: Path) -> pd.DataFrame:
    """Point `folder` at stage 1's selected/<id> output; leave every other column untouched."""
    dataframe = dataframe.copy()
    dataframe["folder"] = dataframe["id"].apply(lambda scene_id: str(selected_root / scene_id))
    return dataframe


def run(cfg) -> None:
    raw_root = Path(cfg.paths.raw_root)
    selected_root = Path(cfg.paths.processed_root) / "selected"
    splits_root = Path(cfg.paths.processed_root) / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)

    train_full = pd.read_csv(raw_root / cfg.dataset_cfg.train_csv)
    test_df = pd.read_csv(raw_root / cfg.dataset_cfg.test_csv)

    train_df, val_df = split_scenes(
        train_full, cfg.split.val_fraction, cfg.split.seed, cfg.split.stratify_by
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        repoint_folder(split_df, selected_root).to_csv(splits_root / f"{name}.csv", index=False)


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="data")
    def _run(cfg):
        run(cfg)

    _run()


if __name__ == "__main__":
    main()
