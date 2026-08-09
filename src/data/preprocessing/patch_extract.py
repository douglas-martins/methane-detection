"""Stage 3 of the TASK-1.2 DVC pipeline: patch extraction.

Reuses vendor/starcop's own tiled_dataframe (via _vendor_starcop) rather
than reimplementing window creation or the frac_positives/has_plume
computation -- it already does exactly the tiling STARCOP's own training
pipeline does. The one gap: tiled_dataframe hardcodes its has_plume rule to
`frac_positives > 10/64**2`, with no parameter to change it. So this stage
reuses tiled_dataframe for the expensive part (window creation, reading
real label data) and then recomputes has_plume itself against the
configured `patch.has_plume_threshold`, overriding the hardcoded value.
"""

from pathlib import Path

import pandas as pd

from _vendor_starcop import tiled_dataframe


def patch_scenes(
    dataframe: pd.DataFrame,
    patch_size: list[int],
    overlap: list[int],
    output_products: list[str],
    has_plume_threshold: float,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Tile `dataframe`'s scenes into patches, with a configurable has_plume threshold."""
    indexed = dataframe.set_index("id")
    tiled = tiled_dataframe(
        indexed,
        tile_size=tuple(patch_size),
        overlap=tuple(overlap),
        output_products=output_products,
        num_workers=num_workers,
    )
    tiled["has_plume"] = tiled["frac_positives"] > has_plume_threshold
    return tiled


def run(cfg) -> None:
    splits_root = Path(cfg.paths.processed_root) / "splits"
    patches_root = Path(cfg.paths.processed_root) / "patches"
    patches_root.mkdir(parents=True, exist_ok=True)
    patch_size = list(cfg.patch.size)

    for name in ["train", "val", "test"]:
        dataframe = pd.read_csv(splits_root / f"{name}.csv")
        tiled = patch_scenes(
            dataframe,
            patch_size=patch_size,
            overlap=list(cfg.patch.overlap),
            output_products=list(cfg.dataset_cfg.output_products),
            has_plume_threshold=cfg.patch.has_plume_threshold,
            num_workers=cfg.patch.num_workers,
        )
        out_columns = [c for c in tiled.columns if c != "window"]
        out_path = patches_root / f"{name}_tiled_{patch_size[0]}_{patch_size[1]}.csv"
        tiled[out_columns].to_csv(out_path)


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="data")
    def _run(cfg):
        run(cfg)

    _run()


if __name__ == "__main__":
    main()
