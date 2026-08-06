"""Stage 1 of the TASK-1.2 DVC pipeline: select + validate.

STARCOP's per-scene folders already contain corrected, per-band TOA
GeoTIFFs plus a precomputed mag1c product -- there's no atmospheric
correction left to do here, it already happened upstream. STARCOP also
never bakes band normalization into files on disk (see
_vendor_starcop.BAND_NORMALIZATION): it applies a fixed offset/factor/clip
table at train time via DataNormalizer, reading raw per-band tifs. So this
stage:

  1. copies only the configured input_products/output_products band tifs
     for each scene into processed_root/selected/<scene_id>/, unchanged,
  2. raises if any configured band contains NaN/Inf,
  3. flags (does not reject) scenes whose values would fall outside their
     BAND_NORMALIZATION clip range once normalized -- visibility for the
     TASK-1.3 dataset report, not a hard gate.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio

from _vendor_starcop import BAND_NORMALIZATION


def select_scene(
    scene_folder: Path,
    output_folder: Path,
    input_products: list[str],
    output_products: list[str],
) -> list[str]:
    """Copy one scene's configured bands; return band names flagged by the range check."""
    output_folder.mkdir(parents=True, exist_ok=True)
    flagged = []

    for band_name in [*input_products, *output_products]:
        src_path = scene_folder / f"{band_name}.tif"
        with rasterio.open(src_path) as src:
            array = src.read(1)

        if not np.isfinite(array).all():
            raise ValueError(
                f"Band '{band_name}' in scene '{scene_folder.name}' contains NaN/Inf values"
            )

        if band_name in BAND_NORMALIZATION:
            norm = BAND_NORMALIZATION[band_name]
            normalized = (array - norm["offset"]) / norm["factor"]
            clip_min, clip_max = norm["clip"]
            if (normalized < clip_min).any() or (normalized > clip_max).any():
                flagged.append(band_name)

        shutil.copy2(src_path, output_folder / f"{band_name}.tif")

    return flagged


def run(cfg) -> None:
    raw_root = Path(cfg.paths.raw_root)
    selected_root = Path(cfg.paths.processed_root) / "selected"
    input_products = list(cfg.dataset_cfg.input_products)
    output_products = list(cfg.dataset_cfg.output_products)

    range_check = {}
    for scene_folder in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        flagged = select_scene(
            scene_folder, selected_root / scene_folder.name, input_products, output_products
        )
        if flagged:
            range_check[scene_folder.name] = flagged

    selected_root.mkdir(parents=True, exist_ok=True)
    (selected_root / "range_check.json").write_text(json.dumps(range_check, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="data")
    def _run(cfg):
        run(cfg)

    _run()


if __name__ == "__main__":
    main()
