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
import pandas as pd
import rasterio

from _vendor_starcop import BAND_NORMALIZATION


def find_scene_folder(raw_root: Path, scene_id: str) -> Path:
    """Locate a scene's folder, flat (mini) or one subfolder level deep (raw).

    Collects both the flat and nested candidates before deciding, so a scene
    that exists in both places is caught as ambiguous rather than silently
    resolving to whichever candidate happened to be checked first.
    """
    direct = raw_root / scene_id
    matches = [direct] if direct.is_dir() else []
    matches += list(raw_root.glob(f"*/{scene_id}"))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Scene '{scene_id}' found in multiple locations under {raw_root}: {matches}")
    raise FileNotFoundError(f"Scene '{scene_id}' not found under {raw_root}")


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
    """DVC entry point: select + validate every scene in the configured dataset's manifest."""
    raw_root = Path(cfg.paths.raw_root)
    selected_root = Path(cfg.paths.processed_root) / "selected"
    input_products = list(cfg.dataset_cfg.input_products)
    output_products = list(cfg.dataset_cfg.output_products)

    train_ids = pd.read_csv(raw_root / cfg.dataset_cfg.train_csv)["id"]
    test_ids = pd.read_csv(raw_root / cfg.dataset_cfg.test_csv)["id"]
    scene_ids = sorted(set(train_ids) | set(test_ids))

    range_check = {}
    missing = []
    for scene_id in scene_ids:
        try:
            scene_folder = find_scene_folder(raw_root, scene_id)
        except FileNotFoundError:
            missing.append(scene_id)
            continue
        flagged = select_scene(scene_folder, selected_root / scene_id, input_products, output_products)
        if flagged:
            range_check[scene_id] = flagged

    selected_root.mkdir(parents=True, exist_ok=True)
    (selected_root / "range_check.json").write_text(json.dumps(range_check, indent=2))
    missing_report = selected_root / "missing_scenes.json"
    if missing:
        missing_report.write_text(json.dumps(missing, indent=2))
    else:
        missing_report.unlink(missing_ok=True)


def main() -> None:
    """CLI entry point: resolve the Hydra config and dispatch to run()."""
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="data")
    def _run(cfg):
        """Hydra-decorated wrapper receiving the composed config."""
        run(cfg)

    _run()


if __name__ == "__main__":
    main()
