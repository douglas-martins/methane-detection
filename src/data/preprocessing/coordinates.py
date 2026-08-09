"""Stage 5 of the TASK-1.2 DVC pipeline: per-scene geographic coordinates.

STARCOP's train/test CSVs carry no lat/lon column -- only a `folder` path
string and a date. Every scene's GeoTIFFs do carry a real CRS/bounds
(e.g. EPSG:32613 for the Permian Basin AVIRIS-NG scenes), so coordinates
are derived here by reprojecting each scene's bounds to WGS84 and taking
the centroid. Feeds the TASK-1.3 dataset report.
"""

import csv
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


def scene_centroid_latlon(band_path: Path) -> tuple[float, float]:
    """Return (lat, lon) in WGS84 for one band's raster extent."""
    with rasterio.open(band_path) as src:
        left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return (bottom + top) / 2, (left + right) / 2


def collect_scene_coordinates(selected_root: Path, reference_band: str) -> list[dict]:
    """One row per scene folder under `selected_root`, keyed off `reference_band.tif`.

    `reference_band` should be a band present in every scene -- normalize.py
    (stage 1) already guarantees every configured input_products band was
    copied and validated for every scene it kept.
    """
    rows = []
    for scene_folder in sorted(p for p in selected_root.iterdir() if p.is_dir()):
        lat, lon = scene_centroid_latlon(scene_folder / f"{reference_band}.tif")
        rows.append({"scene_id": scene_folder.name, "lat": lat, "lon": lon})
    return rows


def run(cfg) -> None:
    """DVC entry point: write `scene_coordinates.csv` for the configured dataset."""
    selected_root = Path(cfg.paths.processed_root) / "selected"
    coordinates_root = Path(cfg.paths.processed_root) / "coordinates"
    coordinates_root.mkdir(parents=True, exist_ok=True)

    reference_band = cfg.dataset_cfg.input_products[0]
    rows = collect_scene_coordinates(selected_root, reference_band)

    with (coordinates_root / "scene_coordinates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "lat", "lon"])
        writer.writeheader()
        writer.writerows(rows)


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
