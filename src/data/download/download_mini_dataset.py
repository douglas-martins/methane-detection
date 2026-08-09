"""Download the STARCOP mini demo assets used by TASK-0.3 baseline validation.

Fetches the same mini dataset and pretrained checkpoints that
`vendor/starcop/notebooks/model_demos_AVIRIS.ipynb` downloads on Google Colab,
so the notebook can run locally against Environment A
(`vendor/starcop/.venv`, Python 3.10) without any Colab-specific setup.

Usage:
    uv run --python vendor/starcop/.venv/bin/python src/data/download/download_mini_dataset.py
"""

import shutil
import zipfile
from pathlib import Path

import gdown

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "starcop_mini"
MODELS_DIR = PROJECT_ROOT / "models" / "starcop_baseline"

# (Google Drive file ID, output filename, extraction target) — same source
# files referenced by model_demos_AVIRIS.ipynb.
ASSETS = [
    ("1Qw96Drmk2jzBYSED0YPEUyuc2DnBechl", "STARCOP_mini.zip", DATA_DIR),
    ("1TXFlAHO_eRdfbJGLNNt3KY0lJqjm3fdX", "multistarcop_varon.zip", MODELS_DIR),
    ("1Kvnc_lOBn4z-xO1HFRyLZOMEldXWQvql", "hyperstarcop_magic_rgb.zip", MODELS_DIR),
]


def download_and_extract(file_id: str, filename: str, target_dir: Path) -> None:
    """Download `filename` from Google Drive into `target_dir`, extract it, then delete the zip."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / filename
    if not zip_path.exists():
        gdown.download(id=file_id, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    zip_path.unlink()

    # STARCOP_mini.zip wraps its contents in a top-level STARCOP_mini/ folder;
    # flatten it so everything lands directly in target_dir.
    nested_dir = target_dir / Path(filename).stem
    if nested_dir.is_dir():
        for item in nested_dir.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        nested_dir.rmdir()


def main() -> None:
    """Download and extract every asset in ASSETS."""
    for file_id, filename, target_dir in ASSETS:
        print(f"--- {filename} -> {target_dir} ---")
        download_and_extract(file_id, filename, target_dir)
    print("Done.")


if __name__ == "__main__":
    main()
