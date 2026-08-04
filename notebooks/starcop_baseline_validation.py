"""baseline validation — local equivalent of
`vendor/starcop/notebooks/model_demos_AVIRIS.ipynb`.

The original notebook is written for Google Colab (installs its own package
versions, clones the repo into /content, uses CUDA). This script reproduces
its HyperSTARCOP / MultiSTARCOP inference cells for a local run against
Environment A (`vendor/starcop/.venv`, Python 3.10, the original STARCOP
torch 1.13.1 / pytorch-lightning 1.6.4 stack), using the mini dataset and
pretrained checkpoints fetched by `src/data/download_mini_dataset.py`.

Usage:
    uv run --python vendor/starcop/.venv/bin/python notebooks/starcop_baseline_validation.py
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARCOP_ROOT = PROJECT_ROOT / "vendor" / "starcop"
DATA_ROOT = PROJECT_ROOT / "data" / "starcop_mini"
MODELS_ROOT = PROJECT_ROOT / "models" / "starcop_baseline"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "baseline_validation"

sys.path.insert(0, str(STARCOP_ROOT))

from starcop.torch_utils import to_device  # noqa: E402
import starcop.plot as starcoplot  # noqa: E402
from starcop.data.datamodule import Permian2019DataModule  # noqa: E402
from starcop.models.model_module import ModelModule  # noqa: E402

device = torch.device("cpu")
config_general = omegaconf.OmegaConf.load(STARCOP_ROOT / "scripts" / "configs" / "config.yaml")
root_folder = str(DATA_ROOT)


def load_model_with_datamodule(model_path: Path, config_path: Path):
    config_model = omegaconf.OmegaConf.load(config_path)
    config = omegaconf.OmegaConf.merge(config_general, config_model)

    dataset_dict = ast.literal_eval(config_model["_content"]["value"]["dataset"])
    dataset_dict["root_folder"] = root_folder
    dataset_dict["train_csv"] = str(DATA_ROOT / "train_mini10.csv")
    config.dataset = dataset_dict
    config.products_plot = config_model["_content"]["value"]["products_plot"]

    data_module = Permian2019DataModule(config)
    data_module.test_csv = str(DATA_ROOT / "test_mini10.csv")
    data_module.settings["dataset"] = dataset_dict
    data_module.prepare_data()

    model = ModelModule.load_from_checkpoint(str(model_path), settings=config, map_location=device)
    model.to(device)
    model.eval()

    print(
        f"Loaded model with {model.num_channels} input channels, "
        f"data module with {len(data_module.input_products)} inputs: {data_module.input_products}"
    )
    return model, data_module, config


def validate_model(name: str, model_dir: Path) -> dict:
    print(f"\n=== {name} ===")
    model_path = model_dir / "final_checkpoint_model.ckpt"
    config_path = model_dir / "config.yaml"
    model, data_module, config = load_model_with_datamodule(model_path, config_path)

    # Save one sample segmentation mask figure — the "valid segmentation
    # mask PNG" required by TASK-0.3's validation criterion.
    dataloader = data_module.test_dataloader(batch_size=1)
    sample = next(iter(dataloader))
    sample = model.batch_with_preds(to_device(sample, model.device))
    fig, _ = starcoplot.plot_batch(
        to_device(sample, "cpu"),
        input_products=config.dataset.input_products,
        products_plot=config.products_plot,
        figsize_ax=(4, 4),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUTPUT_DIR / f"{name.lower()}_sample_mask.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"Saved segmentation mask sample: {fig_path}")

    # Aggregate pixel-level metrics over the full mini test split.
    # `starcop.validation.run_validation`'s built-in aggregation assumes the
    # full dataset's easy/hard difficulty split, which the mini subset
    # doesn't carry — so metrics are computed directly here instead of
    # depending on that helper (no change to the submodule needed).
    dataloader = data_module.test_dataloader(batch_size=1)
    y_true, y_pred = [], []
    for plume_data in tqdm(dataloader, desc=f"{name} test set"):
        plume_data = model.batch_with_preds(to_device(plume_data, model.device))
        y_true.append(plume_data["output_norm"].cpu().numpy().astype(int).ravel())
        y_pred.append(plume_data["pred_binary"].cpu().numpy().astype(int).ravel())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    metrics = {
        "overall_accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_methane": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "iou_methane": float(jaccard_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "n_test_samples": len(dataloader),
        "n_pixels": int(y_true.size),
    }
    print(f"{name} metrics: {metrics}")
    return metrics


def main() -> None:
    results = {}
    results["HyperSTARCOP"] = validate_model("HyperSTARCOP", MODELS_ROOT / "hyperstarcop_magic_rgb")
    results["MultiSTARCOP"] = validate_model("MultiSTARCOP", MODELS_ROOT / "multistarcop_varon")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved metrics: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
