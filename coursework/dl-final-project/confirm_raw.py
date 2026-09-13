"""R1 contract confirmation (plan Section 0.1 / Sections 4-5): the coursework's own
preprocessing, loader, and architecture code, run against real patches from
the configured dataset -- shapes, dtypes, post-normalization ranges, band
order, label passthrough, and (Section 5) a forward+backward pass for every
model on a real batch. Minutes, not hours; no training.

Deliberately not a pytest suite: `data/` is entirely gitignored/DVC-tracked,
so a real slice of it cannot be committed as a test fixture, and a test that
`skip`s when the data is absent (CI included) would prove nothing about the
real path -- see Section 4's "Decisions taken at review time" note. This
script *is* the R1 check; its output is what gets recorded in the plan, the
same way Section 0.1's `dvc repro` run was recorded as a real run, not a
test result. Built in Section 4, extended here in Section 5 rather than
each section inventing its own check, per that same decision.

Usage: python confirm_raw.py [dataset=starcop_raw]
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from architectures import build_e1, build_e2, build_e3
from dataset import PatchDataset
from omegaconf import OmegaConf
from preprocessing import BAND_NORMALIZATION
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_SIZE = 300
_SEED = 42


def _parse_dataset_arg(argv: list[str]) -> str:
    """Extract `dataset=<name>` from CLI args (Hydra-style, matching the Makefile target)."""
    for arg in argv:
        if arg.startswith("dataset="):
            return arg.split("=", 1)[1]
    return "starcop_raw"


def load_sample(dataset: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Load the train patches CSV for `dataset` and return a seeded, scene-spanning sample."""
    patches_path = (
        _REPO_ROOT / "data" / "processed" / dataset / "patches" / "train_tiled_128_128.csv"
    )
    full = pd.read_csv(patches_path, low_memory=False)
    n = min(sample_size, len(full))
    return full.sample(n=n, random_state=seed).reset_index(drop=True)


def _expected_patch_size() -> tuple[int, int]:
    """Read `patch.size` from configs/data.yaml, so a future config change can't go unnoticed."""
    config = OmegaConf.load(_REPO_ROOT / "configs" / "data.yaml")
    height, width = config.patch.size
    return int(height), int(width)


def check_architectures(patch_dataset: PatchDataset, batch_size: int = 8) -> None:
    """R1 (Section 5): forward + backward pass for E1/E2/E3 on one real batch.

    A random `torch.randn` tensor can't catch a NaN arriving from raw
    `mag1c`, a dtype mismatch, or an all-negative batch -- all three are
    real `starcop_raw` conditions this checks against actual data instead.
    """
    loader = DataLoader(patch_dataset, batch_size=min(batch_size, len(patch_dataset)))
    batch = next(iter(loader))
    input_batch, output_batch = batch["input"], batch["output"]
    print(f"  architecture batch: {tuple(input_batch.shape)}")

    for name, build in [("E1", build_e1), ("E2", build_e2), ("E3", build_e3)]:
        model = build() if name == "E1" else build(pretrained=True)
        model.train()
        logits = model(input_batch)
        assert logits.shape == output_batch.shape, (
            f"{name}: expected output shape {tuple(output_batch.shape)}, got {tuple(logits.shape)}"
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, output_batch)
        model.zero_grad()
        loss.backward()
        assert torch.isfinite(loss), f"{name}: non-finite loss {loss.item()}"
        missing_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing_grad, f"{name}: {len(missing_grad)} parameters got no gradient"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,} params, loss={loss.item():.4f}, all gradients present")


def run(dataset: str) -> None:
    """Run every R1 check against a real sample of `dataset` and print a pass/fail summary."""
    print(f"R1 confirmation -- dataset={dataset}")

    sample = load_sample(dataset, _SAMPLE_SIZE, _SEED)
    scene_count = sample["name"].nunique()
    print(f"  sampled {len(sample)} patches across {scene_count} distinct scenes")
    assert scene_count > 1, "sample must span more than one scene to be a real R1 check"

    patch_dataset = PatchDataset(sample, dataset=dataset, augment=False)
    input_products = patch_dataset.input_products
    patch_height, patch_width = _expected_patch_size()
    print(f"  input_products (band order): {input_products}")
    print(f"  expected patch size (configs/data.yaml): {patch_height}x{patch_width}")

    negative_count = 0
    for index in range(len(patch_dataset)):
        item = patch_dataset[index]
        input_tensor, output_tensor = item["input"], item["output"]

        expected_input_shape = (len(input_products), patch_height, patch_width)
        assert tuple(input_tensor.shape) == expected_input_shape, (
            f"expected input shape {expected_input_shape}, got {tuple(input_tensor.shape)}"
        )
        expected_output_shape = (1, patch_height, patch_width)
        assert tuple(output_tensor.shape) == expected_output_shape, (
            f"expected output shape {expected_output_shape}, got {tuple(output_tensor.shape)}"
        )
        assert str(input_tensor.dtype) == "torch.float32", input_tensor.dtype
        for product in input_products:
            clip_min, clip_max = BAND_NORMALIZATION[product]["clip"]
            assert input_tensor.min() >= clip_min - 1e-5, (
                f"{product} min {input_tensor.min()} below clip {clip_min}"
            )
            assert input_tensor.max() <= clip_max + 1e-5, (
                f"{product} max {input_tensor.max()} above clip {clip_max}"
            )

        label_values = set(output_tensor.unique().tolist())
        assert label_values <= {0.0, 1.0}, f"label has non-binary values: {label_values}"
        if label_values == {0.0} or len(label_values) == 0:
            negative_count += 1

    total = len(patch_dataset)
    negative_fraction = negative_count / total
    print(f"  shapes/dtypes/ranges/band-order/label-passthrough: PASSED ({total} patches)")
    print(
        f"  all-negative patches in sample: {negative_count}/{total} "
        f"({negative_fraction:.1%}) -- informational, a design input for"
        " Sections 6/7's sampling strategy, not a failure here"
    )

    check_architectures(patch_dataset)
    print("R1 PASSED")


def main() -> None:
    """CLI entry point: resolve `dataset=` from argv and run the R1 check."""
    dataset = _parse_dataset_arg(sys.argv[1:])
    try:
        run(dataset)
    except AssertionError as error:
        print(f"R1 FAILED: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
