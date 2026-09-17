"""One-time precompute of the on-disk patch cache (see `patch_cache.py`).

Reads every patch through `PatchDataset`'s existing live read path
(`augment=False` -- same pre-augmentation contract as its own RAM cache)
exactly once, and writes the flat float16/uint8 arrays `PatchDataset(...,
cache_dir=...)` reads back. Uses a plain `DataLoader` purely to parallelize
that one-time read across `num_workers` processes (matching `train.py`'s
own real-run `num_workers`) -- `shuffle=False` keeps batch order aligned
with `patches_df`'s row order, which the cache's array position depends on.

Usage: python precompute_patch_cache.py dataset=starcop_raw [splits=train,val,test]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import patch_cache
from dataset import PatchDataset
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COURSEWORK_ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = _COURSEWORK_ROOT / "patch_cache"


def build_cache_for_split(
    dataset: str,
    patches_df: pd.DataFrame,
    cache_dir: Path,
    num_workers: int = 4,
    batch_size: int = 256,
) -> None:
    """Read every patch in `patches_df` once and write it to `cache_dir` (see `patch_cache.py`)."""
    live_dataset = PatchDataset(patches_df, dataset=dataset, augment=False, max_cache_bytes=0)
    loader = DataLoader(live_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n = len(live_dataset)
    inputs = None
    # Both placeholders exist only so the loop's first iteration can detect
    # "not yet allocated" via `if inputs is None:` below (which allocates both
    # arrays together) -- outputs's own placeholder value is never read before
    # that reassignment, so its exact sentinel value is unobservable.
    outputs = None  # pragma: no mutate (never read before being reassigned, see above)
    offset = 0
    for batch in loader:
        # No separate `.astype(np.float16/np.uint8)` step on these: `inputs`/
        # `outputs` below are pre-allocated in those same dtypes, and a numpy
        # slice assignment already casts its RHS into the target array's own
        # dtype -- an explicit intermediate cast here would be redundant.
        batch_input_array = batch["input"].numpy()
        batch_output_array = batch["output"].numpy()
        if inputs is None:
            _, c, h, w = batch_input_array.shape
            # float16/uint8 here (rather than batch_input_array's own float32)
            # bounds this accumulator's peak memory for starcop_raw's 141k-patch
            # split (see the module docstring) -- patch_cache.write() re-casts
            # to these same dtypes before saving regardless, so the exact
            # intermediate dtype only ever affects peak memory, never a saved
            # value (see TestBuildCacheForSplitDefaults's dtype assertion,
            # which is what actually exercises this choice).
            inputs = np.empty((n, c, h, w), dtype=np.float16)
            outputs = np.empty((n, 1, h, w), dtype=np.uint8)
        b = batch_input_array.shape[0]
        inputs[offset : offset + b] = batch_input_array
        outputs[offset : offset + b] = batch_output_array
        offset += b
    patch_cache.write(cache_dir, patches_df, inputs, outputs)


def _parse_kv_args(argv: list[str]) -> dict[str, str]:
    """Parse `key=value` CLI args (Hydra-style, matching `train.py`/`confirm_raw.py`)."""
    parsed = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
    return parsed


def main() -> None:
    """CLI entry point -- see this module's own docstring for usage."""
    args = _parse_kv_args(sys.argv[1:])
    dataset = args.get("dataset", "starcop_mini")
    splits = args.get("splits", "train,val,test").split(",")
    patches_root = _REPO_ROOT / "data" / "processed" / dataset / "patches"

    for split in splits:
        csv_path = patches_root / f"{split}_tiled_128_128.csv"
        patches_df = pd.read_csv(csv_path, low_memory=False)
        cache_dir = _CACHE_ROOT / dataset / split
        start = time.perf_counter()
        build_cache_for_split(dataset, patches_df, cache_dir)
        elapsed = time.perf_counter() - start
        print(f"{dataset}/{split}: {len(patches_df)} patches -> {cache_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
