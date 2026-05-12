"""Shared helpers for the modern PYSKL experiment entrypoints."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmengine.config import Config
from mmengine.fileio import dump
from mmengine.runner import load_checkpoint

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyskl  # noqa: F401  # installs legacy OpenMMLab shims
from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model


def set_seed(seed: int) -> None:
    """Set random seeds used by the experiment scripts."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Move nested tensors in a batch to the target device."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def ensure_parent(path: str | Path) -> None:
    """Create the parent directory for a file path."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> Config:
    """Load a config file and ensure PYSKL modules are registered."""

    return Config.fromfile(str(path))


def build_split_dataloader(cfg: Config, split: str, shuffle: bool = False):
    """Build dataset and dataloader for a config split.

    Args:
        cfg: Loaded experiment config.
        split: One of ``train``, ``val`` or ``test``.
        shuffle: Whether to shuffle samples.

    Returns:
        Tuple of dataset and dataloader.
    """

    data_cfg = cfg.data[split]
    dataset = build_dataset(data_cfg, dict(test_mode=(split != "train")))
    loader_overrides = cfg.data.get(f"{split}_dataloader", {})
    dataloader = build_dataloader(
        dataset,
        videos_per_gpu=loader_overrides.get("videos_per_gpu", cfg.data.get("videos_per_gpu", 1)),
        workers_per_gpu=loader_overrides.get("workers_per_gpu", cfg.data.get("workers_per_gpu", 0)),
        persistent_workers=cfg.data.get("persistent_workers", False),
        shuffle=shuffle,
        seed=cfg.get("seed", 0),
        **{key: value for key, value in loader_overrides.items() if key not in {"videos_per_gpu", "workers_per_gpu"}},
    )
    return dataset, dataloader


def build_model_from_config(cfg: Config, checkpoint: str | None = None, device: str = "cuda"):
    """Build a PYSKL model and optionally load a checkpoint."""

    model = build_model(cfg.model)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    progress_interval: int = 20,
    partial_out: str | Path | None = None,
    partial_interval: int = 50,
    max_batches: int | None = None,
) -> list[np.ndarray]:
    """Run model inference and collect score arrays.

    Args:
        model: Model used for evaluation.
        dataloader: Validation or test dataloader.
        device: Target torch device.
        progress_interval: Minimum seconds between progress log lines. Set to
            ``0`` to disable time-based progress logging.
        partial_out: Optional path for periodically dumping collected scores.
            This file is intentionally separate from the final output because it
            may contain only a prefix of the dataset.
        partial_interval: Number of batches between partial score dumps.
        max_batches: Optional batch limit for smoke or throughput checks.

    Returns:
        Collected per-sample score arrays.
    """

    model.eval()
    outputs: list[np.ndarray] = []
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
    start_time = time.monotonic()
    last_log_time = start_time
    partial_path = Path(partial_out) if partial_out else None
    if partial_path:
        ensure_parent(partial_path)

    for batch_idx, batch in enumerate(dataloader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        batch = move_to_device(batch, device)
        scores = model(return_loss=False, **batch)
        if isinstance(scores, np.ndarray):
            outputs.extend(list(scores))
        else:
            outputs.extend(scores)

        now = time.monotonic()
        should_log = progress_interval > 0 and now - last_log_time >= progress_interval
        if should_log:
            elapsed = now - start_time
            speed = len(outputs) / elapsed if elapsed > 0 else 0.0
            if total:
                percent = len(outputs) / total * 100
                remaining = (total - len(outputs)) / speed if speed > 0 else float("inf")
                print(
                    f"progress: {len(outputs)}/{total} ({percent:.2f}%), "
                    f"{speed:.2f} samples/s, eta {remaining / 60:.1f} min",
                    flush=True,
                )
            else:
                print(f"progress: {len(outputs)} samples, {speed:.2f} samples/s", flush=True)
            last_log_time = now

        if partial_path and partial_interval > 0 and batch_idx % partial_interval == 0:
            dump(outputs, str(partial_path))

    if partial_path:
        dump(outputs, str(partial_path))
    return outputs


def evaluate_and_dump(dataset, outputs: list[np.ndarray], out: str | Path, metrics: list[str]) -> dict[str, float]:
    """Dump score outputs and evaluate requested metrics."""

    ensure_parent(out)
    dataset.dump_results(outputs, out=str(out))
    eval_results = dataset.evaluate(outputs, metrics=metrics)
    dump(eval_results, str(Path(out).with_suffix(".metrics.json")))
    return eval_results


def resolve_checkpoint(path: str | None, cfg: Config) -> str | None:
    """Resolve a user checkpoint argument against config defaults."""

    if path:
        return path
    if cfg.get("load_from"):
        return cfg.load_from
    return None


def configure_threads() -> None:
    """Set conservative defaults for local experiment runs."""

    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
