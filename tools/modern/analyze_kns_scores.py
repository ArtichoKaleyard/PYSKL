"""Analyze KNS test-time score files against a 1-clip uniform baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from mmengine.fileio import dump, load

from common import ensure_parent


def parse_named_path(value: str) -> tuple[str, str]:
    """Parse a NAME=PATH command line item."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {value!r}.")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {value!r}.")
    return name, path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Analyze KNS score migration tables.")
    parser.add_argument("--score", action="append", type=parse_named_path, required=True, help="Named score path.")
    parser.add_argument(
        "--metadata", action="append", type=parse_named_path, required=True, help="Named sampler metadata path."
    )
    parser.add_argument("--baseline", default="E1", help="Baseline experiment name.")
    parser.add_argument("--ten-clip", default="E5", help="10-clip reference experiment name.")
    parser.add_argument("--out", required=True, help="Output analysis json path.")
    return parser.parse_args()


def load_scores(path: str) -> np.ndarray:
    """Load a score pickle and return an ``N x C`` array."""

    scores = load(path)
    array = np.asarray(scores, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D score array in {path}, got shape {array.shape}.")
    return array


def topk_correct(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean vector indicating whether labels are in top-k."""

    topk = np.argsort(scores, axis=1)[:, -k:]
    return (topk == labels[:, None]).any(axis=1)


def mean_class_accuracy(correct: np.ndarray, labels: np.ndarray) -> float:
    """Compute unweighted mean class accuracy."""

    values = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.any():
            values.append(float(correct[mask].mean()))
    return float(np.mean(values)) if values else 0.0


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute top-1, top-5 and mean class accuracy."""

    top1 = topk_correct(scores, labels, 1)
    top5 = topk_correct(scores, labels, 5)
    return dict(
        top1=float(top1.mean()),
        top5=float(top5.mean()),
        mean_class_accuracy=mean_class_accuracy(top1, labels),
    )


def migration(base_correct: np.ndarray, target_correct: np.ndarray) -> dict[str, int]:
    """Count prediction migration groups against the baseline."""

    return dict(
        both_correct=int(np.logical_and(base_correct, target_correct).sum()),
        target_fixes=int(np.logical_and(~base_correct, target_correct).sum()),
        target_breaks=int(np.logical_and(base_correct, ~target_correct).sum()),
        both_wrong=int(np.logical_and(~base_correct, ~target_correct).sum()),
    )


def subset_summary(mask: np.ndarray, records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize metadata for a selected sample subset."""

    selected = [record for record, keep in zip(records, mask) if keep]
    confidences = []
    for record in selected:
        for meta in record.get("kns_meta", []):
            if isinstance(meta, dict) and str(meta.get("sampler", "")).startswith("KNS-"):
                confidences.append(float(meta.get("mean_confidence", 0.0)))
    lengths = [int(record["total_frames"]) for record in selected]
    return dict(
        count=len(selected),
        mean_total_frames=float(np.mean(lengths)) if lengths else 0.0,
        mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
    )


def main() -> int:
    """Create the analysis report."""

    args = parse_args()
    score_paths = dict(args.score)
    metadata_paths = dict(args.metadata)
    missing_meta = sorted(set(score_paths) - set(metadata_paths))
    if missing_meta:
        raise ValueError(f"Missing metadata for experiments: {missing_meta}")

    scores = {name: load_scores(path) for name, path in score_paths.items()}
    records = {name: load(path) for name, path in metadata_paths.items()}
    labels = np.asarray([record["label"] for record in records[args.baseline]], dtype=np.int64)
    sample_ids = [record["sample_id"] for record in records[args.baseline]]

    for name, array in scores.items():
        if len(array) != len(labels):
            raise ValueError(f"{name} has {len(array)} scores but baseline has {len(labels)} labels.")
        ids = [record["sample_id"] for record in records[name]]
        if ids != sample_ids:
            raise ValueError(f"{name} metadata sample order differs from baseline.")

    correct = {name: topk_correct(array, labels, 1) for name, array in scores.items()}
    base_correct = correct[args.baseline]
    ten_clip_fixed = np.logical_and(~base_correct, correct[args.ten_clip])

    migrations = {}
    overlap = {}
    distributions = {}
    for name, target_correct in correct.items():
        if name == args.baseline:
            continue
        migrations[name] = migration(base_correct, target_correct)
        target_fixed = np.logical_and(~base_correct, target_correct)
        denom = int(ten_clip_fixed.sum())
        overlap[name] = dict(
            fixed_by_10clip=denom,
            fixed_by_target=int(target_fixed.sum()),
            shared_with_10clip=int(np.logical_and(ten_clip_fixed, target_fixed).sum()),
            ten_clip_gain_coverage=float(np.logical_and(ten_clip_fixed, target_fixed).sum() / denom) if denom else 0.0,
        )
        distributions[name] = dict(
            fixes=subset_summary(np.logical_and(~base_correct, target_correct), records[name]),
            breaks=subset_summary(np.logical_and(base_correct, ~target_correct), records[name]),
        )

    report = dict(
        metrics={name: metrics(array, labels) for name, array in scores.items()},
        migration_from_baseline=migrations,
        ten_clip_gain_overlap=overlap,
        fixes_breaks_metadata=distributions,
        files=dict(scores=score_paths, metadata=metadata_paths),
    )

    ensure_parent(args.out)
    dump(report, args.out)
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
