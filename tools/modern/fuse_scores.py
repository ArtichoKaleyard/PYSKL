"""Fuse score pickles and evaluate NTU-style classification metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mmengine.fileio import dump, load


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Fuse PYSKL score files.")
    parser.add_argument("--scores", nargs="+", required=True, help="Score pkl files.")
    parser.add_argument("--ann-file", default="data/nturgbd/ntu60_hrnet.pkl")
    parser.add_argument("--split", default="xsub_val")
    parser.add_argument("--weights", nargs="*", default=["1:1", "2:1", "1:2"])
    parser.add_argument("--out", required=True, help="Output json path.")
    return parser.parse_args()


def load_labels(ann_file: str, split: str) -> list[int]:
    """Load labels from a PYSKL annotation file."""

    data = load(ann_file)
    annotations = data["annotations"]
    split_set = set(data["split"][split])
    key = "filename" if "filename" in annotations[0] else "frame_dir"
    return [item["label"] for item in annotations if item[key] in split_set]


def top_k_accuracy(scores: np.ndarray, labels: np.ndarray, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    """Compute top-k accuracy."""

    results = []
    for k in topk:
        predictions = np.argsort(scores, axis=1)[:, -k:]
        results.append(float(np.mean([label in pred for label, pred in zip(labels, predictions)])))
    return results


def mean_class_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean class accuracy."""

    preds = np.argmax(scores, axis=1)
    values = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        values.append(float(np.mean(preds[mask] == labels[mask])))
    return float(np.mean(values))


def parse_weight_spec(spec: str, expected: int) -> list[float]:
    """Parse a colon-separated fusion weight spec."""

    values = [float(item) for item in spec.split(":")]
    if len(values) != expected:
        raise ValueError(f"Weight spec {spec!r} does not match {expected} score files.")
    return values


def main() -> int:
    """Run score fusion."""

    args = parse_args()
    score_sets = [np.asarray(load(path), dtype=np.float32) for path in args.scores]
    labels = np.asarray(load_labels(args.ann_file, args.split), dtype=np.int64)
    if any(len(scores) != len(labels) for scores in score_sets):
        lengths = [len(scores) for scores in score_sets]
        raise ValueError(f"Score lengths {lengths} do not match labels {len(labels)}.")

    report = {}
    for spec in args.weights:
        coeffs = parse_weight_spec(spec, len(score_sets))
        fused = sum(weight * scores for weight, scores in zip(coeffs, score_sets))
        top1, top5 = top_k_accuracy(fused, labels, topk=(1, 5))
        report[spec] = {
            "top1": top1,
            "top5": top5,
            "mean_class_accuracy": mean_class_accuracy(fused, labels),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(report, str(out))
    for spec, metrics in report.items():
        print(
            f"{spec}: top1={metrics['top1']:.04f}, "
            f"top5={metrics['top5']:.04f}, mean_class={metrics['mean_class_accuracy']:.04f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
