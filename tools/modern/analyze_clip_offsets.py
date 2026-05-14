"""Analyze which offsets in a multi-clip test actually contribute."""

from __future__ import annotations

import argparse
import itertools
from typing import Any

import numpy as np
from mmengine.fileio import dump, load

from common import ensure_parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Analyze per-offset usefulness in a multi-clip score file.")
    parser.add_argument("--per-clip-score", required=True, help="Score pkl with shape N x K x C.")
    parser.add_argument("--baseline-score", required=True, help="Low-cost baseline score pkl with shape N x C.")
    parser.add_argument("--ten-clip-score", required=True, help="Existing averaged 10-clip score pkl with shape N x C.")
    parser.add_argument("--metadata", required=True, help="Sampling metadata pkl exported for the 10-clip config.")
    parser.add_argument(
        "--primary-metric",
        choices=["top1", "mean_class_accuracy"],
        default="top1",
        help="Metric used to rank offset subsets.",
    )
    parser.add_argument(
        "--per-offset-score-out",
        default=None,
        help="Optional output pkl for grouped per-offset probabilities with shape N x num_offsets x C.",
    )
    parser.add_argument(
        "--per-offset-metadata-out",
        default=None,
        help="Optional output pkl for sample_id, label, offset_id and frame_inds aligned with per-offset scores.",
    )
    parser.add_argument("--out", required=True, help="Output analysis json path.")
    return parser.parse_args()


def load_2d_scores(path: str) -> np.ndarray:
    """Load a score pickle and validate it as ``N x C``."""

    array = np.asarray(load(path), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D score array in {path}, got shape {array.shape}.")
    return array


def load_per_clip_scores(path: str) -> np.ndarray:
    """Load a per-clip score pickle and validate it as ``N x K x C``."""

    array = np.asarray(load(path), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D per-clip score array in {path}, got shape {array.shape}.")
    return array


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the class dimension."""

    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def topk_correct(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Return whether each sample is correct under top-k accuracy."""

    topk = np.argsort(scores, axis=1)[:, -k:]
    return (topk == labels[:, None]).any(axis=1)


def mean_class_accuracy(correct: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean class accuracy from per-sample correctness."""

    values = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.any():
            values.append(float(correct[mask].mean()))
    return float(np.mean(values)) if values else 0.0


def score_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute top-1, top-5 and mean class accuracy."""

    top1 = topk_correct(scores, labels, 1)
    top5 = topk_correct(scores, labels, 5)
    return dict(
        top1=float(top1.mean()),
        top5=float(top5.mean()),
        mean_class_accuracy=mean_class_accuracy(top1, labels),
    )


def migration(base_correct: np.ndarray, target_correct: np.ndarray) -> dict[str, int]:
    """Count prediction migration groups."""

    return dict(
        both_correct=int(np.logical_and(base_correct, target_correct).sum()),
        target_fixes=int(np.logical_and(~base_correct, target_correct).sum()),
        target_breaks=int(np.logical_and(base_correct, ~target_correct).sum()),
        both_wrong=int(np.logical_and(~base_correct, ~target_correct).sum()),
    )


def temporal_probabilities(raw_probabilities: np.ndarray, num_offsets: int) -> tuple[np.ndarray, int]:
    """Group test-time views into temporal offsets.

    PoseC3D test configs use ``GeneratePoseTarget(double=True)``, so a
    10-clip sample is emitted as 20 model views: ten original temporal clips
    followed by ten horizontally flipped clips. The offset analysis should rank
    temporal positions, not count the flip augmentation as another offset.
    """

    num_views = raw_probabilities.shape[1]
    if num_views % num_offsets != 0:
        raise ValueError(f"Cannot group {num_views} views into {num_offsets} temporal offsets.")
    views_per_offset = num_views // num_offsets
    grouped = raw_probabilities.reshape(raw_probabilities.shape[0], views_per_offset, num_offsets, -1)
    return grouped.mean(axis=1), views_per_offset


def ensemble_scores(probabilities: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    """Average probabilities over the selected offsets."""

    return probabilities[:, offsets, :].mean(axis=1)


def offset_position_summary(records: list[dict[str, Any]], offset: int) -> dict[str, float]:
    """Summarize where an offset lands in normalized video time."""

    centers = []
    starts = []
    ends = []
    for record in records:
        clip_len = int(record["clip_len"])
        total = max(int(record["total_frames"]) - 1, 1)
        frame_inds = np.asarray(record["frame_inds"], dtype=np.float32)
        inds = frame_inds[offset * clip_len:(offset + 1) * clip_len]
        if inds.size == 0:
            continue
        starts.append(float(np.min(inds) / total))
        ends.append(float(np.max(inds) / total))
        centers.append(float(np.mean(inds) / total))
    return dict(
        mean_start=float(np.mean(starts)) if starts else 0.0,
        mean_center=float(np.mean(centers)) if centers else 0.0,
        mean_end=float(np.mean(ends)) if ends else 0.0,
    )


def build_per_offset_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact metadata aligned with a ``N x num_offsets x C`` score array."""

    output = []
    for sample_index, record in enumerate(records):
        clip_len = int(record["clip_len"])
        offsets = []
        frame_inds = np.asarray(record["frame_inds"], dtype=np.int64)
        for offset in range(int(record["num_clips"])):
            inds = frame_inds[offset * clip_len:(offset + 1) * clip_len]
            offsets.append(dict(offset_id=offset, frame_inds=inds.tolist()))
        output.append(
            dict(
                sample_index=sample_index,
                sample_id=record["sample_id"],
                label=int(record["label"]),
                total_frames=int(record["total_frames"]),
                clip_len=clip_len,
                offsets=offsets,
            )
        )
    return output


def greedy_selection(
    probabilities: np.ndarray,
    labels: np.ndarray,
    all_offsets: tuple[int, ...],
    primary_metric: str,
) -> list[dict[str, Any]]:
    """Select offsets one by one by maximizing the validation metric."""

    selected: list[int] = []
    remaining = set(all_offsets)
    rows = []
    while remaining:
        best_row = None
        for offset in sorted(remaining):
            candidate = tuple(selected + [offset])
            scores = ensemble_scores(probabilities, candidate)
            metrics = score_metrics(scores, labels)
            row = dict(
                cost=len(candidate),
                added_offset=offset,
                offsets=list(candidate),
                metrics=metrics,
            )
            if best_row is None or metrics[primary_metric] > best_row["metrics"][primary_metric]:
                best_row = row
        selected = best_row["offsets"]
        remaining.remove(best_row["added_offset"])
        rows.append(best_row)
    return rows


def main() -> int:
    """Run offset analysis."""

    args = parse_args()
    per_clip = load_per_clip_scores(args.per_clip_score)
    baseline = load_2d_scores(args.baseline_score)
    ten_clip = load_2d_scores(args.ten_clip_score)
    records = load(args.metadata)
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    sample_ids = [record.get("sample_id") for record in records]

    if len({len(per_clip), len(baseline), len(ten_clip), len(labels)}) != 1:
        raise ValueError("Score and metadata lengths do not match.")
    if any(sample_id is None for sample_id in sample_ids):
        raise ValueError("Metadata records must include sample_id.")

    raw_probabilities = softmax(per_clip)
    num_offsets_set = {int(record["num_clips"]) for record in records}
    if len(num_offsets_set) != 1:
        raise ValueError(f"Metadata records disagree on num_clips: {sorted(num_offsets_set)}.")
    num_offsets = num_offsets_set.pop()
    probabilities, views_per_offset = temporal_probabilities(raw_probabilities, num_offsets)
    all_offsets = tuple(range(num_offsets))
    all_ensemble = ensemble_scores(probabilities, all_offsets)

    if args.per_offset_score_out is not None:
        ensure_parent(args.per_offset_score_out)
        dump(probabilities.astype(np.float32), args.per_offset_score_out)
        print(f"wrote: {args.per_offset_score_out}")
    if args.per_offset_metadata_out is not None:
        ensure_parent(args.per_offset_metadata_out)
        dump(build_per_offset_metadata(records), args.per_offset_metadata_out)
        print(f"wrote: {args.per_offset_metadata_out}")

    base_correct = topk_correct(baseline, labels, 1)
    ten_correct = topk_correct(ten_clip, labels, 1)
    all_correct = topk_correct(all_ensemble, labels, 1)
    ten_clip_gain = np.logical_and(~base_correct, ten_correct)
    all_gain = np.logical_and(~base_correct, all_correct)
    gain_denominator = int(ten_clip_gain.sum())

    offset_rows = []
    for offset in all_offsets:
        scores = ensemble_scores(probabilities, (offset,))
        correct = topk_correct(scores, labels, 1)
        shared_gain = np.logical_and(ten_clip_gain, correct)
        row = dict(
            offset=offset,
            offsets=[offset],
            metrics=score_metrics(scores, labels),
            migration_from_baseline=migration(base_correct, correct),
            ten_clip_gain_coverage=float(shared_gain.sum() / gain_denominator) if gain_denominator else 0.0,
            ten_clip_gain_shared=int(shared_gain.sum()),
            position=offset_position_summary(records, offset),
        )
        offset_rows.append(row)

    best_by_cost = []
    for cost in range(1, num_offsets + 1):
        best_row = None
        for offsets in itertools.combinations(all_offsets, cost):
            scores = ensemble_scores(probabilities, offsets)
            metrics = score_metrics(scores, labels)
            row = dict(cost=cost, offsets=list(offsets), metrics=metrics)
            if best_row is None or metrics[args.primary_metric] > best_row["metrics"][args.primary_metric]:
                best_row = row
        best_by_cost.append(best_row)

    greedy_by_cost = greedy_selection(probabilities, labels, all_offsets, args.primary_metric)

    leave_one_out = []
    full_metrics = score_metrics(all_ensemble, labels)
    for offset in all_offsets:
        offsets = tuple(item for item in all_offsets if item != offset)
        scores = ensemble_scores(probabilities, offsets)
        metrics = score_metrics(scores, labels)
        leave_one_out.append(
            dict(
                removed_offset=offset,
                offsets=list(offsets),
                metrics=metrics,
                delta_from_all={key: float(metrics[key] - full_metrics[key]) for key in full_metrics},
            )
        )

    report = dict(
        num_samples=int(len(labels)),
        num_offsets=int(num_offsets),
        raw_views_per_sample=int(per_clip.shape[1]),
        views_per_offset=int(views_per_offset),
        primary_metric=args.primary_metric,
        baseline_metrics=score_metrics(baseline, labels),
        existing_ten_clip_metrics=score_metrics(ten_clip, labels),
        recomputed_all_offsets_metrics=full_metrics,
        recomputed_vs_existing_delta={
            key: float(full_metrics[key] - score_metrics(ten_clip, labels)[key]) for key in full_metrics
        },
        ten_clip_gain_count=gain_denominator,
        all_offsets_gain_count=int(all_gain.sum()),
        offset_metrics=offset_rows,
        best_by_cost=best_by_cost,
        greedy_by_cost=greedy_by_cost,
        leave_one_out=leave_one_out,
        files=dict(
            per_clip_score=args.per_clip_score,
            per_offset_score=args.per_offset_score_out,
            per_offset_metadata=args.per_offset_metadata_out,
            baseline_score=args.baseline_score,
            ten_clip_score=args.ten_clip_score,
            metadata=args.metadata,
        ),
    )
    ensure_parent(args.out)
    dump(report, args.out)
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
