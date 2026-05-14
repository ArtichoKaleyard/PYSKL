"""Analyze temporal weight patterns induced by uniform offset subsets.

This script is intentionally analysis-only: it does not train a model, modify
PoseC3D, or propose a new sampler.  It asks whether the non-uniform frame
weights created by averaging a subset of 10 uniform offsets are associated with
classification score, margin, fixes, and breaks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from mmengine.fileio import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyskl  # noqa: F401  # installs legacy OpenMMLab shims

from common import ensure_parent


@dataclass(frozen=True)
class DatasetSpec:
    """Paths and defaults for one offset weight analysis dataset."""

    key: str
    display_name: str
    primary_metric: str
    baseline_score: str
    ten_clip_score: str
    kns_score: str
    per_offset_score: str
    per_offset_metadata: str
    offset_analysis: str
    kns_metadata: str


DATASETS = {
    "finegym": DatasetSpec(
        key="finegym",
        display_name="FineGYM",
        primary_metric="mean_class_accuracy",
        baseline_score="work_dirs/modern/scores/gym_joint_e1_1clip_uniform.pkl",
        ten_clip_score="work_dirs/modern/scores/gym_joint_e5_10clip_uniform.pkl",
        kns_score="work_dirs/modern/scores/gym_joint_e4_uniform_kns.pkl",
        per_offset_score="work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.pkl",
        per_offset_metadata="work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.metadata.pkl",
        offset_analysis="work_dirs/modern/scores/gym_joint_10clip_offsets_analysis.json",
        kns_metadata="work_dirs/modern/scores/gym_joint_e4_uniform_kns.sampling.pkl",
    ),
    "ntu60": DatasetSpec(
        key="ntu60",
        display_name="NTU60 XSub",
        primary_metric="top1",
        baseline_score="work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl",
        ten_clip_score="work_dirs/modern/scores/posec3d_joint.pkl",
        kns_score="work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.pkl",
        per_offset_score="work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.pkl",
        per_offset_metadata="work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.metadata.pkl",
        offset_analysis="work_dirs/modern/scores/posec3d_joint_10clip_offsets_analysis.json",
        kns_metadata="work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.sampling.pkl",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Analyze temporal weight patterns of offset subsets.")
    parser.add_argument("--out-dir", default="work_dirs/modern/offset_analysis", help="Output directory.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["finegym", "ntu60"])
    parser.add_argument("--max-cost", type=int, default=4, help="Maximum offset subset size to enumerate.")
    parser.add_argument("--cluster-alpha", type=float, default=0.5, help="Mean + alpha * std cluster threshold.")
    parser.add_argument(
        "--sample-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample, per-subset compact metrics CSV.",
    )
    return parser.parse_args()


def as_array(path: str) -> np.ndarray:
    """Load a pickle score file as float32 array."""

    return np.asarray(load(path), dtype=np.float32)


def topk_correct(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Return per-sample top-k correctness."""

    if k >= scores.shape[1]:
        topk = np.argsort(scores, axis=1)
    else:
        topk = np.argpartition(scores, -k, axis=1)[:, -k:]
    return (topk == labels[:, None]).any(axis=1)


def mean_class_accuracy(correct: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean class accuracy from per-sample correctness."""

    values = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.any():
            values.append(float(correct[mask].mean()))
    return float(np.mean(values)) if values else 0.0


def label_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the score assigned to each sample's ground-truth label."""

    return scores[np.arange(len(labels)), labels]


def margins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return label score minus the best non-label score."""

    masked = scores.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    return label_scores(scores, labels) - masked.max(axis=1)


def stable_ab_split(sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Split samples into deterministic A/B halves by sample id hash."""

    split_a = []
    for sample_id in sample_ids:
        digest = hashlib.sha1(sample_id.encode("utf-8")).digest()
        split_a.append((digest[0] & 1) == 0)
    mask_a = np.asarray(split_a, dtype=bool)
    return mask_a, ~mask_a


def score_metrics_for_mask(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute classification metrics on a boolean sample subset."""

    if not mask.any():
        return dict(top1=0.0, top5=0.0, mean_class=0.0, label_score_mean=0.0, margin_mean=0.0)
    local_scores = scores[mask]
    local_labels = labels[mask]
    top1 = topk_correct(local_scores, local_labels, 1)
    top5 = topk_correct(local_scores, local_labels, 5)
    return dict(
        top1=float(top1.mean()),
        top5=float(top5.mean()),
        mean_class=float(mean_class_accuracy(top1, local_labels)),
        label_score_mean=float(label_scores(local_scores, local_labels).mean()),
        margin_mean=float(margins(local_scores, local_labels).mean()),
    )


def rank_average(values: np.ndarray) -> np.ndarray:
    """Return average ranks with tie handling for Spearman correlation."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x_values: Iterable[float], y_values: Iterable[float]) -> float | None:
    """Compute Spearman correlation without requiring SciPy."""

    x = np.asarray(list(x_values), dtype=np.float64)
    y = np.asarray(list(y_values), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = rank_average(x)
    ry = rank_average(y)
    denom = float(np.std(rx) * np.std(ry))
    if denom <= 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summary_stats(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a list of finite values."""

    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return dict(count=0, mean=None, median=None, p25=None, p75=None)
    return dict(
        count=int(array.size),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        p25=float(np.percentile(array, 25)),
        p75=float(np.percentile(array, 75)),
    )


def load_best_offsets(path: str, max_cost: int) -> dict[int, tuple[int, ...]]:
    """Load best offsets by cost from the existing offset analysis JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    output = {}
    for row in data["best_by_cost"]:
        cost = int(row["cost"])
        if cost <= max_cost:
            output[cost] = tuple(int(offset) for offset in row["offsets"])
    return output


def offset_count_matrix(record: dict[str, Any]) -> np.ndarray:
    """Return ``num_offsets x total_frames`` count matrix for one sample."""

    total_frames = int(record["total_frames"])
    matrix = np.zeros((len(record["offsets"]), total_frames), dtype=np.float32)
    for offset in record["offsets"]:
        offset_id = int(offset["offset_id"])
        for frame in offset["frame_inds"]:
            frame_id = int(frame)
            if 0 <= frame_id < total_frames:
                matrix[offset_id, frame_id] += 1.0
    return matrix


def weight_distribution(counts: np.ndarray) -> np.ndarray:
    """Normalize frame counts to a probability-like temporal weight vector."""

    total = float(counts.sum())
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float32)
    return (counts / total).astype(np.float32)


def continuous_clusters(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive continuous clusters from a boolean frame mask."""

    clusters = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            clusters.append((start, idx - 1))
            start = None
    if start is not None:
        clusters.append((start, len(mask) - 1))
    return clusters


def cluster_metrics(weights: np.ndarray, mode: str, alpha: float) -> dict[str, Any]:
    """Compute high-weight cluster metrics for one temporal weight vector."""

    total_frames = len(weights)
    if total_frames == 0 or float(weights.sum()) <= 0:
        return dict(
            cluster_count=0,
            largest_cluster_mass=0.0,
            largest_cluster_width=0,
            largest_cluster_width_norm=0.0,
            cluster_effective_number=0.0,
            cluster_span=0.0,
            phase_coverage_count=0,
            is_single_peak=0,
            is_multi_phase=0,
            cluster_centers_normalized=[],
        )

    if mode == "top20":
        top_k = max(1, int(math.ceil(total_frames * 0.2)))
        high_indices = np.argsort(weights)[-top_k:]
        mask = np.zeros(total_frames, dtype=bool)
        mask[high_indices] = True
    elif mode == "threshold":
        threshold = float(weights.mean() + alpha * weights.std())
        mask = weights >= threshold
        if not mask.any():
            mask[int(np.argmax(weights))] = True
    else:
        raise ValueError(f"Unknown cluster mode: {mode}")

    clusters = continuous_clusters(mask)
    masses = []
    widths = []
    centers = []
    phases = set()
    denom = max(total_frames - 1, 1)
    for start, end in clusters:
        cluster_weights = weights[start:end + 1]
        mass = float(cluster_weights.sum())
        width = int(end - start + 1)
        center = float((start + end) / 2.0 / denom)
        masses.append(mass)
        widths.append(width)
        centers.append(center)
        phases.add(min(2, int(center * 3)))
    largest_idx = int(np.argmax(masses)) if masses else 0
    mass_array = np.asarray(masses, dtype=np.float64)
    effective = float(1.0 / np.sum(mass_array * mass_array)) if mass_array.size and np.sum(mass_array) > 0 else 0.0
    span = float((max(end for _, end in clusters) - min(start for start, _ in clusters)) / denom) if clusters else 0.0
    return dict(
        cluster_count=int(len(clusters)),
        largest_cluster_mass=float(masses[largest_idx]) if masses else 0.0,
        largest_cluster_width=int(widths[largest_idx]) if widths else 0,
        largest_cluster_width_norm=float(widths[largest_idx] / total_frames) if widths else 0.0,
        cluster_effective_number=effective,
        cluster_span=span,
        phase_coverage_count=int(len(phases)),
        is_single_peak=int(len(clusters) == 1),
        is_multi_phase=int(len(phases) >= 2),
        cluster_centers_normalized=centers,
    )


def weight_metrics(weights: np.ndarray, alpha: float) -> dict[str, Any]:
    """Compute concentration and high-weight cluster metrics."""

    total_frames = len(weights)
    nonzero = weights[weights > 0]
    sum_sq = float(np.sum(weights * weights))
    n_eff = float(1.0 / sum_sq) if sum_sq > 0 else 0.0
    if total_frames <= 1:
        concentration = 0.0
    else:
        concentration = float((sum_sq - 1.0 / total_frames) / (1.0 - 1.0 / total_frames))
    top = cluster_metrics(weights, "top20", alpha)
    threshold = cluster_metrics(weights, "threshold", alpha)
    return dict(
        unique_frame_count=int(nonzero.size),
        effective_frame_count=n_eff,
        max_frame_weight=float(nonzero.max()) if nonzero.size else 0.0,
        concentration=concentration,
        cluster_count=top["cluster_count"],
        largest_cluster_mass=top["largest_cluster_mass"],
        largest_cluster_width=top["largest_cluster_width"],
        largest_cluster_width_norm=top["largest_cluster_width_norm"],
        cluster_effective_number=top["cluster_effective_number"],
        cluster_span=top["cluster_span"],
        phase_coverage_count=top["phase_coverage_count"],
        is_single_peak=top["is_single_peak"],
        is_multi_phase=top["is_multi_phase"],
        cluster_centers_normalized=top["cluster_centers_normalized"],
        threshold_cluster_count=threshold["cluster_count"],
        threshold_largest_cluster_mass=threshold["largest_cluster_mass"],
        threshold_cluster_span=threshold["cluster_span"],
        threshold_phase_coverage_count=threshold["phase_coverage_count"],
    )


def subset_scores(per_offset_scores: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    """Average per-offset probabilities for one subset."""

    return per_offset_scores[:, offsets, :].mean(axis=1)


def metric_means(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    """Average numeric weight metrics for a subset."""

    output = {}
    for key in keys:
        output[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
    return output


def append_sample_csv(
    writer: csv.DictWriter,
    sample_ids: list[str],
    labels: np.ndarray,
    subset_name: str,
    offset_cost: int,
    weight_rows: list[dict[str, Any]],
    score_rows: dict[str, np.ndarray],
) -> None:
    """Write compact per-sample metrics for one subset."""

    for idx, sample_id in enumerate(sample_ids):
        weight = weight_rows[idx]
        writer.writerow(
            dict(
                sample_id=sample_id,
                label=int(labels[idx]),
                subset=subset_name,
                offset_cost=offset_cost,
                correct=int(score_rows["correct"][idx]),
                label_score=float(score_rows["label_score"][idx]),
                margin=float(score_rows["margin"][idx]),
                unique_frame_count=weight["unique_frame_count"],
                effective_frame_count=weight["effective_frame_count"],
                max_frame_weight=weight["max_frame_weight"],
                concentration=weight["concentration"],
                cluster_count=weight["cluster_count"],
                largest_cluster_mass=weight["largest_cluster_mass"],
                largest_cluster_width=weight["largest_cluster_width"],
                cluster_effective_number=weight["cluster_effective_number"],
                cluster_span=weight["cluster_span"],
                phase_coverage_count=weight["phase_coverage_count"],
                is_single_peak=weight["is_single_peak"],
                is_multi_phase=weight["is_multi_phase"],
            )
        )


def kns_weight_by_sample(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute a KNS clip weight metric for comparison with best subsets."""

    output = {}
    for record in records:
        sample_id = str(record["sample_id"])
        total_frames = int(record["total_frames"])
        clip_len = int(record["clip_len"])
        frame_inds = [int(frame) for frame in record["frame_inds"]]
        metas = record.get("kns_meta") or []
        kns_clip_idx = 0
        for idx, meta in enumerate(metas):
            if str(meta.get("sampler", "")).startswith("KNS"):
                kns_clip_idx = idx
                break
        start = kns_clip_idx * clip_len
        inds = frame_inds[start:start + clip_len]
        counts = np.zeros(total_frames, dtype=np.float32)
        for frame in inds:
            if 0 <= frame < total_frames:
                counts[frame] += 1.0
        output[sample_id] = weight_metrics(weight_distribution(counts), alpha=0.5)
    return output


def compare_groups(values: list[dict[str, Any]], mask_a: np.ndarray, mask_b: np.ndarray, keys: list[str]) -> dict[str, Any]:
    """Compare weight metric distributions between two sample groups."""

    output = {}
    for key in keys:
        a = np.asarray([values[idx][key] for idx in np.where(mask_a)[0]], dtype=np.float64)
        b = np.asarray([values[idx][key] for idx in np.where(mask_b)[0]], dtype=np.float64)
        output[key] = dict(
            group_a_mean=float(np.mean(a)) if a.size else None,
            group_b_mean=float(np.mean(b)) if b.size else None,
            diff=float(np.mean(a) - np.mean(b)) if a.size and b.size else None,
        )
    return output


def parse_subset_name(name: str) -> tuple[int, ...]:
    """Parse a dash-joined subset name back into offset ids."""

    return tuple(int(item) for item in name.split("-") if item != "")


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Compute Jaccard overlap between two offset subsets."""

    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 0.0


def split_stability(subset_rows: list[dict[str, Any]], max_cost: int, primary_metric: str) -> list[dict[str, Any]]:
    """Select best subsets on full/A/B splits and measure their agreement."""

    primary = "mean_class" if primary_metric == "mean_class_accuracy" else primary_metric
    split_a_metric = f"split_a_{primary}"
    split_b_metric = f"split_b_{primary}"
    output = []
    for cost in range(1, max_cost + 1):
        rows = [row for row in subset_rows if int(row["offset_cost"]) == cost]
        full_best = max(rows, key=lambda row: float(row[primary]))
        a_best = max(rows, key=lambda row: float(row[split_a_metric]))
        b_best = max(rows, key=lambda row: float(row[split_b_metric]))
        full_offsets = parse_subset_name(full_best["subset"])
        a_offsets = parse_subset_name(a_best["subset"])
        b_offsets = parse_subset_name(b_best["subset"])
        output.append(
            dict(
                cost=cost,
                primary_metric=primary,
                full_best=list(full_offsets),
                split_a_best=list(a_offsets),
                split_b_best=list(b_offsets),
                a_b_jaccard=jaccard(a_offsets, b_offsets),
                full_a_jaccard=jaccard(full_offsets, a_offsets),
                full_b_jaccard=jaccard(full_offsets, b_offsets),
                full_primary=float(full_best[primary]),
                split_a_best_on_a=float(a_best[split_a_metric]),
                split_a_best_on_b=float(a_best[split_b_metric]),
                split_b_best_on_b=float(b_best[split_b_metric]),
                split_b_best_on_a=float(b_best[split_a_metric]),
                full_best_on_a=float(full_best[split_a_metric]),
                full_best_on_b=float(full_best[split_b_metric]),
            )
        )
    return output


def analyze_dataset(spec: DatasetSpec, out_dir: Path, max_cost: int, alpha: float, write_sample_csv: bool) -> dict[str, Any]:
    """Run the full offset weight association analysis for one dataset."""

    print(f"[{spec.key}] loading scores and metadata", flush=True)
    per_offset_scores = as_array(spec.per_offset_score)
    baseline_scores = as_array(spec.baseline_score)
    ten_scores = as_array(spec.ten_clip_score)
    kns_scores = as_array(spec.kns_score)
    metadata = load(spec.per_offset_metadata)
    labels = np.asarray([int(row["label"]) for row in metadata], dtype=np.int64)
    sample_ids = [str(row["sample_id"]) for row in metadata]
    best_offsets_by_cost = load_best_offsets(spec.offset_analysis, max_cost)
    num_offsets = per_offset_scores.shape[1]
    subsets = [tuple(combo) for cost in range(1, max_cost + 1) for combo in itertools.combinations(range(num_offsets), cost)]
    split_a_mask, split_b_mask = stable_ab_split(sample_ids)

    baseline_correct = topk_correct(baseline_scores, labels, 1)
    ten_correct = topk_correct(ten_scores, labels, 1)
    kns_correct = topk_correct(kns_scores, labels, 1)
    f10 = np.logical_and(~baseline_correct, ten_correct)
    base_label = label_scores(baseline_scores, labels)
    ten_label = label_scores(ten_scores, labels)
    kns_label = label_scores(kns_scores, labels)

    count_matrices = [offset_count_matrix(record) for record in metadata]
    subset_csv_path = out_dir / f"offset_subset_metrics_{spec.key}.csv"
    sample_csv_path = out_dir / f"sample_level_weight_metrics_{spec.key}.csv"
    fix_json_path = out_dir / f"fix_break_cluster_analysis_{spec.key}.json"
    corr_json_path = out_dir / f"sample_level_correlations_{spec.key}.json"
    ensure_parent(subset_csv_path)

    weight_keys = [
        "unique_frame_count",
        "effective_frame_count",
        "max_frame_weight",
        "concentration",
        "cluster_count",
        "largest_cluster_mass",
        "largest_cluster_width_norm",
        "cluster_effective_number",
        "cluster_span",
        "phase_coverage_count",
        "is_single_peak",
        "is_multi_phase",
        "threshold_cluster_count",
        "threshold_largest_cluster_mass",
        "threshold_cluster_span",
        "threshold_phase_coverage_count",
    ]
    subset_rows: list[dict[str, Any]] = []
    sample_corr_buffers = {
        "label_score": [[] for _ in sample_ids],
        "margin": [[] for _ in sample_ids],
        "concentration": [[] for _ in sample_ids],
        "cluster_span": [[] for _ in sample_ids],
        "cluster_effective_number": [[] for _ in sample_ids],
        "largest_cluster_mass": [[] for _ in sample_ids],
    }
    best_subset_payloads: dict[int, dict[str, Any]] = {}

    sample_writer = None
    sample_handle = None
    if write_sample_csv:
        sample_handle = sample_csv_path.open("w", newline="", encoding="utf-8")
        sample_writer = csv.DictWriter(
            sample_handle,
            fieldnames=[
                "sample_id",
                "label",
                "subset",
                "offset_cost",
                "correct",
                "label_score",
                "margin",
                "unique_frame_count",
                "effective_frame_count",
                "max_frame_weight",
                "concentration",
                "cluster_count",
                "largest_cluster_mass",
                "largest_cluster_width",
                "cluster_effective_number",
                "cluster_span",
                "phase_coverage_count",
                "is_single_peak",
                "is_multi_phase",
            ],
        )
        sample_writer.writeheader()

    try:
        for subset_idx, offsets in enumerate(subsets, start=1):
            if subset_idx % 25 == 0:
                print(f"[{spec.key}] subset {subset_idx}/{len(subsets)}", flush=True)
            subset_name = "-".join(str(offset) for offset in offsets)
            cost = len(offsets)
            scores = subset_scores(per_offset_scores, offsets)
            split_a_metrics = score_metrics_for_mask(scores, labels, split_a_mask)
            split_b_metrics = score_metrics_for_mask(scores, labels, split_b_mask)
            correct = topk_correct(scores, labels, 1)
            top5 = topk_correct(scores, labels, 5)
            label_score = label_scores(scores, labels)
            margin = margins(scores, labels)
            fixes = np.logical_and(~baseline_correct, correct)
            breaks = np.logical_and(baseline_correct, ~correct)
            overlap = float(np.logical_and(fixes, f10).sum() / max(int(f10.sum()), 1))

            weight_rows = []
            for sample_idx, matrix in enumerate(count_matrices):
                counts = matrix[list(offsets)].sum(axis=0)
                metrics = weight_metrics(weight_distribution(counts), alpha)
                weight_rows.append(metrics)
                sample_corr_buffers["label_score"][sample_idx].append(float(label_score[sample_idx]))
                sample_corr_buffers["margin"][sample_idx].append(float(margin[sample_idx]))
                for key in ("concentration", "cluster_span", "cluster_effective_number", "largest_cluster_mass"):
                    sample_corr_buffers[key][sample_idx].append(float(metrics[key]))

            if sample_writer is not None:
                append_sample_csv(
                    sample_writer,
                    sample_ids,
                    labels,
                    subset_name,
                    cost,
                    weight_rows,
                    dict(correct=correct, label_score=label_score, margin=margin),
                )

            row = dict(
                dataset=spec.key,
                subset=subset_name,
                offset_cost=cost,
                top1=float(correct.mean()),
                top5=float(top5.mean()),
                mean_class=float(mean_class_accuracy(correct, labels)),
                label_score_mean=float(label_score.mean()),
                margin_mean=float(margin.mean()),
                fixes=int(fixes.sum()),
                breaks=int(breaks.sum()),
                net_fixes=int(fixes.sum() - breaks.sum()),
                overlap_with_10clip=overlap,
                fix_mean_concentration=float(np.mean([weight_rows[idx]["concentration"] for idx in np.where(fixes)[0]]))
                if fixes.any()
                else None,
                break_mean_concentration=float(np.mean([weight_rows[idx]["concentration"] for idx in np.where(breaks)[0]]))
                if breaks.any()
                else None,
                split_a_top1=split_a_metrics["top1"],
                split_b_top1=split_b_metrics["top1"],
                split_a_mean_class=split_a_metrics["mean_class"],
                split_b_mean_class=split_b_metrics["mean_class"],
                split_a_label_score_mean=split_a_metrics["label_score_mean"],
                split_b_label_score_mean=split_b_metrics["label_score_mean"],
                split_a_margin_mean=split_a_metrics["margin_mean"],
                split_b_margin_mean=split_b_metrics["margin_mean"],
            )
            row.update(metric_means(weight_rows, weight_keys))
            subset_rows.append(row)

            if offsets in best_offsets_by_cost.values():
                best_subset_payloads[cost] = dict(
                    offsets=offsets,
                    scores=scores,
                    correct=correct,
                    label_score=label_score,
                    margin=margin,
                    weight_rows=weight_rows,
                    fixes=fixes,
                    breaks=breaks,
                )
    finally:
        if sample_handle is not None:
            sample_handle.close()

    with subset_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(subset_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset_rows)

    metric_names = [
        "mean_concentration",
        "mean_effective_frame_count",
        "mean_cluster_count",
        "mean_largest_cluster_mass",
        "mean_cluster_span",
        "mean_cluster_effective_number",
        "mean_phase_coverage_count",
        "mean_is_single_peak",
        "mean_is_multi_phase",
    ]
    performance_names = ["top1", "top5", "mean_class", "label_score_mean", "margin_mean", "fixes", "breaks", "net_fixes"]
    subset_correlations = {}
    for cost in range(1, max_cost + 1):
        cost_rows = [row for row in subset_rows if row["offset_cost"] == cost]
        subset_correlations[str(cost)] = {
            metric: {perf: spearman([row[metric] for row in cost_rows], [row[perf] for row in cost_rows]) for perf in performance_names}
            for metric in metric_names
        }

    sample_correlations = {}
    for metric in ("concentration", "cluster_span", "cluster_effective_number", "largest_cluster_mass"):
        label_corrs = []
        margin_corrs = []
        for idx in range(len(sample_ids)):
            label_corrs.append(spearman(sample_corr_buffers[metric][idx], sample_corr_buffers["label_score"][idx]))
            margin_corrs.append(spearman(sample_corr_buffers[metric][idx], sample_corr_buffers["margin"][idx]))
        sample_correlations[metric] = dict(
            label_score=summary_stats(label_corrs),
            margin=summary_stats(margin_corrs),
        )

    best_cost = max(best_offsets_by_cost)
    best_payload = best_subset_payloads[best_cost]
    best_correct = best_payload["correct"]
    best_fix = np.logical_and(~baseline_correct, best_correct)
    best_break = np.logical_and(baseline_correct, ~best_correct)
    best_kns_miss = np.logical_and(best_fix, ~kns_correct)
    fix_break_analysis = dict(
        dataset=spec.key,
        f10_count=int(f10.sum()),
        best_cost=best_cost,
        best_offsets=list(best_payload["offsets"]),
        best_fixes=int(best_fix.sum()),
        best_breaks=int(best_break.sum()),
        best_overlap_with_10clip=float(np.logical_and(best_fix, f10).sum() / max(int(f10.sum()), 1)),
        subset_correlations=subset_correlations,
        sample_correlations=sample_correlations,
        split_a_count=int(split_a_mask.sum()),
        split_b_count=int(split_b_mask.sum()),
        split_stability=split_stability(subset_rows, max_cost, spec.primary_metric),
        fix_vs_break_weight_metrics=compare_groups(
            best_payload["weight_rows"],
            best_fix,
            best_break,
            ["concentration", "effective_frame_count", "cluster_count", "largest_cluster_mass", "cluster_span", "cluster_effective_number"],
        ),
    )

    kns_records = load(spec.kns_metadata)
    kns_weights = kns_weight_by_sample(kns_records)
    shared_indices = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in kns_weights]
    if shared_indices:
        selected = np.asarray(shared_indices, dtype=np.int64)
        mask = np.logical_and(f10[selected], ~kns_correct[selected])
        if mask.any():
            best_rows = [best_payload["weight_rows"][idx] for idx in selected[mask]]
            kns_rows = [kns_weights[sample_ids[idx]] for idx in selected[mask]]
            fix_break_analysis["kns_vs_best_on_f10_kns_miss"] = {
                key: dict(
                    best_mean=float(np.mean([row[key] for row in best_rows])),
                    kns_mean=float(np.mean([row[key] for row in kns_rows])),
                )
                for key in ["concentration", "effective_frame_count", "cluster_count", "largest_cluster_mass", "cluster_span"]
            }

    Path(fix_json_path).write_text(json.dumps(fix_break_analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(corr_json_path).write_text(json.dumps(sample_correlations, indent=2, ensure_ascii=False), encoding="utf-8")

    typical = typical_samples(
        spec,
        sample_ids,
        labels,
        baseline_scores,
        ten_scores,
        kns_scores,
        best_payload,
        f10,
        kns_correct,
        count_matrices,
        alpha,
    )
    return dict(
        spec=spec,
        subset_rows=subset_rows,
        fix_break_analysis=fix_break_analysis,
        typical=typical,
        files=dict(
            subset_csv=str(subset_csv_path),
            sample_csv=str(sample_csv_path) if write_sample_csv or sample_csv_path.exists() else None,
            fix_json=str(fix_json_path),
            corr_json=str(corr_json_path),
        ),
    )


def compact_distribution(weights: np.ndarray) -> list[dict[str, float | int]]:
    """Return non-zero frame weights for JSON examples."""

    return [dict(frame=int(idx), weight=float(value)) for idx, value in enumerate(weights) if value > 0]


def prediction_summary(scores: np.ndarray, labels: np.ndarray, idx: int) -> dict[str, Any]:
    """Build a compact prediction summary for one sample."""

    pred = int(np.argmax(scores[idx]))
    return dict(
        pred=pred,
        correct=bool(pred == int(labels[idx])),
        label_score=float(scores[idx, labels[idx]]),
        margin=float(margins(scores[idx:idx + 1], labels[idx:idx + 1])[0]),
    )


def pick_first(mask: np.ndarray, used: set[int]) -> int | None:
    """Pick the first unused sample index from a mask."""

    for idx in np.where(mask)[0]:
        if int(idx) not in used:
            used.add(int(idx))
            return int(idx)
    return None


def typical_samples(
    spec: DatasetSpec,
    sample_ids: list[str],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    ten_scores: np.ndarray,
    kns_scores: np.ndarray,
    best_payload: dict[str, Any],
    f10: np.ndarray,
    kns_correct: np.ndarray,
    count_matrices: list[np.ndarray],
    alpha: float,
) -> list[dict[str, Any]]:
    """Select a few representative sample records for viewer/report follow-up."""

    best_correct = best_payload["correct"]
    best_weights = best_payload["weight_rows"]
    high_conc = np.asarray([row["concentration"] for row in best_weights]) >= np.percentile(
        [row["concentration"] for row in best_weights],
        75,
    )
    multi_phase = np.asarray([row["is_multi_phase"] for row in best_weights], dtype=bool)
    single_peak = np.asarray([row["is_single_peak"] for row in best_weights], dtype=bool)
    baseline_correct = topk_correct(baseline_scores, labels, 1)
    used: set[int] = set()
    masks = [
        ("10clip_fix_kns_wrong_best_correct", np.logical_and.reduce((f10, ~kns_correct, best_correct))),
        ("baseline_wrong_best_correct_high_cluster", np.logical_and.reduce((~baseline_correct, best_correct, high_conc))),
        ("baseline_correct_high_cluster_break", np.logical_and.reduce((baseline_correct, ~best_correct, high_conc))),
        ("multi_phase_correct", np.logical_and(multi_phase, best_correct)),
        ("single_peak_wrong", np.logical_and(single_peak, ~best_correct)),
    ]
    output = []
    for reason, mask in masks:
        idx = pick_first(mask, used)
        if idx is None:
            continue
        counts = count_matrices[idx][list(best_payload["offsets"])].sum(axis=0)
        weights = weight_distribution(counts)
        output.append(
            dict(
                dataset=spec.key,
                reason=reason,
                sample_id=sample_ids[idx],
                label=int(labels[idx]),
                baseline=prediction_summary(baseline_scores, labels, idx),
                ten_clip=prediction_summary(ten_scores, labels, idx),
                kns=prediction_summary(kns_scores, labels, idx),
                best_subset=prediction_summary(best_payload["scores"], labels, idx),
                best_offsets=list(best_payload["offsets"]),
                frame_weight_distribution=compact_distribution(weights),
                cluster_metrics=weight_metrics(weights, alpha),
                sampled_frame_indices=[
                    dict(offset=int(offset), frame_inds=np.where(count_matrices[idx][offset] > 0)[0].astype(int).tolist())
                    for offset in best_payload["offsets"]
                ],
            )
        )
    return output


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Write a markdown interpretation summary for the generated tables."""

    lines = [
        "# Offset Subset Temporal Weight Association",
        "",
        "本分析只验证 offset 子集形成的时间权重模式是否与分类收益存在关联；不训练模型，",
        "不修改 PoseC3D，也不把结果表述为新采样策略。",
        "",
        "## Outputs",
        "",
    ]
    for result in results:
        spec = result["spec"]
        lines.extend(
            [
                f"### {spec.display_name}",
                "",
                f"- subset metrics: `{result['files']['subset_csv']}`",
                f"- sample-level metrics: `{result['files']['sample_csv']}`",
                f"- fix/break cluster analysis: `{result['files']['fix_json']}`",
                f"- sample-level correlations: `{result['files']['corr_json']}`",
                "",
            ]
        )

    lines.extend(["## Key Questions", ""])
    for result in results:
        spec = result["spec"]
        rows = result["subset_rows"]
        fix = result["fix_break_analysis"]
        primary = "mean_class" if spec.primary_metric == "mean_class_accuracy" else spec.primary_metric
        best_row = max((row for row in rows if row["offset_cost"] == fix["best_cost"]), key=lambda row: row[primary])
        corrs = fix["subset_correlations"][str(fix["best_cost"])]
        conc_corr = corrs["mean_concentration"][primary]
        eff_corr = corrs["mean_effective_frame_count"][primary]
        span_corr = corrs["mean_cluster_span"][primary]
        mass_corr = corrs["mean_largest_cluster_mass"][primary]
        sample_conc = fix["sample_correlations"]["concentration"]["label_score"]
        kns_compare = fix.get("kns_vs_best_on_f10_kns_miss", {})
        best_split = next(row for row in fix["split_stability"] if row["cost"] == fix["best_cost"])
        lines.extend(
            [
                f"### {spec.display_name}",
                "",
                f"1. offset subset 确实形成非均匀时间权重。best-{fix['best_cost']} "
                f"`{fix['best_offsets']}` 的平均 concentration 为 "
                f"{best_row['mean_concentration']:.4f}，平均有效帧数为 "
                f"{best_row['mean_effective_frame_count']:.2f}。",
                f"2. subset 级别上，best-cost 内 concentration 与 `{primary}` 的 Spearman 为 "
                f"{conc_corr}；N_eff 与 `{primary}` 的 Spearman 为 {eff_corr}；cluster span 为 {span_corr}；"
                f"largest cluster mass 为 {mass_corr}。",
                f"3. best subset fixes={fix['best_fixes']}，breaks={fix['best_breaks']}，"
                f"overlap_with_10clip={fix['best_overlap_with_10clip']:.4f}。",
                f"4. 样本级 concentration-label_score 相关分布："
                f"median={sample_conc['median']}，p25={sample_conc['p25']}，p75={sample_conc['p75']}。",
                f"5. A/B split stability: split A n={fix['split_a_count']}，split B n={fix['split_b_count']}；"
                f"best-cost 下 A-best={best_split['split_a_best']}，B-best={best_split['split_b_best']}，"
                f"Jaccard={best_split['a_b_jaccard']:.4f}。",
            ]
        )
        if kns_compare:
            lines.append(
                "6. 在 F10 且 KNS 错的样本上，best subset 与 KNS 的权重模式差异见 "
                f"`kns_vs_best_on_f10_kns_miss`；例如 concentration "
                f"best={kns_compare['concentration']['best_mean']:.4f}, "
                f"KNS={kns_compare['concentration']['kns_mean']:.4f}。"
            )
        lines.extend(
            [
                "",
                "| cost | full-best | A-best | B-best | A/B Jaccard | full/A Jaccard | full/B Jaccard |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for split_row in fix["split_stability"]:
            lines.append(
                f"| {split_row['cost']} | `{split_row['full_best']}` | "
                f"`{split_row['split_a_best']}` | `{split_row['split_b_best']}` | "
                f"{split_row['a_b_jaccard']:.4f} | {split_row['full_a_jaccard']:.4f} | "
                f"{split_row['full_b_jaccard']:.4f} |"
            )
        lines.extend(
            [
                "",
                "解释边界：这些数值只说明关联强弱。若相关性弱，当前证据不支持继续从时间权重",
                "聚类角度设计采样策略；若相关性强，也只能作为后续 label-free temporal weighting",
                "研究假设的依据。",
                "",
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run all requested analyses."""

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    typical = []
    for key in args.datasets:
        result = analyze_dataset(DATASETS[key], out_dir, args.max_cost, args.cluster_alpha, args.sample_csv)
        results.append(result)
        typical.extend(result["typical"])
    (out_dir / "typical_samples.json").write_text(json.dumps(typical, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(results, out_dir)
    print(f"wrote: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
