"""Generate an interactive HTML viewer for skeleton sampling analysis."""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mmcv
import numpy as np
from mmengine.fileio import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyskl  # noqa: F401  # installs legacy OpenMMLab shims

from common import ensure_parent


COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]


@dataclass(frozen=True)
class ViewerPreset:
    """Paths and default choices for one analysis dataset."""

    name: str
    ann_file: str
    split: str
    label_map: str
    baseline_score: str
    ten_score: str
    kns_score: str
    per_offset_score: str
    per_offset_metadata: str
    best_offsets: tuple[int, ...]
    kns_label: str
    tracks: tuple[tuple[str, str], ...]


PRESETS = {
    "ntu": ViewerPreset(
        name="NTU60 XSub",
        ann_file="data/nturgbd/ntu60_hrnet.pkl",
        split="xsub_val",
        label_map="tools/data/label_map/nturgbd_120.txt",
        baseline_score="work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl",
        ten_score="work_dirs/modern/scores/posec3d_joint.pkl",
        kns_score="work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.pkl",
        per_offset_score="work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.pkl",
        per_offset_metadata="work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.metadata.pkl",
        best_offsets=(2, 3, 4, 7, 8),
        kns_label="E2 KNS-only",
        tracks=(
            ("E1 1-clip uniform", "work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.sampling.pkl"),
            ("E3 2-clip uniform", "work_dirs/modern/scores/posec3d_joint_e3_2clip_uniform.sampling.pkl"),
            ("E4 uniform+KNS-v1", "work_dirs/modern/scores/posec3d_joint_e4_uniform_kns.sampling.pkl"),
            ("E5 10-clip uniform", "work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.metadata.pkl"),
        ),
    ),
    "gym": ViewerPreset(
        name="FineGYM",
        ann_file="data/gym/gym_hrnet.pkl",
        split="val",
        label_map="tools/data/label_map/gym.txt",
        baseline_score="work_dirs/modern/scores/gym_joint_e1_1clip_uniform.pkl",
        ten_score="work_dirs/modern/scores/gym_joint_e5_10clip_uniform.pkl",
        kns_score="work_dirs/modern/scores/gym_joint_e4_uniform_kns.pkl",
        per_offset_score="work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.pkl",
        per_offset_metadata="work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.metadata.pkl",
        best_offsets=(2, 3, 4, 9),
        kns_label="F4 uniform+KNS-v1",
        tracks=(
            ("F1 1-clip uniform", "work_dirs/modern/scores/gym_joint_e1_1clip_uniform.sampling.pkl"),
            ("F3 2-clip uniform", "work_dirs/modern/scores/gym_joint_e3_2clip_uniform.sampling.pkl"),
            ("F4 uniform+KNS-v1", "work_dirs/modern/scores/gym_joint_e4_uniform_kns.sampling.pkl"),
            ("F4-v2 uniform+KNS-v2", "work_dirs/modern/scores/gym_joint_e4_uniform_kns_v2.sampling.pkl"),
            ("F5 10-clip uniform", "work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.metadata.pkl"),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Build a static HTML skeleton sampling viewer.")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True, help="Dataset preset to visualize.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--recommend-top-k", type=int, default=20, help="Number of recommended samples to embed.")
    parser.add_argument("--sample-id", nargs="*", default=None, help="Optional explicit sample ids to embed.")
    parser.add_argument("--max-persons", type=int, default=2, help="Maximum persons per sample to embed.")
    parser.add_argument("--score-precision", type=int, default=5, help="Decimal places for score-like arrays.")
    parser.add_argument("--coord-precision", type=int, default=2, help="Decimal places for keypoint coordinates.")
    return parser.parse_args()


def read_label_map(path: str) -> list[str]:
    """Read one label name per line."""

    label_path = Path(path)
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split_annotations(path: str, split_name: str) -> list[dict[str, Any]]:
    """Load annotations in the same order as the configured split."""

    data = mmcv.load(path)
    if not isinstance(data, dict) or "annotations" not in data:
        return data
    annotations = data["annotations"]
    if split_name:
        split = set(data["split"][split_name])
        identifier = "filename" if "filename" in annotations[0] else "frame_dir"
        annotations = [item for item in annotations if item[identifier] in split]
    return annotations


def sample_id_of(annotation: dict[str, Any], fallback: int) -> str:
    """Return the stable sample identifier used by sampling metadata."""

    return str(annotation.get("frame_dir", annotation.get("filename", fallback)))


def as_score_array(path: str) -> np.ndarray:
    """Load a score pickle as float32 array."""

    return np.asarray(load(path), dtype=np.float32)


def predict(scores: np.ndarray) -> np.ndarray:
    """Return top-1 prediction ids for a score array."""

    return np.asarray(scores).argmax(axis=-1)


def label_score(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return each sample's score assigned to its ground-truth label."""

    return scores[np.arange(len(labels)), labels]


def robust_normalize(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize a temporal signal with P5/P95 clipping."""

    values = values.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [5, 95])
    scale = hi - lo
    if scale <= eps:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (scale + eps), 0, 1).astype(np.float32)


def smooth(values: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a short moving average."""

    if kernel_size <= 1 or values.size <= 1:
        return values
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def pose_confidence(keypoint: np.ndarray, keypoint_score: np.ndarray | None, eps: float = 1e-6) -> np.ndarray:
    """Estimate per-frame pose reliability in the KNS-v1 style."""

    if keypoint_score is not None:
        score = keypoint_score.astype(np.float32)
        valid = score > eps
        denom = np.maximum(valid.sum(axis=(0, 2)), 1)
        return (score * valid).sum(axis=(0, 2)) / denom

    coords = keypoint[..., :2].astype(np.float32)
    valid = np.abs(coords).sum(axis=(-1, -2)) > eps
    denom = np.maximum(valid.sum(axis=0), 1)
    return valid.sum(axis=0).astype(np.float32) / denom


def motion_signals(keypoint: np.ndarray, keypoint_score: np.ndarray | None) -> dict[str, list[float]]:
    """Compute normalized velocity, acceleration and confidence signals."""

    total_frames = keypoint.shape[1]
    coords = keypoint[..., :2].astype(np.float32)
    confidence = pose_confidence(keypoint, keypoint_score)
    velocity = np.zeros(total_frames, dtype=np.float32)
    acceleration = np.zeros(total_frames, dtype=np.float32)
    if total_frames <= 1:
        return dict(velocity=velocity.tolist(), acceleration=acceleration.tolist(), confidence=confidence.tolist())

    diffs = coords[:, 1:] - coords[:, :-1]
    speed = np.linalg.norm(diffs, axis=-1)
    if keypoint_score is not None:
        pair_score = np.minimum(keypoint_score[:, 1:], keypoint_score[:, :-1]).astype(np.float32)
        denom = np.maximum((pair_score > 1e-6).sum(axis=(0, 2)), 1)
        velocity[1:] = (speed * pair_score).sum(axis=(0, 2)) / denom
    else:
        valid_pair = np.abs(coords[:, 1:]).sum(axis=(-1, -2)) > 1e-6
        valid_pair &= np.abs(coords[:, :-1]).sum(axis=(-1, -2)) > 1e-6
        denom = np.maximum(valid_pair.sum(axis=0), 1)
        velocity[1:] = (speed.sum(axis=2) * valid_pair).sum(axis=0) / denom

    if total_frames > 2:
        accel_vec = diffs[:, 1:] - diffs[:, :-1]
        accel = np.linalg.norm(accel_vec, axis=-1)
        if keypoint_score is not None:
            triple_score = np.minimum.reduce((keypoint_score[:, 2:], keypoint_score[:, 1:-1], keypoint_score[:, :-2]))
            denom = np.maximum((triple_score > 1e-6).sum(axis=(0, 2)), 1)
            acceleration[2:] = (accel * triple_score).sum(axis=(0, 2)) / denom
        else:
            valid_triple = np.abs(coords[:, 2:]).sum(axis=(-1, -2)) > 1e-6
            valid_triple &= np.abs(coords[:, 1:-1]).sum(axis=(-1, -2)) > 1e-6
            valid_triple &= np.abs(coords[:, :-2]).sum(axis=(-1, -2)) > 1e-6
            denom = np.maximum(valid_triple.sum(axis=0), 1)
            acceleration[2:] = (accel.sum(axis=2) * valid_triple).sum(axis=0) / denom

    velocity = smooth(robust_normalize(velocity) * confidence)
    acceleration = smooth(robust_normalize(acceleration) * confidence)
    return dict(
        velocity=round_list(velocity, 4),
        acceleration=round_list(acceleration, 4),
        confidence=round_list(confidence, 4),
    )


def round_list(values: np.ndarray, precision: int) -> list[float]:
    """Round an array for JSON output."""

    return np.round(values.astype(np.float32), precision).tolist()


def round_nested_keypoint(
    keypoint: np.ndarray,
    keypoint_score: np.ndarray | None,
    max_persons: int,
    precision: int,
) -> list[Any]:
    """Build compact ``person x frame x joint x (x, y, score)`` JSON data."""

    persons = min(max_persons, keypoint.shape[0])
    coords = keypoint[:persons, :, :, :2].astype(np.float32)
    if keypoint_score is None:
        scores = np.ones(coords.shape[:-1], dtype=np.float32)
    else:
        scores = keypoint_score[:persons].astype(np.float32)
    packed = np.concatenate([coords, scores[..., None]], axis=-1)
    return np.round(packed, precision).tolist()


def split_clip_frames(frame_inds: list[int], clip_len: int) -> list[list[int]]:
    """Split concatenated sampler frames into clip chunks."""

    return [frame_inds[start:start + clip_len] for start in range(0, len(frame_inds), clip_len)]


def kns_role_points(frames: list[int], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Tag KNS selected frames as velocity, acceleration, merged peak or fill."""

    peaks = meta.get("partition_peaks") or []
    if not peaks:
        return [dict(frame=int(frame), role="kns") for frame in frames]
    chunks = np.array_split(np.asarray(frames, dtype=np.int64), len(peaks))
    points: list[dict[str, Any]] = []
    for chunk, peak in zip(chunks, peaks):
        tv = int(peak.get("tv", -1))
        ta = int(peak.get("ta", -1))
        center = int(round((tv + ta) / 2))
        for frame in chunk:
            frame = int(frame)
            if abs(tv - ta) <= 2 and frame == center:
                role = "merged"
            elif frame == tv:
                role = "velocity"
            elif frame == ta:
                role = "acceleration"
            else:
                role = "fill"
            points.append(dict(frame=frame, role=role))
    return points


def make_track_row(name: str, role: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach compact diagnostics to one sampling track row."""

    ordered_points = []
    for order, point in enumerate(points):
        ordered = dict(point)
        ordered["order"] = order
        ordered_points.append(ordered)
    frames = [int(point["frame"]) for point in ordered_points]
    return dict(
        name=name,
        role=role,
        frames=ordered_points,
        count=len(frames),
        unique=len(set(frames)),
        start=frames[0] if frames else None,
        end=frames[-1] if frames else None,
    )


def metadata_track(name: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one sampling metadata record into viewer track rows."""

    if "offsets" in record:
        rows = []
        for offset in record["offsets"]:
            points = [
                dict(frame=int(frame), role=f"offset-{offset['offset_id']}")
                for frame in offset["frame_inds"]
            ]
            rows.append(make_track_row(f"{name} offset {offset['offset_id']}", "offset", points))
        return rows

    clip_len = int(record["clip_len"])
    clips = split_clip_frames([int(item) for item in record["frame_inds"]], clip_len)
    metas = record.get("kns_meta") or []
    rows = []
    for idx, frames in enumerate(clips):
        meta = metas[idx] if idx < len(metas) else {}
        sampler = meta.get("sampler", "uniform")
        row_name = f"{name} clip {idx}" if len(clips) > 1 else name
        if sampler.startswith("KNS"):
            points = kns_role_points(frames, meta)
            row_name = f"{row_name} {sampler}"
        else:
            points = [dict(frame=int(frame), role="uniform") for frame in frames]
        rows.append(make_track_row(row_name, sampler, points))
    return rows


def best_offset_track(name: str, offset_record: dict[str, Any], best_offsets: tuple[int, ...]) -> dict[str, Any]:
    """Build a combined track for the best offset subset."""

    frames = []
    for offset in offset_record["offsets"]:
        offset_id = int(offset["offset_id"])
        if offset_id in best_offsets:
            frames.extend(dict(frame=int(frame), role=f"best-{offset_id}") for frame in offset["frame_inds"])
    return make_track_row(name, "best-subset", frames)


def load_metadata_maps(preset: ViewerPreset) -> dict[str, dict[str, dict[str, Any]]]:
    """Load all available sampling metadata files as sample-id maps."""

    maps = {}
    for name, path in preset.tracks:
        if not Path(path).exists():
            continue
        records = load(path)
        maps[name] = {str(record["sample_id"]): record for record in records}
    return maps


def recommendation_rows(
    preset: ViewerPreset,
    labels: np.ndarray,
    sample_ids: list[str],
    explicit_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Rank samples and return score-derived arrays for later rendering."""

    baseline = as_score_array(preset.baseline_score)
    ten = as_score_array(preset.ten_score)
    kns = as_score_array(preset.kns_score)
    per_offset = as_score_array(preset.per_offset_score)
    subset = per_offset[:, preset.best_offsets, :].mean(axis=1)

    base_pred = predict(baseline)
    ten_pred = predict(ten)
    kns_pred = predict(kns)
    subset_pred = predict(subset)
    base_correct = base_pred == labels
    ten_correct = ten_pred == labels
    kns_correct = kns_pred == labels
    subset_correct = subset_pred == labels

    base_label_score = label_score(baseline, labels)
    ten_label_score = label_score(ten, labels)
    kns_label_score = label_score(kns, labels)
    subset_label_score = label_score(subset, labels)

    rows = []
    explicit_set = set(explicit_ids or [])
    for idx, sample_id in enumerate(sample_ids):
        if explicit_ids is not None and sample_id not in explicit_set:
            continue
        if not base_correct[idx] and ten_correct[idx] and not kns_correct[idx]:
            priority, reason = 0, "10clip_fix_kns_miss"
        elif not base_correct[idx] and subset_correct[idx] and not kns_correct[idx]:
            priority, reason = 1, "best_subset_fix_kns_miss"
        elif not base_correct[idx] and ten_correct[idx]:
            priority, reason = 2, "10clip_fix"
        elif base_correct[idx] and not kns_correct[idx]:
            priority, reason = 3, "kns_break"
        elif not base_correct[idx] and subset_correct[idx]:
            priority, reason = 4, "best_subset_fix"
        else:
            priority, reason = 5, "context"
        value = float(subset_label_score[idx] - kns_label_score[idx] + 0.25 * (ten_label_score[idx] - base_label_score[idx]))
        rows.append(dict(index=idx, sample_id=sample_id, priority=priority, reason=reason, value=value))

    if explicit_ids is not None:
        order = {sample_id: pos for pos, sample_id in enumerate(explicit_ids)}
        rows.sort(key=lambda row: order.get(row["sample_id"], len(order)))
    else:
        rows.sort(key=lambda row: (row["priority"], -row["value"], row["sample_id"]))

    arrays = dict(
        baseline=baseline,
        ten=ten,
        kns=kns,
        subset=subset,
        base_pred=base_pred,
        ten_pred=ten_pred,
        kns_pred=kns_pred,
        subset_pred=subset_pred,
        base_correct=base_correct,
        ten_correct=ten_correct,
        kns_correct=kns_correct,
        subset_correct=subset_correct,
        base_label_score=base_label_score,
        ten_label_score=ten_label_score,
        kns_label_score=kns_label_score,
        subset_label_score=subset_label_score,
    )
    return rows, arrays


def label_name(labels: list[str], label_id: int) -> str:
    """Return label text with fallback."""

    if 0 <= label_id < len(labels):
        return labels[label_id]
    return f"class {label_id}"


def prediction_payload(
    arrays: dict[str, np.ndarray],
    labels: list[str],
    idx: int,
    true_label: int,
    precision: int,
) -> dict[str, Any]:
    """Build prediction summary for one sample."""

    rows = {}
    specs = [
        ("baseline", "E1/F1 1-clip uniform", "base_pred", "base_correct", "base_label_score"),
        ("ten", "E5/F5 10-clip uniform", "ten_pred", "ten_correct", "ten_label_score"),
        ("kns", "KNS-v1", "kns_pred", "kns_correct", "kns_label_score"),
        ("subset", "best offset subset", "subset_pred", "subset_correct", "subset_label_score"),
    ]
    for key, title, pred_key, correct_key, score_key in specs:
        pred_id = int(arrays[pred_key][idx])
        rows[key] = dict(
            title=title,
            pred=pred_id,
            pred_name=label_name(labels, pred_id),
            correct=bool(arrays[correct_key][idx]),
            label_score=round(float(arrays[score_key][idx]), precision),
        )
    rows["truth"] = dict(label=true_label, name=label_name(labels, true_label))
    return rows


def build_tracks(
    sample_id: str,
    metadata_maps: dict[str, dict[str, dict[str, Any]]],
    best_offsets: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Build all sampling rows for one sample."""

    rows = []
    offset_record = None
    for name, record_map in metadata_maps.items():
        record = record_map.get(sample_id)
        if record is None:
            continue
        if "10-clip" in name and "offsets" in record:
            offset_record = record
        rows.extend(metadata_track(name, record))
    if offset_record is not None:
        rows.append(best_offset_track("best offset subset", offset_record, best_offsets))
    return rows


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Load files and build the self-contained viewer payload."""

    preset = PRESETS[args.preset]
    annotations = load_split_annotations(preset.ann_file, preset.split)
    offset_metadata = load(preset.per_offset_metadata)
    sample_ids = [str(record["sample_id"]) for record in offset_metadata]
    ann_by_id = {sample_id_of(annotation, idx): annotation for idx, annotation in enumerate(annotations)}
    labels = np.asarray([int(record["label"]) for record in offset_metadata], dtype=np.int64)
    label_names = read_label_map(preset.label_map)
    metadata_maps = load_metadata_maps(preset)
    rows, arrays = recommendation_rows(preset, labels, sample_ids, args.sample_id)
    rows = rows[:args.recommend_top_k if args.sample_id is None else len(rows)]

    samples = []
    missing = []
    for rank, row in enumerate(rows, start=1):
        idx = int(row["index"])
        sample_id = row["sample_id"]
        annotation = ann_by_id.get(sample_id)
        if annotation is None:
            missing.append(sample_id)
            continue
        keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
        keypoint_score = np.asarray(annotation["keypoint_score"], dtype=np.float32) if "keypoint_score" in annotation else None
        true_label = int(labels[idx])
        samples.append(
            dict(
                rank=rank,
                sample_id=sample_id,
                reason=row["reason"],
                priority=int(row["priority"]),
                value=round(float(row["value"]), 5),
                total_frames=int(annotation.get("total_frames", keypoint.shape[1])),
                img_shape=list(annotation.get("img_shape", (480, 640))),
                label=dict(id=true_label, name=label_name(label_names, true_label)),
                predictions=prediction_payload(arrays, label_names, idx, true_label, args.score_precision),
                keypoint=round_nested_keypoint(keypoint, keypoint_score, args.max_persons, args.coord_precision),
                signals=motion_signals(keypoint, keypoint_score),
                tracks=build_tracks(sample_id, metadata_maps, preset.best_offsets),
            )
        )

    if missing:
        print(f"warning: {len(missing)} requested sample ids were not found in annotations: {missing[:5]}", flush=True)

    return dict(
        preset=dict(
            key=args.preset,
            name=preset.name,
            best_offsets=list(preset.best_offsets),
            kns_label=preset.kns_label,
        ),
        edges=COCO_EDGES,
        samples=samples,
    )


def html_template(payload: dict[str, Any]) -> str:
    """Render the static HTML document."""

    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    title = f"Skeleton Sampling Viewer - {payload['preset']['name']}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f7f5;
  --panel: #ffffff;
  --ink: #1e2428;
  --muted: #667079;
  --line: #d7dbdf;
  --accent: #0b7285;
  --bad: #c92a2a;
  --good: #2b8a3e;
  --skeleton-height: clamp(310px, 36vh, 440px);
  --signal-height: clamp(112px, 12vh, 150px);
  --track-label-width: 230px;
  --signal-key-width: 178px;
  --track-height: 360px;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); overflow: auto; }}
header {{ height: 54px; padding: 8px 16px; border-bottom: 1px solid var(--line); background: var(--panel); display: flex; justify-content: space-between; gap: 16px; align-items: center; }}
h1 {{ margin: 0; font-size: 17px; font-weight: 650; }}
.sub {{ color: var(--muted); font-size: 12px; }}
main {{ min-height: calc(100vh - 54px); display: grid; grid-template-columns: 320px minmax(860px, 1fr); gap: 10px; padding: 10px; overflow: visible; align-items: start; }}
aside, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
aside {{ height: calc(100vh - 74px); min-height: 0; overflow: auto; position: sticky; top: 10px; }}
.sample {{ border-bottom: 1px solid var(--line); padding: 8px 10px; cursor: pointer; }}
.sample:hover, .sample.active {{ background: #eef7f8; }}
.sample-title {{ font-weight: 650; font-size: 13px; overflow-wrap: anywhere; }}
.sample-meta {{ margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.35; }}
.workspace {{ min-height: calc(100vh - 74px); overflow: visible; padding: 14px; display: grid; gap: 10px; align-content: start; }}
.topbar {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: start; }}
.preview-grid {{ display: grid; grid-template-columns: minmax(540px, 1fr) minmax(340px, 430px); gap: 14px; align-items: stretch; }}
.cards {{ display: grid; grid-template-columns: 1fr; grid-auto-rows: 1fr; gap: 8px; min-height: var(--skeleton-height); }}
.card {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 74px; }}
.card.good {{ border-color: #9bd3a6; background: #f0fbf2; }}
.card.bad {{ border-color: #f2b8b5; background: #fff5f5; }}
.card-title {{ font-size: 12px; color: var(--muted); }}
.card-main {{ margin-top: 3px; font-size: 13px; font-weight: 650; }}
.card-small {{ margin-top: 3px; color: var(--muted); font-size: 12px; }}
.controls {{ display: grid; grid-template-columns: auto 1fr auto auto auto; gap: 10px; align-items: center; }}
button {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 6px; padding: 6px 9px; cursor: pointer; }}
button:hover {{ border-color: var(--accent); }}
input[type=range] {{ width: 100%; }}
.signal-row {{ display: grid; grid-template-columns: var(--signal-key-width) minmax(0, 1fr); gap: 8px; align-items: stretch; }}
.signal-key {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 12px 14px; display: grid; align-content: center; gap: 16px; color: var(--ink); font-size: 12px; }}
canvas {{ width: 100%; border: 1px solid var(--line); border-radius: 8px; background: #fff; display: block; }}
#skeletonCanvas {{ height: var(--skeleton-height); }}
#signalCanvas {{ height: var(--signal-height); }}
#trackCanvas {{ min-height: 0; height: var(--track-height); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 9px; color: var(--muted); font-size: 12px; line-height: 1.2; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align: -1px; }}
@media (max-width: 980px) {{
  header {{ height: auto; }}
  main {{ height: auto; }}
  main {{ grid-template-columns: 1fr; }}
  .workspace {{ overflow: visible; }}
  .preview-grid {{ grid-template-columns: 1fr; }}
  .cards {{ min-height: auto; grid-auto-rows: auto; }}
  .signal-row {{ grid-template-columns: 1fr; }}
  .signal-key {{ grid-template-columns: repeat(3, max-content); gap: 12px; }}
  aside {{ height: auto; max-height: 360px; position: static; }}
  #skeletonCanvas {{ height: 320px; }}
  #signalCanvas {{ height: 160px; }}
  #trackCanvas {{ height: var(--track-height); }}
}}
</style>
</head>
<body>
<header>
  <div>
    <h1>{html.escape(title)}</h1>
    <div class="sub">Static report. Drag the time slider, play skeleton motion, and compare sampling tracks.</div>
  </div>
  <div class="sub">Best offsets: <span id="bestOffsets"></span></div>
</header>
<main>
  <aside id="sampleList"></aside>
  <section class="workspace">
    <div class="topbar">
      <div>
        <h2 id="sampleTitle" style="margin:0 0 4px 0;font-size:17px;"></h2>
        <div id="sampleSubtitle" class="sub"></div>
      </div>
    </div>
    <div class="preview-grid">
      <canvas id="skeletonCanvas" width="720" height="540"></canvas>
      <div class="cards" id="predictionCards"></div>
    </div>
    <div class="controls">
      <button id="playButton">Play</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0">
      <div class="sub" id="frameReadout"></div>
      <button id="prevButton">Prev</button>
      <button id="nextButton">Next</button>
    </div>
    <div class="signal-row">
      <div class="signal-key">
        <span><i class="dot" style="background:#228be6"></i>velocity</span>
        <span><i class="dot" style="background:#e03131"></i>acceleration</span>
        <span><i class="dot" style="background:#2f9e44"></i>confidence</span>
      </div>
      <canvas id="signalCanvas" width="900" height="260"></canvas>
    </div>
    <canvas id="trackCanvas" width="1200" height="430"></canvas>
    <div class="legend">
      <span><i class="dot" style="background:#228be6"></i>velocity</span>
      <span><i class="dot" style="background:#e03131"></i>acceleration</span>
      <span><i class="dot" style="background:#2f9e44"></i>confidence</span>
      <span><i class="dot" style="background:#495057"></i>uniform/fill</span>
      <span><i class="dot" style="background:#f08c00"></i>KNS velocity peak</span>
      <span><i class="dot" style="background:#9c36b5"></i>KNS acceleration peak</span>
      <span><i class="dot" style="background:#d6336c"></i>KNS merged peak</span>
      <span><i class="dot" style="background:#0b7285"></i>best subset</span>
      <span><i class="dot" style="background:#212529"></i>track start/end</span>
    </div>
  </section>
</main>
<script>
const DATA = {payload_json};
let sampleIndex = 0;
let frame = 0;
let playing = false;
let timer = null;

const colors = {{
  velocity: "#f08c00",
  acceleration: "#9c36b5",
  merged: "#d6336c",
  fill: "#495057",
  uniform: "#4dabf7",
  "best": "#0b7285",
  offset: "#adb5bd"
}};

function roleColor(role) {{
  if (role === "velocity") return colors.velocity;
  if (role === "acceleration") return colors.acceleration;
  if (role === "merged") return colors.merged;
  if (role === "fill") return colors.fill;
  if (role === "uniform") return colors.uniform;
  if (role.startsWith("best-")) return colors.best;
  if (role.startsWith("offset-")) return colors.uniform;
  return "#495057";
}}
function hexToRgba(hex, alpha) {{
  const clean = hex.replace("#", "");
  const value = Number.parseInt(clean, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
}}
function rolePriority(role) {{
  if (role === "merged") return 70;
  if (role === "acceleration") return 60;
  if (role === "velocity") return 50;
  if (role.startsWith("best-")) return 40;
  if (role === "fill") return 20;
  if (role === "uniform" || role.startsWith("offset-")) return 10;
  return 0;
}}
function aggregateTrackPoints(points) {{
  const frames = new Map();
  for (const point of points) {{
    const key = String(point.frame);
    const current = frames.get(key);
    if (!current) {{
      frames.set(key, {{
        frame: point.frame,
        role: point.role,
        count: 1,
        firstOrder: point.order,
        lastOrder: point.order
      }});
      continue;
    }}
    current.count += 1;
    current.firstOrder = Math.min(current.firstOrder, point.order);
    current.lastOrder = Math.max(current.lastOrder, point.order);
    if (rolePriority(point.role) > rolePriority(current.role)) {{
      current.role = point.role;
    }}
  }}
  const values = Array.from(frames.values()).sort((a, b) => a.frame - b.frame);
  const minCount = values.reduce((value, point) => Math.min(value, point.count), Number.POSITIVE_INFINITY);
  const maxCount = values.reduce((value, point) => Math.max(value, point.count), 1);
  return [values, minCount, maxCount];
}}

function sample() {{ return DATA.samples[sampleIndex]; }}
function clampFrame(value) {{
  const s = sample();
  return Math.max(0, Math.min(s.total_frames - 1, value));
}}
function resizeCanvas(canvas) {{
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return [ctx, rect.width, rect.height];
}}
function updateLayout() {{
  const s = sample();
  const root = document.documentElement;
  const rows = s.tracks || [];
  const trackNeeded = Math.max(320, rows.length * 20 + 38);
  root.style.setProperty("--track-height", `${{trackNeeded}}px`);
}}
function drawSkeleton() {{
  const canvas = document.getElementById("skeletonCanvas");
  const [ctx, w, h] = resizeCanvas(canvas);
  const s = sample();
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  const imgH = s.img_shape[0] || h;
  const imgW = s.img_shape[1] || w;
  const scale = Math.min((w - 28) / imgW, (h - 28) / imgH);
  const ox = (w - imgW * scale) / 2;
  const oy = (h - imgH * scale) / 2;
  ctx.strokeStyle = "#e9ecef";
  ctx.strokeRect(ox, oy, imgW * scale, imgH * scale);
  const persons = s.keypoint || [];
  for (let p = 0; p < persons.length; p++) {{
    const joints = persons[p][frame] || [];
    ctx.lineWidth = p === 0 ? 3 : 2;
    ctx.strokeStyle = p === 0 ? "#1c7ed6" : "#868e96";
    for (const edge of DATA.edges) {{
      const a = joints[edge[0]], b = joints[edge[1]];
      if (!a || !b || a[2] < 0.2 || b[2] < 0.2) continue;
      ctx.beginPath();
      ctx.moveTo(ox + a[0] * scale, oy + a[1] * scale);
      ctx.lineTo(ox + b[0] * scale, oy + b[1] * scale);
      ctx.stroke();
    }}
    ctx.fillStyle = p === 0 ? "#1864ab" : "#495057";
    for (const joint of joints) {{
      if (!joint || joint[2] < 0.2) continue;
      ctx.beginPath();
      ctx.arc(ox + joint[0] * scale, oy + joint[1] * scale, p === 0 ? 3.2 : 2.4, 0, Math.PI * 2);
      ctx.fill();
    }}
  }}
}}
function drawSignal() {{
  const canvas = document.getElementById("signalCanvas");
  const [ctx, w, h] = resizeCanvas(canvas);
  const s = sample();
  const padL = 44, padR = 16, padT = 14, padB = 28;
  const pw = w - padL - padR, ph = h - padT - padB;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#dee2e6";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = padT + ph * i / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
  }}
  function plot(values, color) {{
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {{
      const x = padL + (values.length <= 1 ? 0 : i / (values.length - 1)) * pw;
      const y = padT + (1 - Math.max(0, Math.min(1, values[i]))) * ph;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}
  plot(s.signals.velocity, "#228be6");
  plot(s.signals.acceleration, "#e03131");
  plot(s.signals.confidence, "#2f9e44");
  const x = padL + frame / Math.max(1, s.total_frames - 1) * pw;
  ctx.strokeStyle = "#212529";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, padT);
  ctx.lineTo(x, padT + ph);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#495057";
  ctx.font = "12px sans-serif";
  ctx.fillText("normalized motion signals", padL, h - 8);
}}
function drawTracks() {{
  const canvas = document.getElementById("trackCanvas");
  const [ctx, w, h] = resizeCanvas(canvas);
  const s = sample();
  const rows = s.tracks || [];
  const padL = w < 720 ? 160 : 230, padR = 18, padT = 14;
  const rowH = Math.max(19, Math.min(22, (h - padT - 12) / Math.max(1, rows.length)));
  const pointH = Math.max(11, rowH - 7);
  const pw = w - padL - padR;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.font = "11px sans-serif";
  function fittedText(text, x, y, maxWidth) {{
    let value = String(text);
    while (ctx.measureText(value).width > maxWidth && value.length > 6) {{
      value = value.slice(0, -5) + "...";
    }}
    ctx.fillText(value, x, y);
  }}
  for (let r = 0; r < rows.length; r++) {{
    const y = padT + r * rowH;
    ctx.fillStyle = "#495057";
    const countText = rows[r].count ? ` ${{rows[r].unique}}/${{rows[r].count}}` : "";
    fittedText(`${{rows[r].name}}${{countText}}`, 10, y + 4, padL - 18);
    ctx.strokeStyle = "#e9ecef";
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
    const [points, minCount, maxCount] = aggregateTrackPoints(rows[r].frames);
    for (const point of points) {{
      const x = padL + point.frame / Math.max(1, s.total_frames - 1) * pw;
      const hasCountContrast = maxCount > minCount;
      const normalizedCount = hasCountContrast ? (point.count - minCount) / Math.max(1, maxCount - minCount) : 1;
      const alpha = hasCountContrast ? 0.30 + normalizedCount * 0.70 : 1;
      ctx.fillStyle = hexToRgba(roleColor(point.role), alpha.toFixed(3));
      ctx.fillRect(x - 1.6, y - pointH / 2, 3.2, pointH);
      if (point.firstOrder === 0) {{
        ctx.fillStyle = "#212529";
        ctx.beginPath();
        ctx.moveTo(x, y - pointH / 2 - 4);
        ctx.lineTo(x - 4, y - pointH / 2 - 10);
        ctx.lineTo(x + 4, y - pointH / 2 - 10);
        ctx.closePath();
        ctx.fill();
      }}
      if (point.lastOrder === rows[r].count - 1) {{
        ctx.fillStyle = "#212529";
        ctx.beginPath();
        ctx.arc(x, y + pointH / 2 + 4, 2.3, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}
  }}
  const x = padL + frame / Math.max(1, s.total_frames - 1) * pw;
  ctx.strokeStyle = "#212529";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, 6);
  ctx.lineTo(x, Math.min(h - 8, padT + rows.length * rowH));
  ctx.stroke();
  ctx.setLineDash([]);
}}
function renderPredictions() {{
  const s = sample();
  const container = document.getElementById("predictionCards");
  const keys = ["baseline", "ten", "kns", "subset"];
  container.innerHTML = keys.map(key => {{
    const p = s.predictions[key];
    const cls = p.correct ? "good" : "bad";
    const mark = p.correct ? "correct" : "wrong";
    return `<div class="card ${{cls}}"><div class="card-title">${{p.title}}</div><div class="card-main">${{p.pred}} · ${{escapeHtml(p.pred_name)}}</div><div class="card-small">${{mark}}, label score ${{p.label_score}}</div></div>`;
  }}).join("");
}}
function escapeHtml(text) {{
  return String(text).replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#039;"}}[ch]));
}}
function renderSampleList() {{
  const list = document.getElementById("sampleList");
  list.innerHTML = DATA.samples.map((s, idx) => `
    <div class="sample ${{idx === sampleIndex ? "active" : ""}}" data-index="${{idx}}">
      <div class="sample-title">#${{s.rank}} ${{escapeHtml(s.sample_id)}}</div>
      <div class="sample-meta">${{escapeHtml(s.reason)}} · label ${{s.label.id}} ${{escapeHtml(s.label.name)}}</div>
    </div>`).join("");
  for (const item of list.querySelectorAll(".sample")) {{
    item.addEventListener("click", () => {{
      sampleIndex = Number(item.dataset.index);
      frame = 0;
      stop();
      renderAll();
    }});
  }}
}}
function renderAll() {{
  const s = sample();
  document.getElementById("bestOffsets").textContent = DATA.preset.best_offsets.join(", ");
  document.getElementById("sampleTitle").textContent = `${{s.sample_id}}`;
  document.getElementById("sampleSubtitle").textContent = `${{s.reason}} · label ${{s.label.id}} ${{s.label.name}} · value ${{s.value}}`;
  const slider = document.getElementById("frameSlider");
  slider.max = Math.max(0, s.total_frames - 1);
  slider.value = frame;
  document.getElementById("frameReadout").textContent = `frame ${{frame}} / ${{s.total_frames - 1}}`;
  renderSampleList();
  renderPredictions();
  updateLayout();
  drawSkeleton();
  drawSignal();
  drawTracks();
}}
function step() {{
  frame = clampFrame(frame + 1);
  if (frame >= sample().total_frames - 1) frame = 0;
  renderAll();
}}
function play() {{
  if (playing) return;
  playing = true;
  document.getElementById("playButton").textContent = "Pause";
  timer = setInterval(step, 90);
}}
function stop() {{
  playing = false;
  document.getElementById("playButton").textContent = "Play";
  if (timer) clearInterval(timer);
  timer = null;
}}
document.getElementById("playButton").addEventListener("click", () => playing ? stop() : play());
document.getElementById("prevButton").addEventListener("click", () => {{ frame = clampFrame(frame - 1); renderAll(); }});
document.getElementById("nextButton").addEventListener("click", () => {{ frame = clampFrame(frame + 1); renderAll(); }});
document.getElementById("frameSlider").addEventListener("input", event => {{ frame = clampFrame(Number(event.target.value)); renderAll(); }});
window.addEventListener("resize", renderAll);
if (!DATA.samples.length) {{
  document.getElementById("sampleList").innerHTML = "<div class='sample'>No samples embedded.</div>";
}} else {{
  renderAll();
}}
</script>
</body>
</html>
"""


def main() -> int:
    """Build and write a static viewer."""

    args = parse_args()
    payload = build_payload(args)
    ensure_parent(args.out)
    Path(args.out).write_text(html_template(payload), encoding="utf-8")
    print(f"wrote: {args.out}")
    print(f"embedded_samples: {len(payload['samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
