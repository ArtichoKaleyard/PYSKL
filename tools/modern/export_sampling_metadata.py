"""Export test-time sampler metadata without running model inference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mmcv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyskl  # noqa: F401  # registers PYSKL datasets and pipelines
from mmcv.utils import build_from_cfg
from mmengine.fileio import dump

from common import ensure_parent, load_config
from pyskl.datasets.builder import PIPELINES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Export frame indices selected by a modern test sampler.")
    parser.add_argument("config", help="Config file path.")
    parser.add_argument("--out", required=True, help="Output metadata pkl/json path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for smoke checks.")
    parser.add_argument("--progress-interval", type=int, default=30, help="Seconds between progress logs.")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    """Convert numpy scalar containers into dump-friendly Python values."""

    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def load_split_annotations(config_path: str) -> list[dict[str, Any]]:
    """Load raw annotations in the same order as the configured test split."""

    cfg = load_config(config_path)
    data_cfg = cfg.data.test
    data = mmcv.load(data_cfg.ann_file)
    if data_cfg.get("split"):
        split = set(data["split"][data_cfg.split])
        annotations = data["annotations"]
        identifier = "filename" if "filename" in annotations[0] else "frame_dir"
        annotations = [item for item in annotations if item[identifier] in split]
    else:
        annotations = data
    return annotations


def build_sampler(config_path: str):
    """Build the first transform from the configured test pipeline."""

    cfg = load_config(config_path)
    data_cfg = cfg.data.test
    if not data_cfg.pipeline:
        raise ValueError(f"No test pipeline found in {config_path}.")
    return build_from_cfg(data_cfg.pipeline[0], PIPELINES)


def main() -> int:
    """Export one metadata record per sample."""

    args = parse_args()
    annotations = load_split_annotations(args.config)
    sampler = build_sampler(args.config)
    records = []
    start_time = time.monotonic()
    last_log_time = start_time
    if args.max_samples is not None:
        annotations = annotations[:args.max_samples]
    for idx, annotation in enumerate(annotations):
        item = dict(annotation)
        item["start_index"] = 0
        item["modality"] = "Pose"
        item["test_mode"] = True
        item = sampler(item)
        records.append(
            dict(
                sample_id=item.get("frame_dir", item.get("filename", idx)),
                label=int(item["label"]),
                total_frames=int(item["total_frames"]),
                frame_inds=to_builtin(item["frame_inds"]),
                num_clips=int(item["num_clips"]),
                clip_len=int(item["clip_len"]),
                kns_meta=to_builtin(item.get("kns_meta", [])),
            )
        )
        now = time.monotonic()
        if args.progress_interval > 0 and now - last_log_time >= args.progress_interval:
            elapsed = now - start_time
            speed = len(records) / elapsed if elapsed > 0 else 0.0
            remaining = (len(annotations) - len(records)) / speed if speed > 0 else float("inf")
            print(
                f"progress: {len(records)}/{len(annotations)} "
                f"({len(records) / len(annotations) * 100:.2f}%), "
                f"{speed:.2f} samples/s, eta {remaining / 60:.1f} min",
                flush=True,
            )
            last_log_time = now

    ensure_parent(args.out)
    dump(records, args.out)
    print(f"dumped_samples: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
