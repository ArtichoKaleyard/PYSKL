"""Modern single-process evaluation and score export entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine.fileio import dump

from common import (
    build_model_from_config,
    build_split_dataloader,
    configure_threads,
    evaluate_and_dump,
    load_config,
    resolve_checkpoint,
    run_inference,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a PYSKL model with the modern stack.")
    parser.add_argument("config", help="Config file path.")
    parser.add_argument("-C", "--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument("--out", required=True, help="Output score pkl path.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", nargs="+", default=["top_k_accuracy", "mean_class_accuracy"])
    parser.add_argument("--progress-interval", type=int, default=20, help="Seconds between progress logs.")
    parser.add_argument("--partial-out", default=None, help="Optional partial score pkl path for progress inspection.")
    parser.add_argument("--partial-interval", type=int, default=50, help="Batches between partial score dumps.")
    parser.add_argument("--videos-per-gpu", type=int, default=None, help="Override test batch size.")
    parser.add_argument("--workers-per-gpu", type=int, default=None, help="Override test dataloader workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for smoke or throughput checks.")
    parser.add_argument("--skip-eval", action="store_true", help="Dump scores without metric evaluation.")
    return parser.parse_args()


def main() -> int:
    """Run evaluation."""

    configure_threads()
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg.seed = args.seed
    if args.videos_per_gpu is not None:
        cfg.data.test_dataloader.videos_per_gpu = args.videos_per_gpu
    if args.workers_per_gpu is not None:
        cfg.data.test_dataloader.workers_per_gpu = args.workers_per_gpu
    checkpoint = resolve_checkpoint(args.checkpoint, cfg)
    if checkpoint is None:
        raise ValueError("A checkpoint is required. Pass --checkpoint or set load_from in the config.")

    device = torch.device(args.device)
    dataset, dataloader = build_split_dataloader(cfg, "test", shuffle=False)
    model = build_model_from_config(cfg, checkpoint=checkpoint, device=args.device)
    outputs = run_inference(
        model,
        dataloader,
        device,
        progress_interval=args.progress_interval,
        partial_out=args.partial_out,
        partial_interval=args.partial_interval,
        max_batches=args.max_batches,
    )
    if args.skip_eval:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dump(outputs, args.out)
        print(f"dumped_samples: {len(outputs)}")
    else:
        metrics = evaluate_and_dump(dataset, outputs, Path(args.out), args.eval)
        for key, value in metrics.items():
            print(f"{key}: {value:.04f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
