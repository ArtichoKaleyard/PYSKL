"""Modern single-process training entrypoint for targeted PYSKL experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from common import (
    build_model_from_config,
    build_split_dataloader,
    configure_threads,
    evaluate_and_dump,
    load_config,
    move_to_device,
    run_inference,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Train a PYSKL model with the modern stack.")
    parser.add_argument("config", help="Config file path.")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional validation score pkl path.")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from the legacy config schema."""

    cfg = cfg.copy()
    optimizer_type = cfg.pop("type")
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), **cfg)
    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), **cfg)
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), **cfg)
    raise KeyError(f"Unsupported optimizer type: {optimizer_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg, epochs: int):
    """Build the scheduler needed by the target configs."""

    lr_cfg = cfg.get("lr_config", {})
    if lr_cfg.get("policy") == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=lr_cfg.get("min_lr", 0),
        )
    return None


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg_text: str) -> None:
    """Save a checkpoint that mmengine can load later."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {"epoch": epoch, "config": cfg_text},
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def parse_loss(outputs: dict) -> torch.Tensor:
    """Extract loss tensor from PYSKL ``train_step`` output."""

    if "loss" not in outputs:
        raise KeyError(f"train_step output has no loss: {outputs.keys()}")
    return outputs["loss"]


def main() -> int:
    """Run training."""

    configure_threads()
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg.seed = args.seed
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.max_epochs or int(cfg.get("total_epochs", 1))

    device = torch.device(args.device)
    _, train_loader = build_split_dataloader(cfg, "train", shuffle=True)
    model = build_model_from_config(cfg, checkpoint=args.resume or cfg.get("load_from"), device=args.device)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg, epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", unit="batch")
        for batch in progress:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.train_step(batch, optimizer)
            loss = parse_loss(outputs)
            loss.backward()
            grad_clip = cfg.get("optimizer_config", {}).get("grad_clip")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip.get("max_norm", 40), grad_clip.get("norm_type", 2))
            optimizer.step()
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(work_dir / "latest.pth", model, optimizer, epoch, cfg.pretty_text)
        if cfg.get("checkpoint_config", {}).get("interval", 0) and epoch % cfg.checkpoint_config.interval == 0:
            save_checkpoint(work_dir / f"epoch_{epoch}.pth", model, optimizer, epoch, cfg.pretty_text)

    if args.validate:
        dataset, val_loader = build_split_dataloader(cfg, "val", shuffle=False)
        outputs = run_inference(model, val_loader, device)
        out = args.out or str(work_dir / f"val_scores_{time.strftime('%Y%m%d_%H%M%S')}.pkl")
        metrics = evaluate_and_dump(dataset, outputs, out, cfg.evaluation.get("metrics", ["top_k_accuracy"]))
        for key, value in metrics.items():
            print(f"{key}: {value:.04f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
