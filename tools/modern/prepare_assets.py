"""Prepare and verify assets for the modern PYSKL experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class Asset:
    """Downloadable experiment asset."""

    name: str
    url: str
    path: Path


ASSETS = [
    Asset(
        name="ntu60_hrnet",
        url="https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl",
        path=Path("data/nturgbd/ntu60_hrnet.pkl"),
    ),
    Asset(
        name="posec3d_joint",
        url="http://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu60_xsub/joint.pth",
        path=Path("checkpoints/posec3d/slowonly_r50_ntu60_xsub/joint.pth"),
    ),
    Asset(
        name="posec3d_limb",
        url="http://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu60_xsub/limb.pth",
        path=Path("checkpoints/posec3d/slowonly_r50_ntu60_xsub/limb.pth"),
    ),
    Asset(
        name="msg3d_hrnet_joint",
        url="http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.pth",
        path=Path("checkpoints/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.pth"),
    ),
    Asset(
        name="gym_hrnet",
        url="https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl",
        path=Path("data/gym/gym_hrnet.pkl"),
    ),
    Asset(
        name="posec3d_gym_joint",
        url="http://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_gym/joint.pth",
        path=Path("checkpoints/posec3d/slowonly_r50_gym/joint.pth"),
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Prepare modern PYSKL experiment assets.")
    parser.add_argument("--root", default=".", help="Repository root containing data/ and checkpoints/.")
    parser.add_argument("--download", action="store_true", help="Download missing assets.")
    parser.add_argument("--check", action="store_true", help="Only check whether assets exist.")
    return parser.parse_args()


def download_asset(asset: Asset, root: Path) -> None:
    """Download one asset to its target path."""

    target = root / asset.path
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(asset.url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with target.open("wb") as file, tqdm(total=total, unit="B", unit_scale=True, desc=asset.name) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file.write(chunk)
            progress.update(len(chunk))


def main() -> int:
    """Check or download assets."""

    args = parse_args()
    root = Path(args.root).resolve()
    missing = []
    for asset in ASSETS:
        target = root / asset.path
        if target.exists() and target.stat().st_size > 0:
            print(f"[OK] {asset.name}: {target}")
            continue
        print(f"[MISSING] {asset.name}: {target}")
        print(f"          {asset.url}")
        missing.append(asset)

    if missing and args.download:
        for asset in missing:
            download_asset(asset, root)
        return main_check(root)

    return 1 if missing else 0


def main_check(root: Path) -> int:
    """Verify all assets after download."""

    failed = False
    for asset in ASSETS:
        target = root / asset.path
        if target.exists() and target.stat().st_size > 0:
            print(f"[OK] {asset.name}: {target}")
        else:
            print(f"[MISSING] {asset.name}: {target}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
