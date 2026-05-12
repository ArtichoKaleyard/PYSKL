# Modern NTU60 xsub Experiment Entrypoints

This directory contains the focused OpenMMLab 2.x / PyTorch 2.7 migration path
for the NTU60 xsub experiments.

The tentative experiment matrix and execution gates are recorded in
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).
Completed modern-branch metrics are recorded in [`RESULTS.md`](RESULTS.md).

## Environment

```bash
uv sync --group dev
uv run python scripts/check_env.py
```

## Assets

```bash
uv run python tools/modern/prepare_assets.py --check
uv run python tools/modern/prepare_assets.py --download
```

The asset script prepares:

- `data/nturgbd/ntu60_hrnet.pkl`
- `checkpoints/posec3d/slowonly_r50_ntu60_xsub/joint.pth`
- `checkpoints/posec3d/slowonly_r50_ntu60_xsub/limb.pth`
- `checkpoints/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.pth`

## A-D Official-Weight Evaluation

```bash
uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint.py \
  --out work_dirs/modern/scores/posec3d_joint.pkl

uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_limb.py \
  --out work_dirs/modern/scores/posec3d_limb.pkl

uv run python tools/modern/fuse_scores.py \
  --scores work_dirs/modern/scores/posec3d_joint.pkl work_dirs/modern/scores/posec3d_limb.pkl \
  --ann-file data/nturgbd/ntu60_hrnet.pkl \
  --split xsub_val \
  --out work_dirs/modern/scores/posec3d_joint_limb_fusion.json

uv run python tools/modern/test.py \
  configs/modern/msg3d/ntu60_xsub_hrnet_joint.py \
  --out work_dirs/modern/scores/msg3d_hrnet_joint.pkl \
  --eval top_k_accuracy
```

## E-G Training

```bash
uv run python tools/modern/train.py \
  configs/modern/posec3d/ntu60_xsub_joint_original_sampling.py \
  --validate

uv run python tools/modern/train.py \
  configs/modern/posec3d/ntu60_xsub_joint_custom_sampling.py \
  --validate

uv run python tools/modern/train.py \
  configs/modern/posec3d/ntu60_xsub_limb_custom_sampling.py \
  --validate
```

Use `--max-epochs 1` for a short smoke run, and omit it for the configured
training schedule.
