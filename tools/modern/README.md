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
- `data/gym/gym_hrnet.pkl`
- `checkpoints/posec3d/slowonly_r50_gym/joint.pth`

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

## E Group KNS Test-Time Sampling

```bash
uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint_1clip_uniform.py \
  --out work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl

uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint_1clip_kns.py \
  --out work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.pkl

uv run python tools/modern/export_sampling_metadata.py \
  configs/modern/posec3d/ntu60_xsub_joint_1clip_kns.py \
  --out work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.sampling.pkl

uv run python tools/modern/analyze_kns_scores.py \
  --score E1=work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl \
  --score E2=work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.pkl \
  --score E5=work_dirs/modern/scores/posec3d_joint.pkl \
  --metadata E1=work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.sampling.pkl \
  --metadata E2=work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.sampling.pkl \
  --metadata E5=work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.sampling.pkl \
  --out work_dirs/modern/scores/posec3d_joint_kns_analysis.json
```

Full E1-E5 commands and decision gates are in `EXPERIMENT_PLAN.md`. The older
training sampler configs remain available for later stages, but they are not
part of the first KNS pass.

## FineGYM Follow-Up

FineGYM uses mean class Top-1 as the primary comparison metric.

```bash
uv run python tools/modern/test.py \
  configs/modern/posec3d/gym_joint_2clip_uniform.py \
  --out work_dirs/modern/scores/gym_joint_e3_2clip_uniform.pkl

uv run python tools/modern/test.py \
  configs/modern/posec3d/gym_joint_uniform_kns.py \
  --out work_dirs/modern/scores/gym_joint_e4_uniform_kns.pkl
```

KNS-v2 configs are available for focused checks, but the current recorded
results do not support expanding v2 into limb/fusion.
