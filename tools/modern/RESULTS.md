# NTU60 xsub Modern Branch Results

本文档记录 `modern` 分支上已完成的关键实验结果。原始 score、metrics 和融合
JSON 位于 `work_dirs/modern/scores/`，该目录是本地实验产物目录，不进入 git。

## 2026-05-12 Official-Weight Evaluation

环境基线：

- Python 3.11
- CUDA 12.8
- PyTorch 2.7.1+cu128
- 数据：`data/nturgbd/ntu60_hrnet.pkl`
- split：`xsub_val`
- validation samples：16487

### Metrics

| 组别 | 模型 | 权重 | top-1 | top-5 | mean class |
| --- | --- | --- | ---: | ---: | ---: |
| A | PoseC3D joint | `checkpoints/posec3d/slowonly_r50_ntu60_xsub/joint.pth` | 0.9373 | 0.9964 | 0.9372 |
| B | PoseC3D limb | `checkpoints/posec3d/slowonly_r50_ntu60_xsub/limb.pth` | 0.9338 | 0.9946 | 0.9337 |
| D | MS-G3D HRNet joint | `checkpoints/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.pth` | 0.9264 | 0.9948 | - |

### PoseC3D Joint+Limb Fusion

| 组别 | 权重比例 | top-1 | top-5 | mean class |
| --- | --- | ---: | ---: | ---: |
| C | joint:limb = 1:1 | 0.9406 | 0.9961 | 0.9405 |
| C | joint:limb = 2:1 | 0.9406 | 0.9967 | 0.9405 |
| C | joint:limb = 1:2 | 0.9391 | 0.9960 | 0.9390 |

当前可作为后续采样实验基线的默认融合结果为 `1:1`。`2:1` 的 top-1 与
mean class 基本等同，但 top-5 略高；若后续报告只保留一个融合口径，优先使用
`1:1`，因为它最接近标准 joint+limb 等权融合。

### Official Comparison

官方结果来自本仓库 model zoo 文档：

- `configs/posec3d/README.md`：NTURGB+D XSub / SlowOnly-R50 /
  HRNet 2D Pose 行。
- `configs/msg3d/README.md`：NTURGB+D XSub / HRNet 2D Skeleton 行。

| 组别 | 指标口径 | 本轮 top-1 | 官方 top-1 | 差值 |
| --- | --- | ---: | ---: | ---: |
| A | PoseC3D joint | 93.73 | 93.70 | +0.03 |
| B | PoseC3D limb | 93.38 | 93.40 | -0.02 |
| C | PoseC3D joint+limb 1:1 | 94.06 | 94.10 | -0.04 |
| D | MS-G3D HRNet joint | 92.64 | 92.70 | -0.06 |

结论：当前 A-D 与官方记录没有明显冲突。差异均小于 0.1 个百分点，处于
不同运行栈、浮点实现、评估脚本和四舍五入口径下可接受的范围。PoseC3D
joint 略高于官方表格，limb、fusion 和 MS-G3D 略低于官方表格，但都没有出现
需要优先排查的数据 split、checkpoint 或 pipeline 错配迹象。

### Config Consistency

`configs/modern/` 通过 `_base_` 继承官方配置。对 A、B、D 展开配置后核对：

- `model` 与官方配置一致。
- `data.test` 与官方配置一致，包括 `ann_file`、`xsub_val` split 和完整
  `test_pipeline`。
- PoseC3D 仍使用官方测试设置：`UniformSampleFrames(clip_len=48,
  num_clips=10)`、`GeneratePoseTarget(double=True)`。
- MS-G3D 仍使用官方测试设置：`UniformSample(clip_len=100, num_clips=10)`。
- `load_from` 是 modern 配置新增项，只是把官方权重路径固定到本地
  `checkpoints/`。
- `work_dir` 是 modern 配置新增项，只改变本地产物目录。

存在的差异都属于运行层面：

| 项 | 官方配置 | modern 配置 / 本轮运行 | 是否影响指标语义 |
| --- | --- | --- | --- |
| PoseC3D `data.workers_per_gpu` | 4 | modern 文件为 0，本轮命令覆盖为 4 | 否，只影响加载/预处理吞吐 |
| PoseC3D `test_dataloader.videos_per_gpu` | 1 | 本轮命令覆盖为 2 | 否，eval 模式下只影响 batch 大小和速度 |
| MS-G3D `data.workers_per_gpu` | 2 | modern 文件为 0，本轮命令使用 0 | 否，只影响 dataloader 并行 |
| MS-G3D `test_dataloader.videos_per_gpu` | 1 | 本轮命令覆盖为 4 | 否，eval 模式下只影响 batch 大小和速度 |
| `persistent_workers` | 未显式设置 | modern 文件设为 `False` | 否，只影响 worker 生命周期 |

因此本轮结果可视为在相同模型、相同数据、相同测试 pipeline 和相同官方权重下的
现代运行栈复测；batch size 和 worker 数的调整是吞吐优化，不改变测试口径。

## Commands

PoseC3D joint 和 limb 使用 `videos_per_gpu=2, workers_per_gpu=4`。最初的
`videos_per_gpu=1` 会导致完整验证过慢；batch=4 反而吞吐下降，因此正式结果采用
batch=2。

```bash
uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint.py \
  --out work_dirs/modern/scores/posec3d_joint.pkl \
  --partial-out work_dirs/modern/scores/posec3d_joint.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4

uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_limb.py \
  --out work_dirs/modern/scores/posec3d_limb.pkl \
  --partial-out work_dirs/modern/scores/posec3d_limb.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4
```

MS-G3D 使用 `videos_per_gpu=4, workers_per_gpu=0`。batch=32 会顶满显存并
显著变慢，因此正式结果采用 batch=4。

```bash
uv run python tools/modern/test.py \
  configs/modern/msg3d/ntu60_xsub_hrnet_joint.py \
  --out work_dirs/modern/scores/msg3d_hrnet_joint.pkl \
  --partial-out work_dirs/modern/scores/msg3d_hrnet_joint.partial.pkl \
  --partial-interval 500 \
  --progress-interval 30 \
  --videos-per-gpu 4 \
  --workers-per-gpu 0 \
  --eval top_k_accuracy
```

融合命令：

```bash
uv run python tools/modern/fuse_scores.py \
  --scores work_dirs/modern/scores/posec3d_joint.pkl work_dirs/modern/scores/posec3d_limb.pkl \
  --ann-file data/nturgbd/ntu60_hrnet.pkl \
  --split xsub_val \
  --out work_dirs/modern/scores/posec3d_joint_limb_fusion.json
```

## Output Files

| 产物 | 路径 |
| --- | --- |
| A score | `work_dirs/modern/scores/posec3d_joint.pkl` |
| A metrics | `work_dirs/modern/scores/posec3d_joint.metrics.json` |
| B score | `work_dirs/modern/scores/posec3d_limb.pkl` |
| B metrics | `work_dirs/modern/scores/posec3d_limb.metrics.json` |
| D score | `work_dirs/modern/scores/msg3d_hrnet_joint.pkl` |
| D metrics | `work_dirs/modern/scores/msg3d_hrnet_joint.metrics.json` |
| C fusion | `work_dirs/modern/scores/posec3d_joint_limb_fusion.json` |

完整性检查结果：

- `posec3d_joint.pkl`：16487 samples
- `posec3d_limb.pkl`：16487 samples
- `msg3d_hrnet_joint.pkl`：16487 samples

## Notes

- partial score 文件只用于长任务进度检查，最终指标以非 partial 的 score 和
  metrics 文件为准。
- 本轮只验证官方权重测试和 PoseC3D 分数融合；E/F/G 训练实验尚未开始。
