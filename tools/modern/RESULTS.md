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

## 2026-05-14 KNS-v1 Test-Time Sampling

本轮按 `tools/modern/EXPERIMENT_PLAN.md` 将 E 组收缩为 test-time sampling：
不训练、不改 PoseC3D backbone、不改 heatmap 生成逻辑，只替换 `PoseDecode`
之前的 temporal sampler。E5 复用 A 组已完成的 10-clip uniform joint score。

统一设置：

- 模型：PoseC3D SlowOnly-R50 joint。
- 权重：`checkpoints/posec3d/slowonly_r50_ntu60_xsub/joint.pth`。
- 数据：`data/nturgbd/ntu60_hrnet.pkl`。
- split：`xsub_val`，16487 samples。
- test-time heatmap：保持官方 `GeneratePoseTarget(double=True)`。

### Metrics

| 组别 | 采样方式 | clip 成本 | top-1 | top-5 | mean class |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | 1-clip uniform | 1x | 0.9358 | 0.9964 | 0.9357 |
| E2 | 1-clip KNS-v1 | 1x | 0.9344 | 0.9959 | 0.9343 |
| E3 | 2-clip uniform | 2x | 0.9364 | 0.9963 | 0.9363 |
| E4 | uniform + KNS-v1 | 2x | 0.9369 | 0.9960 | 0.9367 |
| E5 | 10-clip uniform | 10x | 0.9373 | 0.9964 | 0.9372 |

### Migration From E1

| 目标组别 | Both Correct | Target Fixes | Target Breaks | Both Wrong |
| --- | ---: | ---: | ---: | ---: |
| E2 | 15261 | 145 | 168 | 913 |
| E3 | 15359 | 79 | 70 | 979 |
| E4 | 15362 | 84 | 67 | 974 |
| E5 | 15339 | 115 | 90 | 943 |

### 10-Clip Gain Overlap

以 `F10 = {E1 错, E5 对}` 为 10-clip 有效收益集合，本轮 `|F10| = 115`。

| 目标组别 | target fixes | shared with E5 | E5 gain coverage |
| --- | ---: | ---: | ---: |
| E2 | 145 | 71 | 0.6174 |
| E3 | 79 | 66 | 0.5739 |
| E4 | 84 | 53 | 0.4609 |

### KNS Interpretation

当前结果不支持“单独 1-clip KNS 优于 1-clip uniform”：E2 比 E1 低约
0.14 个百分点，且 E2 的 Target Breaks 多于 Target Fixes。E4 比同成本 E3 高约
0.05 个百分点，说明 KNS 作为第二视图有轻微互补收益；但 E4 仍比 10-clip E5 低约
0.05 个百分点。

按计划中的成功标准，本轮更接近“中等成功”的下沿：KNS 单独不足以替代 uniform，
但与 uniform 组合后略优于 2-clip uniform。后续若继续推进，应优先分析 E2/E4
的 breaks，重点检查 pose 抖动、低置信度样本和短视频分区重复采样是否误导峰值。

### Implementation Note

第一次 E2 正式运行在短视频样本处卡住；原因是 KNS 的分区补点逻辑要求每个分区
补齐 3 个不同帧，但当分区长度小于 3 时没有足够未占用位置。已修正为：优先使用
未覆盖区间中点；分区太短时允许重复帧补齐。这与 PYSKL 原始 uniform sampler
对短视频使用重复/取模补齐的语义一致。

## 2026-05-14 FineGYM And KNS-v2 Follow-Up

按后续判断，FineGYM 只跑 joint 官方权重的 E1/E3/E4/E5，不再跑单视图 KNS；
同时补一个轻量 KNS-v2：每个粗分区固定保留 uniform 中心点，剩余两个采样位
分别给速度峰值和加速度峰值。v2 只新增 sampler 和独立配置，不覆盖 KNS-v1。

### FineGYM Metrics

FineGYM 官方表以 mean class Top-1 为主，因此本节优先看 mean class。

| 组别 | 采样方式 | clip 成本 | top-1 | top-5 | mean class |
| --- | --- | ---: | ---: | ---: | ---: |
| F1 | 1-clip uniform | 1x | 0.9514 | 0.9977 | 0.9291 |
| F3 | 2-clip uniform | 2x | 0.9556 | 0.9974 | 0.9337 |
| F4 | uniform + KNS-v1 | 2x | 0.9508 | 0.9968 | 0.9242 |
| F4-v2 | uniform + KNS-v2 | 2x | 0.9486 | 0.9950 | 0.9203 |
| F5 | 10-clip uniform | 10x | 0.9575 | 0.9977 | 0.9381 |

FineGYM 没有放大 KNS-v1 的价值。F4-v1 比 F3 低 0.95 个 mean-class 百分点，
F4-v2 又低于 F4-v1，说明当前 KNS 峰值视图在 FineGYM joint 流上是负贡献。

### FineGYM Migration From F1

| 目标组别 | Both Correct | Target Fixes | Target Breaks | Both Wrong |
| --- | ---: | ---: | ---: | ---: |
| F3 | 8083 | 60 | 24 | 354 |
| F4 | 8045 | 57 | 62 | 357 |
| F4-v2 | 8029 | 54 | 78 | 360 |
| F5 | 8078 | 81 | 29 | 333 |

以 `F10 = {F1 错, F5 对}` 为 10-clip 有效收益集合，本轮 `|F10| = 81`。
F3 覆盖其中 52 个，F4-v1 覆盖 37 个，F4-v2 覆盖 29 个。KNS 视图不仅没有
接近 10-clip 收益，反而低于普通 2-clip uniform。

### KNS-v2 NTU60 Check

| 组别 | 采样方式 | clip 成本 | top-1 | top-5 | mean class |
| --- | --- | ---: | ---: | ---: | ---: |
| E2-v2 | 1-clip KNS-v2 | 1x | 0.9295 | 0.9955 | 0.9294 |
| E4-v2 | uniform + KNS-v2 | 2x | 0.9344 | 0.9962 | 0.9343 |

KNS-v2 的“uniform 中心点 + 速度峰值 + 加速度峰值”没有修复 v1 的稳定性问题：
NTU60 上 E2-v2 明显低于 E2-v1，E4-v2 也低于 E3 和 E4-v1。这个轻量 v2
分支暂时不值得继续扩展到 limb/fusion。

### Updated Decision

当前证据不支持继续沿 KNS-v1/v2 直接加数据集或加融合。更合理的下一步是先做
失败样本诊断：确认峰值是否被局部抖动、动作方向相反样本或短片段重复采样误导。
若继续设计 v3，应优先考虑门控或置信度/峰值质量过滤，而不是再增加无约束峰值视图。

## 2026-05-14 10-Clip Offset Analysis

本节研究 10-clip uniform 中哪些时间 offset 真的有用。实现上通过
`tools/modern/test.py --average-clips none` 导出未平均的 per-view logits，再由
`tools/modern/analyze_clip_offsets.py` 复原每个时间 offset 的贡献。

注意：PoseC3D 测试 pipeline 使用 `GeneratePoseTarget(double=True)`，因此原始导出
不是 10 个 view，而是 20 个 view：10 个时间 offset 加对应的水平翻转 view。
分析脚本先对同一时间 offset 的两个 view 做概率平均，再比较时间 offset 子集。
两个数据集上，脚本重算的 all-offset 指标与既有 10-clip score 完全一致。
同时导出按时间 offset 聚合后的 score，形状分别为 NTU60 `(16487, 10, 60)`、
FineGYM `(8521, 10, 99)`；对应 metadata 保存 `sample_id`、`label`、`offset_id`
和每个 offset 的 `frame_inds`。

### NTU60 Offset Metrics

| offset | mean center | top-1 | mean class | fixes vs E1 | breaks vs E1 | E5 gain coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.491 | 0.9358 | 0.9357 | 9 | 9 | 0.0435 |
| 1 | 0.499 | 0.9348 | 0.9347 | 129 | 146 | 0.7043 |
| 2 | 0.500 | 0.9359 | 0.9358 | 140 | 138 | 0.8261 |
| 3 | 0.499 | 0.9363 | 0.9361 | 156 | 149 | 0.8783 |
| 4 | 0.492 | 0.9369 | 0.9368 | 156 | 138 | 0.8261 |
| 5 | 0.506 | 0.9359 | 0.9358 | 136 | 135 | 0.7391 |
| 6 | 0.500 | 0.9333 | 0.9332 | 115 | 157 | 0.6348 |
| 7 | 0.490 | 0.9367 | 0.9366 | 156 | 142 | 0.7913 |
| 8 | 0.490 | 0.9359 | 0.9358 | 139 | 138 | 0.7565 |
| 9 | 0.502 | 0.9359 | 0.9358 | 160 | 159 | 0.7913 |

| cost | best offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[4]` | 0.9369 | 0.9368 |
| 2 | `[4, 8]` | 0.9377 | 0.9376 |
| 3 | `[2, 4, 7]` | 0.9381 | 0.9380 |
| 4 | `[2, 4, 5, 7]` | 0.9384 | 0.9383 |
| 5 | `[2, 3, 4, 7, 8]` | 0.9386 | 0.9385 |
| 10 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | 0.9373 | 0.9372 |

Greedy selection 按 top-1 逐步加入 offset：

| cost | greedy offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[4]` | 0.9369 | 0.9368 |
| 2 | `[4, 8]` | 0.9377 | 0.9376 |
| 3 | `[4, 8, 0]` | 0.9381 | 0.9380 |
| 4 | `[4, 8, 0, 5]` | 0.9381 | 0.9380 |
| 5 | `[4, 8, 0, 5, 9]` | 0.9379 | 0.9378 |
| 10 | `[4, 8, 0, 5, 9, 2, 7, 3, 1, 6]` | 0.9373 | 0.9372 |

NTU60 上并不是所有 offset 都提供正收益。单 offset 最好的是 offset 4，已经接近
10-clip；最佳 5-offset 子集 `[2, 3, 4, 7, 8]` 反而比完整 10-clip 高约
0.13 个 top-1 百分点。leave-one-out 也显示 offset 6 最可疑：移除 offset 6 后
top-1 提高约 0.055 个百分点。

### FineGYM Offset Metrics

| offset | mean center | top-1 | mean class | fixes vs F1 | breaks vs F1 | F5 gain coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.464 | 0.9519 | 0.9295 | 9 | 5 | 0.0617 |
| 1 | 0.479 | 0.9509 | 0.9284 | 67 | 71 | 0.6049 |
| 2 | 0.492 | 0.9518 | 0.9318 | 83 | 80 | 0.7778 |
| 3 | 0.503 | 0.9507 | 0.9299 | 84 | 90 | 0.7284 |
| 4 | 0.508 | 0.9514 | 0.9302 | 86 | 86 | 0.7531 |
| 5 | 0.517 | 0.9539 | 0.9341 | 93 | 72 | 0.7531 |
| 6 | 0.517 | 0.9504 | 0.9300 | 80 | 89 | 0.6790 |
| 7 | 0.508 | 0.9514 | 0.9311 | 87 | 87 | 0.7407 |
| 8 | 0.500 | 0.9518 | 0.9309 | 87 | 84 | 0.7160 |
| 9 | 0.493 | 0.9526 | 0.9330 | 75 | 65 | 0.6790 |

| cost | best offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[5]` | 0.9539 | 0.9341 |
| 2 | `[7, 9]` | 0.9556 | 0.9376 |
| 3 | `[5, 7, 9]` | 0.9575 | 0.9387 |
| 4 | `[2, 3, 4, 9]` | 0.9578 | 0.9398 |
| 5 | `[0, 2, 4, 8, 9]` | 0.9572 | 0.9395 |
| 10 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | 0.9575 | 0.9381 |

Greedy selection 按 mean class 逐步加入 offset：

| cost | greedy offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[5]` | 0.9539 | 0.9341 |
| 2 | `[5, 8]` | 0.9562 | 0.9365 |
| 3 | `[5, 8, 7]` | 0.9572 | 0.9381 |
| 4 | `[5, 8, 7, 2]` | 0.9573 | 0.9386 |
| 5 | `[5, 8, 7, 2, 6]` | 0.9575 | 0.9384 |
| 10 | `[5, 8, 7, 2, 6, 3, 4, 9, 1, 0]` | 0.9575 | 0.9381 |

FineGYM 上 offset 价值更集中。按 mean class，offset 5 单独就是最强单视图；
最佳 4-offset 子集 `[2, 3, 4, 9]` 达到 0.9398 mean class，比完整 10-clip 高约
0.17 个百分点。leave-one-out 中移除 offset 0 后 mean class 反而提升约 0.030
个百分点，而移除 offset 2/4/9 会明显下降，说明 10-clip 内部存在可裁剪的弱视图。

### Offset Decision

这轮结果支持继续研究“10-clip 里的有效 offset 选择”，而不是再扩展 KNS-v1/v2。
两个数据集都出现小子集优于完整 10-clip 的现象，且 FineGYM 的集中度更明显。
下一步可以把固定子集作为低成本 test-time policy 验证，例如 NTU 的
`[2, 3, 4, 7, 8]` 与 FineGYM 的 `[2, 3, 4, 9]`，再检查这些 offset 对
E1/F1 fixes 和 breaks 的样本类别分布。

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

10-clip offset 分析先导出不做 clip 平均的 raw view logits，再运行分析脚本：

```bash
uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint_10clip_uniform.py \
  --out work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_clip.pkl \
  --average-clips none \
  --skip-eval \
  --partial-out work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_clip.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4

uv run python tools/modern/analyze_clip_offsets.py \
  --per-clip-score work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_clip.pkl \
  --baseline-score work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl \
  --ten-clip-score work_dirs/modern/scores/posec3d_joint.pkl \
  --metadata work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.sampling.pkl \
  --primary-metric top1 \
  --per-offset-score-out work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.pkl \
  --per-offset-metadata-out work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.metadata.pkl \
  --out work_dirs/modern/scores/posec3d_joint_10clip_offsets_analysis.json

uv run python tools/modern/test.py \
  configs/modern/posec3d/gym_joint_10clip_uniform.py \
  --out work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_clip.pkl \
  --average-clips none \
  --skip-eval \
  --partial-out work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_clip.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4

uv run python tools/modern/analyze_clip_offsets.py \
  --per-clip-score work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_clip.pkl \
  --baseline-score work_dirs/modern/scores/gym_joint_e1_1clip_uniform.pkl \
  --ten-clip-score work_dirs/modern/scores/gym_joint_e5_10clip_uniform.pkl \
  --metadata work_dirs/modern/scores/gym_joint_e5_10clip_uniform.sampling.pkl \
  --primary-metric mean_class_accuracy \
  --per-offset-score-out work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.pkl \
  --per-offset-metadata-out work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.metadata.pkl \
  --out work_dirs/modern/scores/gym_joint_10clip_offsets_analysis.json
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
| E1 score | `work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl` |
| E2 score | `work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.pkl` |
| E3 score | `work_dirs/modern/scores/posec3d_joint_e3_2clip_uniform.pkl` |
| E4 score | `work_dirs/modern/scores/posec3d_joint_e4_uniform_kns.pkl` |
| E analysis | `work_dirs/modern/scores/posec3d_joint_kns_analysis.json` |
| FineGYM F1 score | `work_dirs/modern/scores/gym_joint_e1_1clip_uniform.pkl` |
| FineGYM F3 score | `work_dirs/modern/scores/gym_joint_e3_2clip_uniform.pkl` |
| FineGYM F4 score | `work_dirs/modern/scores/gym_joint_e4_uniform_kns.pkl` |
| FineGYM F4-v2 score | `work_dirs/modern/scores/gym_joint_e4_uniform_kns_v2.pkl` |
| FineGYM F5 score | `work_dirs/modern/scores/gym_joint_e5_10clip_uniform.pkl` |
| FineGYM analysis | `work_dirs/modern/scores/gym_joint_kns_analysis.json` |
| NTU 10-clip per-view score | `work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_clip.pkl` |
| NTU 10-clip per-offset score | `work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.pkl` |
| NTU 10-clip per-offset metadata | `work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.metadata.pkl` |
| NTU offset analysis | `work_dirs/modern/scores/posec3d_joint_10clip_offsets_analysis.json` |
| FineGYM 10-clip per-view score | `work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_clip.pkl` |
| FineGYM 10-clip per-offset score | `work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.pkl` |
| FineGYM 10-clip per-offset metadata | `work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.metadata.pkl` |
| FineGYM offset analysis | `work_dirs/modern/scores/gym_joint_10clip_offsets_analysis.json` |

完整性检查结果：

- `posec3d_joint.pkl`：16487 samples
- `posec3d_limb.pkl`：16487 samples
- `msg3d_hrnet_joint.pkl`：16487 samples
- `posec3d_joint_e1_1clip_uniform.pkl`：16487 samples
- `posec3d_joint_e2_1clip_kns.pkl`：16487 samples
- `posec3d_joint_e3_2clip_uniform.pkl`：16487 samples
- `posec3d_joint_e4_uniform_kns.pkl`：16487 samples
- `gym_joint_e1_1clip_uniform.pkl`：8521 samples
- `gym_joint_e3_2clip_uniform.pkl`：8521 samples
- `gym_joint_e4_uniform_kns.pkl`：8521 samples
- `gym_joint_e4_uniform_kns_v2.pkl`：8521 samples
- `gym_joint_e5_10clip_uniform.pkl`：8521 samples
- `posec3d_joint_e5_10clip_uniform.per_clip.pkl`：16487 samples，first shape `(20, 60)`
- `posec3d_joint_e5_10clip_uniform.per_offset.pkl`：shape `(16487, 10, 60)`
- `posec3d_joint_e5_10clip_uniform.per_offset.metadata.pkl`：16487 samples，每条 10 offsets
- `gym_joint_e5_10clip_uniform.per_clip.pkl`：8521 samples，first shape `(20, 99)`
- `gym_joint_e5_10clip_uniform.per_offset.pkl`：shape `(8521, 10, 99)`
- `gym_joint_e5_10clip_uniform.per_offset.metadata.pkl`：8521 samples，每条 10 offsets

## Notes

- partial score 文件只用于长任务进度检查，最终指标以非 partial 的 score 和
  metrics 文件为准。
- 本文档目前只覆盖官方权重测试、PoseC3D 分数融合、E/F 组 test-time sampling
  和 10-clip offset 分析；训练阶段采样实验尚未开始。
