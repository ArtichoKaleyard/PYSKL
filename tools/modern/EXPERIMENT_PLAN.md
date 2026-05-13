# NTU60 xsub KNS Test-Time Experiment Plan

本文档记录 `modern` 分支当前第一轮 KNS 实验计划。执行口径以
`EXPERIMENT_PLAN_KNS.md` 为准：先证明 **KNS-v1：速度-加速度分区关键邻域采样**
在 test-time sampling 上是否有信息价值，再考虑训练、FineGYM 或融合扩展。

当前分支已经完成 A-D 官方权重校准；剩余优先级是实现并跑完 E1-E5，然后做
样本级正误迁移分析。

## Scope

本轮只做 test-time sampling：

- 不训练模型。
- 不改 PoseC3D backbone。
- 不改 heatmap 生成逻辑。
- 不改官方 checkpoint。
- 第一轮只使用 NTU60 XSub 的 PoseC3D joint 官方权重。

官方 PoseC3D test pipeline 中的 `GeneratePoseTarget(double=True)` 保持不变，
因此 E 组只改变 `PoseDecode` 之前的 temporal sampler。这里的 `clip 成本`
按 temporal clip 数计，和官方 README 中的 10-clip 测试口径对齐。

暂不处理以下内容：

- 训练阶段采样对照。
- FineGYM / NTU120 扩展。
- limb / joint+limb 的 KNS 融合补充。
- KNS 门控、关节加权、超参搜索和复杂可视化报告。

## Assets

资产由 `tools/modern/prepare_assets.py` 管理：

```bash
uv run python tools/modern/prepare_assets.py --check
uv run python tools/modern/prepare_assets.py --download
```

当前 E 组依赖：

| 名称 | 本地路径 | 用途 |
| --- | --- | --- |
| NTU60 HRNet annotation | `data/nturgbd/ntu60_hrnet.pkl` | `xsub_val` 数据入口 |
| PoseC3D joint checkpoint | `checkpoints/posec3d/slowonly_r50_ntu60_xsub/joint.pth` | E1-E5 统一权重 |

`data/`、`checkpoints/` 和 `work_dirs/` 都是本地大文件或实验产物目录，不进入
git。

## Completed Calibration

A-D 已作为可信参照完成，结果整理在 `tools/modern/RESULTS.md`：

| 组别 | 模型 | 权重 | 采样 | top-1 | 作用 |
| --- | --- | --- | --- | ---: | --- |
| A | PoseC3D joint | 官方 | 10-clip uniform | 0.9373 | joint 校准 |
| B | PoseC3D limb | 官方 | 10-clip uniform | 0.9338 | limb 校准 |
| C | PoseC3D joint+limb | 官方 score fusion | 10-clip uniform | 0.9406 | 融合参照 |
| D | MS-G3D HRNet joint | 官方 | 10-clip uniform | 0.9264 | GCN 对照 |

这些结果只负责建立可信运行栈，不直接参与 KNS 结论。

## E Group: NTU60 XSub Joint Test-Time Sampling

统一控制项：

| 项 | 设定 |
| --- | --- |
| 模型 | PoseC3D SlowOnly-R50 joint |
| 权重 | 官方 checkpoint |
| 数据 | NTU60 XSub HRNet 2D Pose |
| split | `xsub_val` |
| backbone | 不改 |
| heatmap | 不改，保持官方 test-time `double=True` |
| 训练 | 不训练 |
| 输出 | score、metrics、sampled indices、KNS 峰值元数据、正误迁移分析 |

实验矩阵：

| 编号 | 采样方式 | clip 成本 | 配置 |
| --- | --- | ---: | --- |
| E1 | 1-clip uniform | 1x | `configs/modern/posec3d/ntu60_xsub_joint_1clip_uniform.py` |
| E2 | 1-clip KNS-v1 | 1x | `configs/modern/posec3d/ntu60_xsub_joint_1clip_kns.py` |
| E3 | 2-clip uniform | 2x | `configs/modern/posec3d/ntu60_xsub_joint_2clip_uniform.py` |
| E4 | uniform + KNS-v1 | 2x | `configs/modern/posec3d/ntu60_xsub_joint_uniform_kns.py` |
| E5 | 10-clip uniform | 10x | `configs/modern/posec3d/ntu60_xsub_joint_10clip_uniform.py` |

关键比较：

```text
E2 vs E1：同为 1-clip，KNS 是否比 uniform 更有效
E4 vs E3：同为 2-clip，KNS 是否提供互补信息
E4 vs E5：2-clip 能否接近 10-clip
```

## KNS-v1 Implementation Contract

KNS-v1 采样器为 `pyskl.datasets.pipelines.KnsSampleFrames`，只接入 test
pipeline。

固定口径：

- `clip_len=48`。
- 每个粗分区取 `r=3` 帧，因此分区数为 `B=16`。
- 每个分区内分别计算速度峰值 `tv` 和加速度峰值 `ta`。
- `v`、`a` 先按整段视频做 P5/P95 robust normalization。
- 归一化后乘姿态置信度可靠性，并做短窗口时间平滑。
- 若 `|tv - ta| <= 2`，合并为一个关键事件簇。
- 剩余采样位从未覆盖片段中按长度降序取中点补齐。
- 每个 clip 内 frame indices 按时间排序。

`uniform + KNS-v1` 使用一个 uniform clip 加一个 KNS clip，模型仍通过
PoseC3D 原有 `average_clips='prob'` 做 clip 平均。

## Commands

环境与资产检查：

```bash
uv sync --group dev
uv run python scripts/check_env.py
uv run python tools/modern/prepare_assets.py --check
```

E1-E4 正式评估命令：

```bash
uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint_1clip_uniform.py \
  --out work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl \
  --partial-out work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4

uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint_1clip_kns.py \
  --out work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.pkl \
  --partial-out work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4

uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint_2clip_uniform.py \
  --out work_dirs/modern/scores/posec3d_joint_e3_2clip_uniform.pkl \
  --partial-out work_dirs/modern/scores/posec3d_joint_e3_2clip_uniform.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4

uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint_uniform_kns.py \
  --out work_dirs/modern/scores/posec3d_joint_e4_uniform_kns.pkl \
  --partial-out work_dirs/modern/scores/posec3d_joint_e4_uniform_kns.partial.pkl \
  --partial-interval 100 \
  --progress-interval 60 \
  --videos-per-gpu 2 \
  --workers-per-gpu 4
```

E5 可直接复用 A 组 `work_dirs/modern/scores/posec3d_joint.pkl`；如需按 E 组命名
重跑，则使用 `ntu60_xsub_joint_10clip_uniform.py`。

采样元数据导出：

```bash
uv run python tools/modern/export_sampling_metadata.py \
  configs/modern/posec3d/ntu60_xsub_joint_1clip_uniform.py \
  --out work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.sampling.pkl

uv run python tools/modern/export_sampling_metadata.py \
  configs/modern/posec3d/ntu60_xsub_joint_1clip_kns.py \
  --out work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.sampling.pkl

uv run python tools/modern/export_sampling_metadata.py \
  configs/modern/posec3d/ntu60_xsub_joint_2clip_uniform.py \
  --out work_dirs/modern/scores/posec3d_joint_e3_2clip_uniform.sampling.pkl

uv run python tools/modern/export_sampling_metadata.py \
  configs/modern/posec3d/ntu60_xsub_joint_uniform_kns.py \
  --out work_dirs/modern/scores/posec3d_joint_e4_uniform_kns.sampling.pkl

uv run python tools/modern/export_sampling_metadata.py \
  configs/modern/posec3d/ntu60_xsub_joint_10clip_uniform.py \
  --out work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.sampling.pkl
```

正误迁移分析：

```bash
uv run python tools/modern/analyze_kns_scores.py \
  --score E1=work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.pkl \
  --score E2=work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.pkl \
  --score E3=work_dirs/modern/scores/posec3d_joint_e3_2clip_uniform.pkl \
  --score E4=work_dirs/modern/scores/posec3d_joint_e4_uniform_kns.pkl \
  --score E5=work_dirs/modern/scores/posec3d_joint.pkl \
  --metadata E1=work_dirs/modern/scores/posec3d_joint_e1_1clip_uniform.sampling.pkl \
  --metadata E2=work_dirs/modern/scores/posec3d_joint_e2_1clip_kns.sampling.pkl \
  --metadata E3=work_dirs/modern/scores/posec3d_joint_e3_2clip_uniform.sampling.pkl \
  --metadata E4=work_dirs/modern/scores/posec3d_joint_e4_uniform_kns.sampling.pkl \
  --metadata E5=work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.sampling.pkl \
  --out work_dirs/modern/scores/posec3d_joint_kns_analysis.json
```

## Success Criteria

强成功：

```text
E2 > E1
E4 > E3
E4 接近 E5
```

中等成功：

```text
E2 ~= E1
E4 > E3
E4 接近 E5
```

弱成功但仍可分析：

```text
总体精度收益小，但 KNS Fixes 明显集中在高运动、细粒度或长视频样本。
```

失败：

```text
E2 < E1
E4 < E3
KNS Breaks 明显多于 KNS Fixes
```

若失败，优先检查极值点是否被 pose 抖动误导、置信度可靠性是否太弱、粗分区
是否过细或 `tv/ta` 合并阈值是否不合适。

## Decision Gates

1. E1-E5 跑完后，先看成本-精度表和正误迁移表。
2. 只有当 E2/E4 至少显示同成本或互补收益时，才进入 FineGYM 的 F1-F5。
3. 只有当 E 或 F 证明 KNS 有价值时，才补 limb / fusion 的 G 组。
4. 只有 test-time 结果站稳后，再考虑训练阶段采样配置。
