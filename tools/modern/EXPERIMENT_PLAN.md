# NTU60 xsub Modern Branch Experiment Plan

本文档记录 `modern` 分支上的暂定实验计划。当前目标不是完整重构
PYSKL，而是在 Python 3.11、CUDA 12.8、PyTorch 2.7 和 OpenMMLab
2.x 依赖下，先打通 NTU60 xsub 的关键验证、对照和采样实验链路。

计划仍可能随实验结果调整。本文档优先作为当前分支的执行清单和对照依据，
不替代正式实验报告。

## Scope

本轮迁移聚焦以下内容：

- 使用 OpenMMLab 官方提供的 `ntu60_hrnet.pkl` 作为 NTU60 HRNet 2D
  pose annotation。
- 使用 PoseC3D joint、PoseC3D limb 和 MS-G3D HRNet 2D 官方权重验证
  现代环境下的测试流程。
- 保留 PoseC3D joint+limb 分数融合作为核心复现目标。
- 围绕 PoseC3D joint 的原始采样和自定义采样做训练对比。
- 仅在需要最终融合或采样结论更完整时，再展开 limb 自定义采样训练。

暂不处理以下内容：

- 全项目 OpenMMLab 1.x 到 2.x 的系统性重构。
- RGBPose、demo、notebook、旧版分布式入口和所有历史配置的全面兼容。
- NTU60 原始 RGB 视频或原始骨架文件的重新预处理。
- 多机训练、完整超参搜索和大规模采样策略网格搜索。

## Assets

资产由 `tools/modern/prepare_assets.py` 管理：

```bash
uv run python tools/modern/prepare_assets.py --check
uv run python tools/modern/prepare_assets.py --download
```

当前计划依赖的官方资源如下：

| 名称                            | 本地路径                                                    | 用途          |
|-------------------------------|---------------------------------------------------------|-------------|
| NTU60 HRNet annotation        | `data/nturgbd/ntu60_hrnet.pkl`                          | A-G 的统一数据入口 |
| PoseC3D joint checkpoint      | `checkpoints/posec3d/slowonly_r50_ntu60_xsub/joint.pth` | A、C 的官方权重   |
| PoseC3D limb checkpoint       | `checkpoints/posec3d/slowonly_r50_ntu60_xsub/limb.pth`  | B、C 的官方权重   |
| MS-G3D HRNet joint checkpoint | `checkpoints/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.pth`  | D 的官方权重     |

`data/` 和 `checkpoints/` 都是本地大文件目录，不进入 git。

## Experiment Matrix

| 组别 | 模型                    |        是否训练 | 配置 / 入口                                                        | 作用                         |
|----|-----------------------|------------:|----------------------------------------------------------------|----------------------------|
| A  | PoseC3D joint 官方权重    |           否 | `configs/modern/posec3d/ntu60_xsub_joint.py`                   | 验证 PoseC3D joint 测试流程      |
| B  | PoseC3D limb 官方权重     |           否 | `configs/modern/posec3d/ntu60_xsub_limb.py`                    | 验证 limb 流和 limb heatmap 配置 |
| C  | PoseC3D joint+limb 融合 | 否 / 可复现分数融合 | `tools/modern/fuse_scores.py`                                  | 复现核心融合精度                   |
| D  | MS-G3D HRNet 2D 官方权重  |           否 | `configs/modern/msg3d/ntu60_xsub_hrnet_joint.py`               | 提供 GCN 对照锚点                |
| E  | PoseC3D joint 原始采样    |           是 | `configs/modern/posec3d/ntu60_xsub_joint_original_sampling.py` | 采样实验 baseline              |
| F  | PoseC3D joint 自定义采样   |           是 | `configs/modern/posec3d/ntu60_xsub_joint_custom_sampling.py`   | 检查采样策略是否有效                 |
| G  | PoseC3D limb 自定义采样    |          可选 | `configs/modern/posec3d/ntu60_xsub_limb_custom_sampling.py`    | 仅在需要最终融合时再跑                |

## Execution Order

### Stage 0: Environment And Assets

先确认现代环境和官方资产：

```bash
uv sync --group dev
uv run python scripts/check_env.py
uv run python tools/modern/prepare_assets.py --check
```

通过标准：

- Python 为 3.11。
- PyTorch 为 2.7.x，CUDA runtime 为 12.8。
- GPU 可见，`mmcv.ops` 可用。
- 四个官方资产均显示 `[OK]`。

### Stage 1: Official-Weight Evaluation

先跑不训练的 A、B、D，确认数据、模型、checkpoint 和 score export 全链路：

```bash
uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_joint.py \
  --out work_dirs/modern/scores/posec3d_joint.pkl

uv run python tools/modern/test.py \
  configs/modern/posec3d/ntu60_xsub_limb.py \
  --out work_dirs/modern/scores/posec3d_limb.pkl

uv run python tools/modern/test.py \
  configs/modern/msg3d/ntu60_xsub_hrnet_joint.py \
  --out work_dirs/modern/scores/msg3d_hrnet_joint.pkl \
  --eval top_k_accuracy
```

通过标准：

- 三个入口都能完整跑完 `xsub_val`。
- 产物写入 `work_dirs/modern/scores/`。
- top-1 / top-5 指标处于合理区间；若明显偏离官方 README，优先排查
  pipeline、checkpoint key 和数据 split。

### Stage 2: PoseC3D Score Fusion

在 A、B 的 score 产物基础上跑 C：

```bash
uv run python tools/modern/fuse_scores.py \
  --scores work_dirs/modern/scores/posec3d_joint.pkl work_dirs/modern/scores/posec3d_limb.pkl \
  --ann-file data/nturgbd/ntu60_hrnet.pkl \
  --split xsub_val \
  --out work_dirs/modern/scores/posec3d_joint_limb_fusion.json
```

通过标准：

- 输出 joint、limb 融合后的 top-1 / top-5 / mean class accuracy。
- 至少保留默认等权融合结果；如后续需要，可追加不同权重组合。

### Stage 3: Sampling Training Baseline

先跑 E，再跑 F。短跑验证可加 `--max-epochs 1`，正式训练不加该参数：

```bash
uv run python tools/modern/train.py \
  configs/modern/posec3d/ntu60_xsub_joint_original_sampling.py \
  --validate

uv run python tools/modern/train.py \
  configs/modern/posec3d/ntu60_xsub_joint_custom_sampling.py \
  --validate
```

对比重点：

- E 和 F 使用同一数据、模型骨架、epoch 设置和评估口径。
- F 只改变采样策略，避免同时引入其他增强或训练策略差异。
- 每轮验证 score 和 checkpoint 写入各自的 `work_dirs/modern/posec3d/`
  子目录。

### Stage 4: Optional Limb Sampling

只有当以下条件之一成立时，再跑 G：

- F 相比 E 有稳定收益，需要验证 joint+limb 采样融合是否继续有效。
- 正式实验报告需要完整的 limb 自定义采样对照。
- A/B/C 复现结果显示 limb 流对最终结论影响较大。

命令：

```bash
uv run python tools/modern/train.py \
  configs/modern/posec3d/ntu60_xsub_limb_custom_sampling.py \
  --validate
```

## Outputs

建议保留以下产物路径：

| 类型            | 路径                                                  |
|---------------|-----------------------------------------------------|
| 官方权重 score    | `work_dirs/modern/scores/*.pkl`                     |
| 融合指标          | `work_dirs/modern/scores/*fusion*.json`             |
| 训练 checkpoint | `work_dirs/modern/posec3d/*/*.pth`                  |
| 训练验证 score    | `work_dirs/modern/posec3d/*/val_scores_epoch_*.pkl` |
| 后续实验报告        | `work_dirs/modern/reports/` 或单独新增报告文档               |

`work_dirs/` 默认视为本地实验产物，不进入 git。需要长期保留的数值结果应
整理进后续报告文档，而不是依赖本地产物目录。

## Current Decision Gates

继续推进前优先检查以下节点：

1. A/B/D 是否能完整复现官方权重测试流程。
2. C 的融合指标是否能作为后续实验可信基线。
3. E/F 是否能在同等训练条件下比较采样策略。
4. 是否有必要跑 G；如果 joint 自定义采样没有收益，G 默认后置。

如果 A-D 任一官方权重验证失败，先修测试链路，不进入训练阶段。如果 E/F
训练成本明显高于预期，先用短跑和较小验证频率定位问题，再决定是否正式跑满。
