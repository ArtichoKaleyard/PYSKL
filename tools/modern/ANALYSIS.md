# Modern Branch Analysis Notes

本文档集中记录 `modern` 分支上不属于主指标复现实验的后续分析。主实验指标仍放在
`tools/modern/RESULTS.md`；本文件关注样本迁移、10-clip offset 贡献、以及
KNS-v1 采样分布与分类正确性的关系。

原始 score、metadata 和分析 JSON 均位于 `work_dirs/modern/scores/`。该目录是
本地实验产物目录，不进入 git。

## 10-Clip Offset Analysis

### Setup

PoseC3D 官方 test pipeline 使用 `UniformSampleFrames(clip_len=48, num_clips=10)`
和 `GeneratePoseTarget(double=True)`。因此模型实际看到的不是 10 个 view，而是
20 个 view：10 个 temporal offsets 加对应的水平翻转 view。

本轮通过 `tools/modern/test.py --average-clips none` 导出 raw per-view logits，
再由 `tools/modern/analyze_clip_offsets.py` 先对同一 temporal offset 的两个
flip views 做概率平均，得到按时间 offset 聚合后的 score：

| 数据集 | per-offset score shape | primary metric |
| --- | ---: | --- |
| NTU60 XSub | `(16487, 10, 60)` | top-1 |
| FineGYM | `(8521, 10, 99)` | mean class accuracy |

脚本重算的 all-offset 指标与既有 10-clip score 完全一致，说明导出和聚合口径与
原始 `average_clips='prob'` 结果一致。

### NTU60 Offset Summary

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

| cost | exhaustive best offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[4]` | 0.9369 | 0.9368 |
| 2 | `[4, 8]` | 0.9377 | 0.9376 |
| 3 | `[2, 4, 7]` | 0.9381 | 0.9380 |
| 4 | `[2, 4, 5, 7]` | 0.9384 | 0.9383 |
| 5 | `[2, 3, 4, 7, 8]` | 0.9386 | 0.9385 |
| 10 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | 0.9373 | 0.9372 |

| cost | greedy offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[4]` | 0.9369 | 0.9368 |
| 2 | `[4, 8]` | 0.9377 | 0.9376 |
| 3 | `[4, 8, 0]` | 0.9381 | 0.9380 |
| 4 | `[4, 8, 0, 5]` | 0.9381 | 0.9380 |
| 5 | `[4, 8, 0, 5, 9]` | 0.9379 | 0.9378 |
| 10 | `[4, 8, 0, 5, 9, 2, 7, 3, 1, 6]` | 0.9373 | 0.9372 |

NTU60 上单 offset 最强的是 offset 4。完整 10-clip top-1 为 0.9373，而最佳
5-offset 子集 `[2, 3, 4, 7, 8]` 达到 0.9386。leave-one-out 显示 offset 6
最可疑：移除 offset 6 后 top-1 提高约 0.055 个百分点。

### FineGYM Offset Summary

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

| cost | exhaustive best offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[5]` | 0.9539 | 0.9341 |
| 2 | `[7, 9]` | 0.9556 | 0.9376 |
| 3 | `[5, 7, 9]` | 0.9575 | 0.9387 |
| 4 | `[2, 3, 4, 9]` | 0.9578 | 0.9398 |
| 5 | `[0, 2, 4, 8, 9]` | 0.9572 | 0.9395 |
| 10 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | 0.9575 | 0.9381 |

| cost | greedy offsets | top-1 | mean class |
| ---: | --- | ---: | ---: |
| 1 | `[5]` | 0.9539 | 0.9341 |
| 2 | `[5, 8]` | 0.9562 | 0.9365 |
| 3 | `[5, 8, 7]` | 0.9572 | 0.9381 |
| 4 | `[5, 8, 7, 2]` | 0.9573 | 0.9386 |
| 5 | `[5, 8, 7, 2, 6]` | 0.9575 | 0.9384 |
| 10 | `[5, 8, 7, 2, 6, 3, 4, 9, 1, 0]` | 0.9575 | 0.9381 |

FineGYM 上 offset 价值更集中。按 mean class，offset 5 是最强单视图；最佳
4-offset 子集 `[2, 3, 4, 9]` 达到 0.9398，比完整 10-clip 高约 0.17 个百分点。
leave-one-out 中移除 offset 0 后 mean class 反而提升约 0.030 个百分点。

## Positive Offsets And KNS-v1

### Correctness-Aware Comparison

比较 KNS-v1 与正收益 uniform offsets 时，必须区分分类正确性，不能只比较采样帧
是否接近。这里的关键集合是：

- NTU60：`F10 = {E1 错, E5 对}`，共 115 个样本。
- FineGYM：`F10 = {F1 错, F5 对}`，共 81 个样本。

| 数据集 | offset 子集 | KNS 口径 | F10 数 | 子集对 | KNS 对 | 两者都对 | 子集对 KNS 错 | KNS 对子集错 | 两者都错 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NTU60 | `[2, 3, 4, 7, 8]` | E2 KNS-only | 115 | 110 | 71 | 69 | 41 | 2 | 3 |
| FineGYM | `[2, 3, 4, 9]` | F4 uniform+KNS | 81 | 73 | 37 | 35 | 38 | 2 | 6 |

在真正由 10-clip 救回的样本里，正收益 offset 子集比 KNS-v1 更能解释收益。尤其是
`子集对 KNS 错` 的样本数量远大于 `KNS 对子集错`，说明 KNS-v1 没有稳定捕获
10-clip 的主要收益来源。

### KNS-v1 Peak And Fill Split

KNS-v1 每个粗分区最多保留速度峰值 `tv`、加速度峰值 `ta`，当两者很近时会合并；
剩余采样位来自最长未覆盖区间的 fill/补点。因此不能把全部 KNS 采样帧都视为峰值帧。

| 数据集 | all frames/sample | unique peak frames/sample | fill frames/sample |
| --- | ---: | ---: | ---: |
| NTU60 | 48.00 | 21.95 | 25.93 |
| FineGYM | 48.00 | 21.06 | 15.67 |

按每个 48-frame offset clip 统计 overlap，可以看到之前“uniform offset 与 KNS
接近”的一部分来自 KNS fill 帧，而不是速度/加速度峰值本身：

| 数据集 | offset 组 | 样本集合 | dist all | dist peak | dist fill | overlap all | overlap peak | overlap fill |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NTU60 | 正收益 offsets | all samples | 0.445 | 1.415 | 1.028 | 0.606 | 0.280 | 0.325 |
| NTU60 | best-5 | F10 且子集对 | 0.623 | 1.842 | 1.187 | 0.490 | 0.217 | 0.273 |
| NTU60 | best-5 | F10 且子集对 KNS 错 | 0.626 | 1.899 | 1.156 | 0.485 | 0.206 | 0.279 |
| FineGYM | 正收益 offsets | all samples | 0.138 | 0.679 | 1.284 | 0.892 | 0.564 | 0.328 |
| FineGYM | best-4 | F10 且子集对 | 0.646 | 1.774 | 1.725 | 0.612 | 0.379 | 0.233 |
| FineGYM | best-4 | F10 且子集对 KNS 错 | 0.517 | 1.483 | 1.607 | 0.683 | 0.437 | 0.246 |

结论：KNS-v1 与正收益 offset 的几何接近不能直接解释分类收益。KNS-v1 的 48 个
采样点中大量是 fill 帧；而在 10-clip 真修复样本里，正收益 uniform offset 子集
经常正确、KNS-v1 错误。当前更合理的解释是：10-clip 收益主要来自 uniform 相位
之间的互补，而不是速度/加速度峰值本身。

## Skeleton Sampling Viewer

`tools/modern/visualize_sampling.py` 可以把骨架序列、归一化速度/加速度/pose
confidence 曲线、以及 E1/F1、E3/F3、KNS-v1、KNS-v2、10-clip offset 和最佳
offset 子集采样条集中导出为一个静态 HTML。HTML 自包含数据和脚本，不依赖本地
web server，适合逐样本拖动时间轴或按视频方式播放骨架运动。

默认推荐样本会优先选择“分类正确性上有解释价值”的目标，而不是只按采样几何接近
排序：

1. 1-clip 错、10-clip 对、KNS-v1 错。
2. 1-clip 错、最佳 offset 子集对、KNS-v1 错。
3. 1-clip 错、10-clip 对。
4. 1-clip 对、KNS-v1 错。
5. 其他上下文样本。

KNS 采样条会区分速度峰值、加速度峰值、合并峰值和 fill 帧，避免把全部 KNS
采样帧都误读成峰值帧。推荐列表和 prediction card 同时显示 E1/F1、E5/F5、
KNS-v1 与最佳 offset 子集的分类正确性，因此可以直接检查“采样位置接近”是否
真的对应“分类被修复”。采样条的横坐标是原始时间轴上的真实 frame id；每行会显示
`unique/total`，并标出该采样序列的起点和终点。对于 `total_frames < clip_len`
的短序列，uniform offset 可能真实覆盖几乎所有帧并发生环绕，因此采样条会显得
非常整齐；这种情况下应结合起点/终点和 `unique/total` 判断 offset 相位，而不是
只看是否覆盖了哪些 frame id。页面高度会按视口和采样行数自适应：优先保证骨架区
与采样条行高可读，空间不足时让页面滚动，而不是压薄采样条。可视区域采用稳定的
模块流：骨架与预测卡位于上方预览区，曲线与采样条各自独立成块，避免动态高度导致
不同 canvas 相互覆盖。所有 uniform offset 使用同一种蓝色，offset 编号只通过行名
和起点/终点标记区分；颜色仅表达采样类型，避免把 offset id 误读成额外语义。同一行
内重复采到同一 frame 时会先聚合计数，再按该行最大重复次数归一化颜色透明度；低频点
仍保留可见的最低透明度，高频重复点颜色更实。若该行所有可见 frame 的采样次数相同，
则不降低透明度，避免把均匀采样误读成低置信度或弱采样。

## Offset Subset Temporal Weight Association

本轮新增 `tools/modern/analyze_offset_weight_patterns.py`，只分析 10 个 uniform
offset 的子集所诱导出的时间权重分布与分类收益之间的关联。不训练模型，不修改
backbone，不把结果解释为新采样器。

分析对 `m=1..4` 的 offset subset 做完整枚举，并为每个样本/子集构造原始帧权重：

```text
W_S[t] = frame t 在 subset S 中被采样的次数 / subset 总采样位数
```

同时计算 `N_eff = 1 / sum(W[t]^2)`、concentration、top-20% 高权重 cluster、
`mean + alpha * std` 高权重 cluster、fix/break、subset 级 Spearman，以及样本级
Spearman 分布。完整结果位于 `work_dirs/modern/offset_analysis/`。

关键结论是：offset subset 确实形成了非均匀时间权重，但这种非均匀性与分类收益的
关联很弱，而且 FineGYM 与 NTU60 不一致。

| 数据集 | best subset | primary | mean concentration | mean N_eff | fixes | breaks | overlap with 10-clip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FineGYM | `[2, 3, 4, 9]` | 0.9398 mean class | 0.0009 | 43.09 | 87 | 33 | 0.9012 |
| NTU60 | `[2, 4, 5, 7]` | 0.9384 top-1 | 0.0021 | 70.83 | 140 | 97 | 0.9130 |

subset 级相关性没有给出稳定支持：

| 数据集 | cost | corr(concentration, primary) | corr(N_eff, primary) | corr(cluster span, primary) | corr(largest mass, primary) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineGYM | 4 | 0.0209 | 0.0574 | -0.0915 | -0.0031 |
| NTU60 | 4 | 0.1657 | -0.1350 | -0.3130 | -0.1530 |

样本级也很弱：在同一样本的不同 offset subset 间，concentration 与 label score 的
Spearman median 只有 FineGYM `0.0230`、NTU60 `0.0193`。这说明“更集中的时间权重
带来更高正确类分数”不是一个稳定现象。

fixes 与 breaks 的权重模式差异也很小。best subset 的 fix 样本 concentration
均值只比 break 样本略高：FineGYM `0.00208 vs 0.00188`，NTU60
`0.00257 vs 0.00244`。多峰覆盖在 best subset 中很常见，但它更像 offset 子集
叠加后的自然结果，而不是足以解释分类收益的独立变量。

KNS 对照进一步说明“集中”不是充分条件。在 `F10 且 KNS 错` 的样本上，KNS 反而比
best subset 更集中：FineGYM concentration `0.0088 vs 0.0019`，NTU60
`0.0105 vs 0.0026`。因此 KNS 失败不能简单归因于“不够集中”；更合理的解释仍是
uniform offset 的相位互补和模型 test-time ensemble 效应，而不是速度/加速度峰值
或单纯高权重 cluster。

当前证据不足以支持继续从“时间权重聚类/集中度”角度直接设计新采样策略。若后续要
继续，应把问题收窄为 label-free 的 offset phase selection 或 ensemble redundancy
分析，而不是把 best subset 的权重深浅直接转成 KNS-v3。

### A/B Split Stability

为检验 best subset 是否稳定，按 `sample_id` 的 SHA1 hash 做 deterministic A/B
拆分；拆分不使用 label 或 prediction。然后在每个 split 内独立选择 best subset，
再比较 A-best、B-best 与 full-best 的重合。

| 数据集 | cost | full-best | A-best | B-best | A/B Jaccard | full/A Jaccard | full/B Jaccard |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| FineGYM | 1 | `[5]` | `[5]` | `[9]` | 0.0000 | 1.0000 | 0.0000 |
| FineGYM | 2 | `[7, 9]` | `[7, 9]` | `[6, 9]` | 0.3333 | 1.0000 | 0.3333 |
| FineGYM | 3 | `[5, 7, 9]` | `[2, 4, 9]` | `[5, 6, 7]` | 0.0000 | 0.2000 | 0.5000 |
| FineGYM | 4 | `[2, 3, 4, 9]` | `[2, 3, 4, 9]` | `[1, 2, 6, 9]` | 0.3333 | 1.0000 | 0.3333 |
| NTU60 | 1 | `[4]` | `[4]` | `[4]` | 1.0000 | 1.0000 | 1.0000 |
| NTU60 | 2 | `[4, 8]` | `[4, 5]` | `[4, 8]` | 0.3333 | 0.3333 | 1.0000 |
| NTU60 | 3 | `[2, 4, 7]` | `[0, 8, 9]` | `[1, 4, 7]` | 0.0000 | 0.0000 | 0.5000 |
| NTU60 | 4 | `[2, 4, 5, 7]` | `[2, 4, 5, 7]` | `[1, 2, 7, 8]` | 0.3333 | 1.0000 | 0.3333 |

这个结果支持“best subset 不稳定”的判断。除了 NTU60 cost=1 这种单 offset 明显最强的
情况，A/B 两半的 best subset 经常不同；FineGYM cost=3 和 NTU60 cost=3 的 A/B
Jaccard 都是 0。full-best 常常更接近 A-best，而 B-best 另选了一组，这说明当前
best subset 很可能带有验证集选择波动，不应被解释为稳定采样规律。

## Commands

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

生成可视化分析器：

```bash
uv run python tools/modern/visualize_sampling.py \
  --preset ntu \
  --recommend-top-k 20 \
  --out work_dirs/modern/analysis/ntu_sampling_viewer.html

uv run python tools/modern/visualize_sampling.py \
  --preset gym \
  --recommend-top-k 20 \
  --out work_dirs/modern/analysis/gym_sampling_viewer.html
```

也可以手动指定样本，便于复查某个修复或误伤样本：

```bash
uv run python tools/modern/visualize_sampling.py \
  --preset ntu \
  --sample-id S001C001P001R001A001 \
  --out work_dirs/modern/analysis/ntu_sampling_viewer_case.html
```

运行 offset 子集时间权重关联性分析：

```bash
uv run python tools/modern/analyze_offset_weight_patterns.py \
  --datasets finegym ntu60 \
  --out-dir work_dirs/modern/offset_analysis \
  --max-cost 4 \
  --sample-csv
```

如果 sample-level CSV 已存在，只需要刷新 summary 和典型样本，可以避免重复写大文件：

```bash
uv run python tools/modern/analyze_offset_weight_patterns.py \
  --datasets finegym ntu60 \
  --out-dir work_dirs/modern/offset_analysis \
  --max-cost 4 \
  --no-sample-csv
```

## Analysis Artifacts

| 产物 | 路径 |
| --- | --- |
| NTU 10-clip per-view score | `work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_clip.pkl` |
| NTU 10-clip per-offset score | `work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.pkl` |
| NTU 10-clip per-offset metadata | `work_dirs/modern/scores/posec3d_joint_e5_10clip_uniform.per_offset.metadata.pkl` |
| NTU offset analysis | `work_dirs/modern/scores/posec3d_joint_10clip_offsets_analysis.json` |
| FineGYM 10-clip per-view score | `work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_clip.pkl` |
| FineGYM 10-clip per-offset score | `work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.pkl` |
| FineGYM 10-clip per-offset metadata | `work_dirs/modern/scores/gym_joint_e5_10clip_uniform.per_offset.metadata.pkl` |
| FineGYM offset analysis | `work_dirs/modern/scores/gym_joint_10clip_offsets_analysis.json` |
| NTU sampling viewer | `work_dirs/modern/analysis/ntu_sampling_viewer.html` |
| FineGYM sampling viewer | `work_dirs/modern/analysis/gym_sampling_viewer.html` |
| Offset weight association summary | `work_dirs/modern/offset_analysis/summary.md` |
| FineGYM offset subset metrics | `work_dirs/modern/offset_analysis/offset_subset_metrics_finegym.csv` |
| NTU offset subset metrics | `work_dirs/modern/offset_analysis/offset_subset_metrics_ntu60.csv` |
| FineGYM sample-level weight metrics | `work_dirs/modern/offset_analysis/sample_level_weight_metrics_finegym.csv` |
| NTU sample-level weight metrics | `work_dirs/modern/offset_analysis/sample_level_weight_metrics_ntu60.csv` |
| Offset weight typical samples | `work_dirs/modern/offset_analysis/typical_samples.json` |
