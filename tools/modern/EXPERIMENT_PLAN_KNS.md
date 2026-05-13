## 实验总目标

验证 **KNS-v1：速度-加速度分区关键邻域采样** 能否在不改 PoseC3D backbone、不训练新模块的情况下，用明显低于 10-clip 的推理成本，获得接近 10-clip 的有效收益。

核心问题不是“能不能刷过 10-clip”，而是：

```text
10-clip 靠多次 uniform 抽样降低运气成分；
KNS-v1 能不能把每个分区内的运气，改成基于骨架运动线索的定向选择。
```

---

## 0. 实验范围先收住

第一轮只做 **test-time sampling**，不训练模型、不动 backbone、不动 heatmap 生成逻辑。

优先数据集：

| 优先级 | 数据集                | 作用                             |
| --- | ------------------ | ------------------------------ |
| 1   | **NTU60 XSub**     | 主验证；和官方 PoseC3D / MS-G3D 对照最清楚 |
| 2   | **FineGYM**        | 细粒度动作更容易体现采样价值                 |
| 3   | NTU120 XSub / XSet | 可选扩展，类别更多但工作量更大                |

pyskl 的 PoseC3D model zoo 已经提供 NTU60、NTU120、Kinetics-400、FineGYM 等权重与配置；其中 NTU60 XSub SlowOnly-R50 的 joint / limb / two-stream 分别是 93.7 / 93.4 / 94.1，FineGYM SlowOnly-R50 的 joint / limb / two-stream 分别是 93.8 / 93.8 / 94.1。([GitHub][1])

---

## 1. A-D：官方权重校准组

这部分 Codex 已经在做，目标是确认环境、数据、checkpoint、fusion 脚本没问题。

| 组别 | 数据集        | 模型                      | 权重              | 采样   | 目的       |
| -- | ---------- | ----------------------- | --------------- | ---- | -------- |
| A  | NTU60 XSub | PoseC3D joint           | 官方              | 官方默认 | 校准 joint |
| B  | NTU60 XSub | PoseC3D limb            | 官方              | 官方默认 | 校准 limb  |
| C  | NTU60 XSub | PoseC3D joint+limb      | 官方 score fusion | 官方默认 | 复现融合精度   |
| D  | NTU60 XSub | MS-G3D / 强 GCN baseline | 官方              | 官方默认 | GCN 对照锚点 |

这一步只负责建立可信参照，不参与 KNS 结论。

---

## 2. E 组：NTU60 XSub 采样主实验

第一轮只用 **PoseC3D joint 官方权重**。joint 流最干净，避免 limb / fusion 把采样效果搅混。

### 统一控制项

| 项        | 设定                                        |
| -------- | ----------------------------------------- |
| 模型       | PoseC3D SlowOnly-R50 joint                |
| 权重       | 官方 checkpoint                             |
| 数据       | NTU60 XSub HRNet 2D Pose                  |
| backbone | 不改                                        |
| heatmap  | 不改                                        |
| 训练       | 不训练                                       |
| 输出       | 每个样本保存预测 score / label / sampled indices  |
| 指标       | Top-1、mean class accuracy、推理 clip 成本、正误迁移 |

pyskl 官方测试命令支持 `--out result.pkl` 导出结果，适合后续做样本级分析。([GitHub][1])

### 实验矩阵

| 编号 | 采样方式                 | clip 成本 | 目的                       |
| -- | -------------------- | ------: | ------------------------ |
| E1 | **1-clip uniform**   |      1× | 低成本基线                    |
| E2 | **1-clip KNS-v1**    |      1× | 验证 KNS 是否在同成本下优于 uniform |
| E3 | **2-clip uniform**   |      2× | 排除“只是多一个 view”的影响        |
| E4 | **uniform + KNS-v1** |      2× | 最强低成本组合                  |
| E5 | **10-clip uniform**  |     10× | 官方高成本参照                  |

最关键比较：

```text
E2 vs E1：同为 1-clip，KNS 是否比 uniform 更有效
E4 vs E3：同为 2-clip，KNS 是否提供互补信息
E4 vs E5：2-clip 能否接近 10-clip
```

pyskl 官方 README 也明确提示多 clip 测试耗时，可将 `num_clips=10` 改为 `num_clips=1`，说明 10-clip 本身就是高成本测试增强而非模型结构能力。([GitHub][1])

---

## 3. KNS-v1 固定实现口径

### 3.1 采样结构

假设 PoseC3D 需要采样 (L) 帧，每个粗分区取 (r=3) 帧：

[
B = L / r
]

流程：

```text
整段视频
→ 划成 B 个粗分区
→ 每个分区内分别找速度极值点 tv 和加速度极值点 ta
→ 若 tv/ta 很近，合并为关键事件簇
→ 剩余采样点按被峰值点切开的片段长度降序分配
→ 所有采样帧按时间排序，组成长度 L 的输入 clip
```

### 3.2 速度和加速度

速度：

[
v_t = \text{mean}*j \left| p*{t,j} - p_{t-1,j} \right|
]

加速度：

[
a_t = \text{mean}*j \left| (p*{t,j}-p_{t-1,j}) - (p_{t-1,j}-p_{t-2,j}) \right|
]

第一版关节先等权，后续再考虑四肢末端加权。

### 3.3 归一化

采用 **分模态二阶段归一化**：

```text
第一阶段：v / a 分别在整段视频内 robust normalization
第二阶段：乘姿态置信度可靠性，再做时间平滑
```

robust normalization：

[
\hat{x}*t =
clip\left(
\frac{x_t - P_5(x)}
{P*{95}(x)-P_5(x)+\epsilon},
0,1
\right)
]

其中 (x) 分别是 (v) 和 (a)。

不做分区归一化，避免每个平静分区都被强行制造出一个“局部最高点”。

### 3.4 近邻合并规则

```text
如果 |tv - ta| <= 2 帧：
    视为同一个关键事件簇
    center = round((tv + ta) / 2)
    剩余采样位按左右片段长度分配
否则：
    固定保留 tv 和 ta
    第 3 个采样点给最长剩余片段的中点
```

---

## 4. F 组：FineGYM 有利数据集验证

当 E 组跑完后，如果 KNS 在 NTU60 上至少不差，就加 FineGYM。

| 编号 | 数据集     | 模型            | 权重 | 采样               |
| -- | ------- | ------------- | -- | ---------------- |
| F1 | FineGYM | PoseC3D joint | 官方 | 1-clip uniform   |
| F2 | FineGYM | PoseC3D joint | 官方 | 1-clip KNS-v1    |
| F3 | FineGYM | PoseC3D joint | 官方 | 2-clip uniform   |
| F4 | FineGYM | PoseC3D joint | 官方 | uniform + KNS-v1 |
| F5 | FineGYM | PoseC3D joint | 官方 | 10-clip uniform  |

FineGYM 官方表报告的是 **mean class Top-1**，不是普通 Top-1，记录时要分开写。([GitHub][1])

---

## 5. G 组：融合补充实验

只有在 E / F 证明 KNS 有价值后，再补 fusion。

| 编号 | 数据集        | 模型                 | 权重              | 采样               | 目的            |
| -- | ---------- | ------------------ | --------------- | ---------------- | ------------- |
| G1 | NTU60 XSub | PoseC3D limb       | 官方              | 1-clip KNS-v1    | 看 limb 是否同样受益 |
| G2 | NTU60 XSub | PoseC3D joint+limb | 官方 score fusion | uniform + KNS-v1 | 看融合精度         |
| G3 | FineGYM    | PoseC3D limb       | 官方              | 1-clip KNS-v1    | 细粒度补充         |
| G4 | FineGYM    | PoseC3D joint+limb | 官方 score fusion | uniform + KNS-v1 | 最终低成本融合       |

这里不要一开始就做，避免实验量翻倍。

---

## 6. 样本级分析必须做

每个实验保存：

```text
sample_id
label
pred_top1
pred_top5
score vector
sampled frame indices
v/a peak positions
global video length
mean confidence
```

### 6.1 正误迁移

以 E1 为基线，对 E2 / E4 / E5 分别统计：

| 类别           | 含义          |
| ------------ | ----------- |
| Both Correct | E1 对，KNS 也对 |
| KNS Fixes    | E1 错，KNS 对  |
| KNS Breaks   | E1 对，KNS 错  |
| Both Wrong   | 都错          |

重点不是只看 Top-1，而是看：

```text
KNS Fixes 是否多于 KNS Breaks
KNS Fixes 是否集中在高运动、细粒度、长视频样本
KNS Breaks 是否集中在低置信度或局部运动误导样本
```

### 6.2 和 10-clip 的收益重合度

定义：

[
F_{10}={x|1c错,10c对}
]

[
F_{kns}={x|1c错,KNS对}
]

看：

[
\frac{|F_{10}\cap F_{kns}|}{|F_{10}|}
]

这个指标回答：**KNS 替代了多少 10-clip 的有效收益。**

---

## 7. 成功标准

不把“超过 10-clip”作为硬目标。

### 强成功

```text
E2 > E1
E4 > E3
E4 接近 E5
```

说明 KNS 在同成本下优于 uniform，并且作为第二视图有互补价值。

### 中等成功

```text
E2 ≈ E1
E4 > E3
E4 接近 E5
```

说明 KNS 单独不足以替代 uniform，但和 uniform 组合能补充信息。

### 弱成功但仍可讲

```text
总精度提升小
但 KNS Fixes 明显集中在高运动/细粒度类别
```

说明 KNS 的作用是局部有效，需要后续门控减少误修正。

### 失败

```text
E2 < E1
E4 < E3
KNS Breaks 明显多于 Fixes
```

优先检查：极值点是否被 pose 抖动骗、置信度可靠性是否过弱、粗分区是否太少、tv/ta 合并阈值是否不合适。

---

## 8. 第一轮最终交付物

实验结束后整理成 4 张表 / 图：

1. **成本-精度表**
   `1 uniform / 1 KNS / 2 uniform / uniform+KNS / 10 uniform`

2. **成本-精度曲线**
   横轴 clip 成本，纵轴 accuracy。

3. **正误迁移表**
   Both Correct / KNS Fixes / KNS Breaks / Both Wrong。

4. **Fixes vs Breaks 分布图**
   对比视频长度、运动峰值、置信度稳定性、类别分布。

---

## 9. 执行顺序

```text
Step 1：完成 A-D 官方权重校准
Step 2：实现 KNS-v1 sampler，只接 test pipeline
Step 3：跑 E1-E5，NTU60 XSub joint
Step 4：做样本级分析
Step 5：若 E 组有效，跑 F1-F5，FineGYM joint
Step 6：若 E/F 至少一个有效，再补 G 组 limb/fusion
Step 7：只在 test-time 结果站稳后，再考虑训练阶段采样
```

这一版计划的原则是：**先证明采样策略本身有信息价值，再谈训练、融合和更复杂门控。**

[1]: https://github.com/kennymckormick/pyskl/blob/main/configs/posec3d/README.md "pyskl/configs/posec3d/README.md at main · kennymckormick/pyskl · GitHub"
