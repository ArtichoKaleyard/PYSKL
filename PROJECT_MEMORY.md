## 2026-05-12 - PYSKL 的 uv 现代环境基线
- Context: 需要在 `WSL2-TiPlus` 上为 PYSKL 搭建可用环境，目标固定为 Python 3.11、CUDA 12.8、PyTorch 2.7，并参考 `/home/atk/PyCharm/OpenMMLab` 的 OpenMMLab 编译经验。
- Decision: 新增项目级 `pyproject.toml` 作为现代 uv 环境入口，不改旧 `requirements.txt` / `setup.py`；依赖锁定为 `torch==2.7.*` cu128、`mmcv==2.1.0` 本地 editable、`mmengine 0.10.x`、`mmdet 3.2.x`、`mmpose 1.3.x`、`mmaction2 1.2.x`。
- Why: 原项目声明仍是 Python 3.7 / CUDA 11.3 / OpenMMLab 1.x 时代的依赖，直接沿用会与 Python 3.11 和 PyTorch 2.7 冲突；OpenMMLab 参考环境已验证 mmcv 2.1.0 可在 CUDA 12.8 下由 uv 编译。
- Action/Command: 使用 `uv sync --group dev` 创建 `.venv`；PyPI 默认源通过全局 `~/.config/uv/uv.toml` 设为 `https://pypi.tuna.tsinghua.edu.cn/simple`，项目 `pyproject.toml` 不保留项目级镜像；PyTorch Linux wheel 继续显式使用 `https://download.pytorch.org/whl/cu128`；新增 `scripts/check_env.py` 作为可复跑的环境冒烟测试。
- Verification: `uv run python scripts/check_env.py` 通过；Python 为 3.11.14，`torch==2.7.1+cu128`、`torch.version.cuda == 12.8`、`torch.cuda.is_available() == True`；真实 GPU 张量计算和 `mmcv.ops.nms` CUDA 调用通过；OpenMMLab 上层包版本为 `mmengine 0.10.7 / mmdet 3.2.0 / mmpose 1.3.2 / mmaction2 1.2.0`。
- Follow-up: PYSKL 源码仍使用 `mmcv 1.x` 的 `digit_version`、`Config`、`runner`、`parallel` 等 API；当前环境只保证现代栈搭建与 mmcv CUDA op 可用，不代表源码已完成 mmcv 2.x 迁移。`uv sync --check` 目前会反复提示同版本 `decord==0.6.0` 需要重装，但运行时导入和核心依赖版本验证正常。

## 2026-05-12 - modern 分支实验链路迁移
- Context: 按 NTU60 xsub 的 A-G 实验计划迁移关键链路，不做全项目 OpenMMLab 2.x 重构。
- Decision: 在 `modern` 分支保留旧入口对照，新增 `tools/modern/` 资产准备、推理、训练和融合入口；通过 `pyskl.utils.mmcv_compat` 给旧模块补足实验链路需要的 mmcv 1.x shim。
- Why: PoseC3D/MS-G3D 的模型和 pipeline 大部分仍可复用，真正需要先打通的是配置解析、注册、batch collate、checkpoint、score export 和短跑训练，不应先重写所有 demo/notebook/data tools。
- Action/Command: 新增 NTU60 xsub modern configs，A/B/D 官方权重测试配置、C 融合脚本、E 原始采样训练配置、F/G `MotionAwareUniformSampleFrames` 自定义采样配置。
- Verification: `scripts/check_env.py` 通过；所有 modern config 可解析并能构建模型；`prepare_assets.py --check` 正确列出缺失资产；`tools/modern/test.py`、`tools/modern/train.py`、`tools/modern/fuse_scores.py` 已用 `/tmp` 假 NTU annotation、随机 checkpoint 和真实 CUDA 跑通。
- Follow-up: A-D 官方权重评估已经完成，结果见 `tools/modern/RESULTS.md`；后续进入 E/F 采样训练前，优先沿用 `tools/modern/test.py` 的进度日志和 batch/workers 覆盖机制做短跑基准。

## 2026-05-12 - NTU60 xsub 官方权重复测结果
- Context: 需要确认 modern 分支在 Python 3.11 / CUDA 12.8 / PyTorch 2.7 下，能否复现 PoseC3D 和 MS-G3D 的官方 NTU60 xsub 测试流程。
- Decision: 使用 OpenMMLab 官方 `ntu60_hrnet.pkl`、PoseC3D joint/limb 官方权重和 MS-G3D HRNet joint 官方权重完成 A-D；将原始 score 保留在 ignored 的 `work_dirs/modern/scores/`，将长期可追踪结果写入 `tools/modern/RESULTS.md`。
- Why: `work_dirs/`、`data/`、`checkpoints/` 都是本地大文件/产物目录，不应进入 git；结果文档能保留官方对照、运行参数和配置一致性结论。
- Action/Command: A/B 使用 `tools/modern/test.py --videos-per-gpu 2 --workers-per-gpu 4`；D 使用 `--videos-per-gpu 4 --workers-per-gpu 0`；C 使用 `tools/modern/fuse_scores.py` 计算 joint+limb 融合。
- Verification: A top1=0.9373，B top1=0.9338，C 1:1 top1=0.9406，D top1=0.9264；三个 score 文件均为 16487 samples；与官方 top-1 差异均小于 0.1 个百分点。
- Follow-up: 进入 E/F 训练时不要默认沿用测试阶段 batch；先用 `--max-epochs 1` 和验证导出做短跑，确认显存、吞吐和 checkpoint/score 写入路径。

## 2026-05-14 - KNS-v1 test-time sampling 边界与结果
- Context: 根据 `tools/modern/EXPERIMENT_PLAN_KNS.md` 将 E 组从训练采样改为 test-time sampling，使用 PoseC3D joint 官方权重比较 1-clip uniform、1-clip KNS、2-clip uniform、uniform+KNS 和 10-clip uniform。
- Decision: 新增 `KnsSampleFrames` 只替换 `PoseDecode` 前的 temporal sampler；官方 test-time `GeneratePoseTarget(double=True)`、backbone 和 checkpoint 保持不变。
- Why: 这一轮要先验证采样策略本身是否有信息价值，避免训练、limb、fusion 或 heatmap 改动搅混结论。
- Action/Command: E1-E4 使用 `tools/modern/test.py --videos-per-gpu 2 --workers-per-gpu 4` 正式评估；E5 复用 A 组 `posec3d_joint.pkl`；`export_sampling_metadata.py` 导出 sampled indices 和 KNS 峰值；`analyze_kns_scores.py` 生成正误迁移和 10-clip 收益重合度。
- Verification: E1/E2/E3/E4/E5 top1 分别为 0.9358/0.9344/0.9364/0.9369/0.9373；E2 相比 E1 负迁移，E4 相比 E3 小幅提升；每个 score 和 sampling 文件均为 16487 samples。
- Follow-up: KNS sampler 必须允许短视频粗分区重复补帧；首次 E2 卡住的根因是分区长度小于 3 时 `_fill_points` 寻找不同未占用帧进入死循环。后续继续优化 KNS 时优先分析 E2/E4 的 breaks，再决定是否进入 FineGYM 或 limb/fusion。
