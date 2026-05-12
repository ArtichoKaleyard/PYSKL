"""检查 PYSKL 现代 uv 环境是否可用。

该脚本聚焦环境本身，而不是旧 PYSKL 源码到 OpenMMLab 2.x 的迁移。
它会验证 Python、CUDA toolkit、PyTorch CUDA、mmcv CUDA op 以及
OpenMMLab 上层包是否能在当前 `uv` 环境里正常工作。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CheckResult:
    """单项检查结果。

    Args:
        name: 检查项名称。
        detail: 成功时展示的关键信息。
    """

    name: str
    detail: str


def run_command(command: list[str]) -> str:
    """运行只读系统命令并返回标准输出。

    Args:
        command: 命令及其参数。

    Returns:
        命令标准输出，已去掉首尾空白。

    Raises:
        RuntimeError: 命令不可执行或返回非零状态。
    """

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"命令不存在: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"命令失败: {' '.join(command)}\n{stderr}") from exc
    return completed.stdout.strip()


def require(condition: bool, message: str) -> None:
    """断言检查条件成立。

    Args:
        condition: 需要满足的条件。
        message: 条件不成立时的诊断信息。

    Raises:
        RuntimeError: 条件不成立。
    """

    if not condition:
        raise RuntimeError(message)


def check_python() -> CheckResult:
    """检查当前解释器版本是否为 Python 3.11。"""

    version = sys.version_info
    require(version.major == 3 and version.minor == 11, "当前环境不是 Python 3.11")
    return CheckResult(
        name="python",
        detail=f"{sys.version.split()[0]} ({sys.executable})",
    )


def check_cuda_toolkit() -> CheckResult:
    """检查系统 CUDA toolkit 入口是否为 12.8。"""

    cuda_home = Path("/usr/local/cuda")
    resolved = cuda_home.resolve()
    nvcc_output = run_command([str(cuda_home / "bin" / "nvcc"), "--version"])
    require("release 12.8" in nvcc_output, "nvcc 不是 CUDA 12.8")
    return CheckResult(name="cuda-toolkit", detail=str(resolved))


def check_torch_cuda() -> CheckResult:
    """检查 PyTorch 2.7 cu128 与 GPU 张量计算。"""

    import torch

    require(torch.__version__.startswith("2.7."), f"PyTorch 版本不是 2.7.x: {torch.__version__}")
    require(torch.version.cuda == "12.8", f"PyTorch CUDA 版本不是 12.8: {torch.version.cuda}")
    require(torch.cuda.is_available(), "PyTorch 看不到 CUDA 设备")

    tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    value = float((tensor * 2).sum().item())
    require(value == 56.0, f"CUDA 张量计算结果异常: {value}")

    device_name = torch.cuda.get_device_name(0)
    return CheckResult(
        name="torch-cuda",
        detail=f"{torch.__version__}, cuda={torch.version.cuda}, device={device_name}",
    )


def check_mmcv_ops() -> CheckResult:
    """检查 mmcv 2.1.0 及其 CUDA/C++ 扩展是否可用。"""

    import mmcv
    import torch
    from mmcv.ops import nms

    require(mmcv.__version__ == "2.1.0", f"mmcv 版本不是 2.1.0: {mmcv.__version__}")
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [30.0, 30.0, 40.0, 40.0]],
        device="cuda",
    )
    scores = torch.tensor([0.95, 0.90, 0.80], device="cuda")
    dets, keep = nms(boxes, scores, iou_threshold=0.5)
    require(int(keep.numel()) == 2, f"mmcv.ops.nms 返回数量异常: {keep.numel()}")
    return CheckResult(name="mmcv-ops", detail=f"mmcv={mmcv.__version__}, nms_keep={keep.tolist()}")


def check_openmmlab_packages() -> CheckResult:
    """检查 OpenMMLab 上层包的版本带。"""

    expected_prefixes = {
        "mmengine": "0.10.",
        "mmdet": "3.2.",
        "mmpose": "1.3.",
        "mmaction": "1.2.",
    }
    versions: list[str] = []
    for module_name, version_prefix in expected_prefixes.items():
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "")
        require(version.startswith(version_prefix), f"{module_name} 版本异常: {version}")
        versions.append(f"{module_name}={version}")
    return CheckResult(name="openmmlab", detail=", ".join(versions))


def check_pyskl_legacy_boundary() -> CheckResult:
    """记录 PYSKL 源码当前的预期兼容性边界。

    当前环境安装的是 OpenMMLab 2.x / mmcv 2.x。PYSKL 源码仍依赖
    `mmcv.digit_version`、`mmcv.runner` 等 1.x API，因此顶层导入失败
    属于源码迁移问题，不属于环境安装失败。
    """

    try:
        importlib.import_module("pyskl")
    except ImportError as exc:
        message = str(exc)
        expected_fragments = ("digit_version", "mmcv.runner", "mmcv.parallel", "Config")
        require(
            any(fragment in message for fragment in expected_fragments),
            f"PYSKL 导入失败，但不是已知 mmcv 旧 API 断点: {message}",
        )
        return CheckResult(name="pyskl-source", detail=f"expected legacy API boundary: {message}")
    return CheckResult(name="pyskl-source", detail="pyskl import succeeded")


def main() -> int:
    """执行全部环境检查。

    Returns:
        进程退出码。全部检查通过返回 0。
    """

    checks: list[Callable[[], CheckResult]] = [
        check_python,
        check_cuda_toolkit,
        check_torch_cuda,
        check_mmcv_ops,
        check_openmmlab_packages,
        check_pyskl_legacy_boundary,
    ]
    for check in checks:
        result = check()
        print(f"[PASS] {result.name}: {result.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
