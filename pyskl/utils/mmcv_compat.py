"""Compatibility shims for running legacy PYSKL code on mmcv 2.x.

PYSKL was written against the OpenMMLab 1.x stack.  The modern experiment
branch keeps the old source layout but installs mmcv 2.x and mmengine, so this
module fills the small subset of removed mmcv symbols that the NTU60 xsub
PoseC3D/MS-G3D experiment path still imports.
"""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from mmengine.config import Config
from mmengine.dist import get_dist_info as mmengine_get_dist_info
from mmengine.dist import init_dist
from mmengine.fileio import FileClient, dump, load
from mmengine.logging import MMLogger, print_log
from mmengine.model import constant_init, kaiming_init, normal_init
from mmengine.registry import Registry, build_from_cfg
from mmengine.runner import load_checkpoint
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.utils import digit_version, get_git_hash, is_list_of, is_seq_of, is_str, is_tuple_of
from mmengine.utils.dl_utils import collect_env
from torch.nn.modules.batchnorm import _BatchNorm


class DataContainer:
    """Small stand-in for ``mmcv.parallel.DataContainer``.

    The modern experiment configs use empty ``meta_keys`` for model inputs, but
    retaining this wrapper keeps older pipelines importable and lets metadata
    pass through unchanged when present.
    """

    def __init__(self, data: Any, stack: bool = False, padding_value: int = 0, cpu_only: bool = False):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data!r})"


def _collate_value(values: list[Any]) -> Any:
    """Collate values emitted by PYSKL pipelines."""

    first = values[0]
    if isinstance(first, DataContainer):
        payload = [item.data for item in values]
        return payload if first.cpu_only else _collate_value(payload)
    if isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    if isinstance(first, Mapping):
        return {key: _collate_value([value[key] for value in values]) for key in first}
    if isinstance(first, tuple):
        return tuple(_collate_value([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        if not first:
            return values
        return [_collate_value([value[index] for value in values]) for index in range(len(first))]
    if isinstance(first, (int, bool)):
        return torch.LongTensor(values)
    if isinstance(first, float):
        return torch.FloatTensor(values)
    return values


def collate(batch: Sequence[Any], samples_per_gpu: int = 1) -> Any:
    """Collate a mini-batch.

    Args:
        batch: Samples returned from a dataset.
        samples_per_gpu: Kept for API compatibility with mmcv 1.x.

    Returns:
        Collated batch suitable for PYSKL recognizers.
    """

    del samples_per_gpu
    return _collate_value(list(batch))


def scatter(inputs: Any, target_gpus: Sequence[torch.device | int]) -> list[Any]:
    """Move a nested batch to the first target GPU."""

    device = torch.device(target_gpus[0])

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    return [move(inputs)]


def get_dist_info() -> tuple[int, int]:
    """Return distributed rank and world size, defaulting to single process."""

    if dist.is_available() and dist.is_initialized():
        return mmengine_get_dist_info()
    return 0, 1


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python process random seed for torch and CUDA."""

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: str | None = None, log_level: int | str = logging.INFO) -> MMLogger:
    """Return a logger compatible with the old ``mmcv.utils.get_logger`` API."""

    logger = MMLogger.get_instance(name, log_file=log_file, log_level=log_level)
    return logger


def mkdir_or_exist(path: str | Path) -> None:
    """Create a directory if it does not exist."""

    Path(path).mkdir(parents=True, exist_ok=True)


def _build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the optimizer types used by the target experiment configs."""

    cfg = cfg.copy()
    optimizer_type = cfg.pop("type")
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), **cfg)
    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), **cfg)
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), **cfg)
    raise KeyError(f"Unsupported optimizer type: {optimizer_type}")


class DistEvalHook:
    """Import-compatible placeholder for the legacy mmcv evaluation hook."""

    by_epoch = True
    start = None

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    def every_n_epochs(self, runner: Any, n: int) -> bool:
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def _should_evaluate(self, runner: Any) -> bool:
        return True


def install_mmcv_legacy_shims() -> None:
    """Install legacy mmcv modules into ``sys.modules``."""

    import mmcv
    import mmcv.cnn as mmcv_cnn

    try:
        from mmaction.registry import MODELS as MMACTION_MODELS
    except Exception:
        MMACTION_MODELS = Registry("model")

    mmcv.Config = Config
    mmcv.load = load
    mmcv.dump = dump
    mmcv.is_str = is_str
    mmcv.is_list_of = is_list_of
    mmcv.is_tuple_of = is_tuple_of
    mmcv.is_seq_of = is_seq_of
    mmcv.mkdir_or_exist = mkdir_or_exist
    mmcv.digit_version = digit_version

    mmcv_cnn.MODELS = getattr(mmcv_cnn, "MODELS", MMACTION_MODELS)
    mmcv_cnn.constant_init = constant_init
    mmcv_cnn.kaiming_init = kaiming_init
    mmcv_cnn.normal_init = normal_init

    runner = types.ModuleType("mmcv.runner")
    runner.get_dist_info = get_dist_info
    runner.init_dist = init_dist
    runner.set_random_seed = set_random_seed
    runner.load_checkpoint = load_checkpoint
    runner._load_checkpoint = _load_checkpoint
    runner.build_optimizer = _build_optimizer
    runner.DistEvalHook = DistEvalHook
    runner.DistSamplerSeedHook = object
    runner.EpochBasedRunner = object
    runner.OptimizerHook = object

    parallel = types.ModuleType("mmcv.parallel")
    parallel.DataContainer = DataContainer
    parallel.collate = collate
    parallel.scatter = scatter
    parallel.MMDistributedDataParallel = torch.nn.parallel.DistributedDataParallel

    engine = types.ModuleType("mmcv.engine")
    engine.multi_gpu_test = None
    engine.single_gpu_test = None

    utils = types.ModuleType("mmcv.utils")
    utils.Registry = Registry
    utils.build_from_cfg = build_from_cfg
    utils.digit_version = digit_version
    utils.collect_env = collect_env
    utils.get_git_hash = get_git_hash
    utils.get_logger = get_logger
    utils.print_log = print_log
    utils._BatchNorm = _BatchNorm

    fileio = types.ModuleType("mmcv.fileio")
    fileio.FileClient = FileClient
    fileio.load = load
    fileio.dump = dump

    fileio_io = types.ModuleType("mmcv.fileio.io")
    fileio_io.file_handlers = {"pkl": object(), "pickle": object(), "json": object(), "yaml": object(), "yml": object()}

    sys.modules["mmcv.runner"] = runner
    sys.modules["mmcv.parallel"] = parallel
    sys.modules["mmcv.engine"] = engine
    sys.modules["mmcv.utils"] = utils
    sys.modules["mmcv.fileio"] = fileio
    sys.modules["mmcv.fileio.io"] = fileio_io
