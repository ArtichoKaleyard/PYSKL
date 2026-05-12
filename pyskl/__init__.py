# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from .utils.mmcv_compat import digit_version, install_mmcv_legacy_shims

from .version import __version__

install_mmcv_legacy_shims()

mmcv_minimum_version = '2.1.0'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

assert (digit_version(mmcv_minimum_version) <= mmcv_version
        <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

__all__ = ['__version__']
