"""运行时环境兼容设置。

训练依赖 PyTorch/NumPy 等包含本地动态库的包。在部分 Windows conda
环境中，OpenMP 动态库可能被重复加载，因此在导入计算库前集中设置兼容项。
"""

from __future__ import annotations

import os


def configure_runtime() -> None:
    """避免 Windows conda CPU 环境因重复 OpenMP DLL 而直接中止。"""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
