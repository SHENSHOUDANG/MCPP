from __future__ import annotations

import os


def configure_runtime() -> None:
    """Keep Windows conda CPU stacks from aborting on duplicate OpenMP DLLs."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
