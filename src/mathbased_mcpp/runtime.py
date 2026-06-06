from __future__ import annotations

import os


def configure_runtime(cpu_threads: int | None = None) -> int:
    """Configure conservative CPU threading for local training runs."""
    threads = _clamp_cpu_threads(cpu_threads)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    return threads


def _clamp_cpu_threads(cpu_threads: int | None) -> int:
    if cpu_threads is None:
        cpu_threads = 4
    return min(max(int(cpu_threads), 1), 4)
