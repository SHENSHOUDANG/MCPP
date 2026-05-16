from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import random
import os
import socket
import struct
import time
from typing import Iterable

from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_dir(run_root: str | Path) -> Path:
    root = Path(run_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_metrics(path: str | Path, rows: Iterable[dict[str, float | int]]) -> None:
    path = Path(path)
    rows = list(rows)
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def make_tensorboard_writer(run_path: str | Path, subdir: str = "tensorboard"):
    try:
        from tensorboard.compat.proto import event_pb2, summary_pb2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("TensorBoard is not installed. Install the 'tensorboard' package to enable live logs.") from exc
    log_dir = Path(run_path) / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    return _SimpleTensorBoardWriter(log_dir, event_pb2, summary_pb2)


def write_tensorboard_rows(writer: object, prefix: str, rows: Iterable[dict[str, float | int]]) -> None:
    for row in rows:
        step = int(row.get("episode", row.get("steps", 0)))
        for key, value in row.items():
            if key == "episode":
                continue
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{prefix}/{key}", float(value), step)
    writer.flush()


class _SimpleTensorBoardWriter:
    """Small scalar-only TensorBoard event writer that uses Python file IO."""

    def __init__(self, log_dir: Path, event_pb2: object, summary_pb2: object) -> None:
        self._event_pb2 = event_pb2
        self._summary_pb2 = summary_pb2
        filename = f"events.out.tfevents.{int(time.time())}.{socket.gethostname()}.{os.getpid()}"
        self._handle = (log_dir / filename).open("wb")
        self._write_event(self._event_pb2.Event(wall_time=time.time(), file_version="brain.Event:2"))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
        )
        self._write_event(self._event_pb2.Event(wall_time=time.time(), step=int(step), summary=summary))

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def _write_event(self, event: object) -> None:
        payload = event.SerializeToString()
        header = struct.pack("<Q", len(payload))
        self._handle.write(header)
        self._handle.write(struct.pack("<I", _masked_crc32c(header)))
        self._handle.write(payload)
        self._handle.write(struct.pack("<I", _masked_crc32c(payload)))


_CRC32C_TABLE: list[int] | None = None


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _crc32c(data: bytes) -> int:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        table = []
        for value in range(256):
            crc = value
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0x82F63B78
                else:
                    crc >>= 1
            table.append(crc & 0xFFFFFFFF)
        _CRC32C_TABLE = table

    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF

