from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "mathbased_mcpp"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))
