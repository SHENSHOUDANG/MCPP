"""命令行模块入口。

既支持安装包后的 ``python -m mathbased_mcpp``，也兼容直接执行本文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # 直接运行源码文件时，Python 不知道同级包的位置，需要手动加入 src。
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from mathbased_mcpp.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    main()
