"""源码目录的导入桥接。

项目采用 ``src/`` 布局，但开发时常直接在仓库根目录运行
``python -m mathbased_mcpp``。这个轻量包将真正实现目录加入搜索路径，
从而不需要先执行安装命令。
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "mathbased_mcpp"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))
