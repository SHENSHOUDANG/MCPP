from __future__ import annotations

from pathlib import Path
import shutil
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.port_inspection import load_inspection_tasks, load_port_grid
from mathbased_mcpp.port_inspection.render import render_port_inspection_map


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render the Dalian port water inspection prototype scenario.")
    parser.add_argument("--config", default="configs/port_dalian_water_v0.toml")
    args = parser.parse_args()

    config = _load_config(args.config)
    grid = load_port_grid(config["grid_path"])
    tasks = load_inspection_tasks(config["tasks_path"], grid)
    output_dir = Path(config.get("output_dir", "outputs/port_inspection/dalian_port_v0"))
    output_path = output_dir / "scenario_preview.png"
    render_port_inspection_map(grid, tasks, output_path)

    data_preview = Path(config["grid_path"]).parent / "preview.png"
    data_preview.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_path, data_preview)
    print(f"scenario_preview={output_path}")
    print(f"data_preview={data_preview}")


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
