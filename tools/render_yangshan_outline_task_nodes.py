from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.port_inspection import load_port_grid


GRID_PATH = Path("data/ports/shanghai_yangshan_osm_v1/shanghai_yangshan_osm_v1_grid.json")
BACKGROUND_PATH = Path("outputs/real_map_tiles/yangshan_arcgis_world_imagery_export_hires.png")
OUTPUT_DIR = Path("outputs/port_inspection/shanghai_yangshan_osm_v1/task_node_design")

# The local ArcGIS imagery export is this smaller view. Do not project the full
# OSM grid extent onto it; that was the source of the previous systematic offset.
IMAGE_BBOX = {
    "south": 30.600,
    "west": 122.010,
    "north": 30.655,
    "east": 122.100,
}

NODE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "node_id": "A01",
        "group_code": "A",
        "xy": (0.342, 0.247),
        "object_type": "桥区通航标识",
        "service": "桥区通航保障设施位置、外观与周边障碍快速筛查",
    },
    {
        "node_id": "A02",
        "group_code": "A",
        "xy": (0.385, 0.326),
        "object_type": "桥涵标/桥墩邻近警示标志",
        "service": "桥涵标识状态与桥区水面异常抵近确认",
    },
    {
        "node_id": "A03",
        "group_code": "A",
        "xy": (0.558, 0.244),
        "object_type": "港池边界灯标或专用标志",
        "service": "港池边界标识位置、倾斜、外观与遮挡筛查",
    },
    {
        "node_id": "A04",
        "group_code": "A",
        "xy": (0.738, 0.245),
        "object_type": "围堰/口门附近灯标",
        "service": "围堰转角通航保障标志及附近障碍筛查",
    },
    {
        "node_id": "A05",
        "group_code": "A",
        "xy": (0.215, 0.557),
        "object_type": "开阔水域浮标巡查预留点",
        "service": "航道浮标或临时专用标志的广域发现与位置复核",
    },
    {
        "node_id": "A06",
        "group_code": "A",
        "xy": (0.421, 0.675),
        "object_type": "进出港航路标识巡查点",
        "service": "进出港航路助航标识与周边碍航物筛查",
    },
    {
        "node_id": "B01",
        "group_code": "B",
        "xy": (0.126, 0.081),
        "object_type": "北侧码头前沿",
        "service": "码头前沿水线附近明显异常筛查与图像采集",
    },
    {
        "node_id": "B02",
        "group_code": "B",
        "xy": (0.247, 0.103),
        "object_type": "桥接岸段码头前沿",
        "service": "岸壁、水线和靠泊设备外观筛查",
    },
    {
        "node_id": "B03",
        "group_code": "B",
        "xy": (0.324, 0.437),
        "object_type": "西侧岛体码头前沿",
        "service": "泊位前沿、水线和桩基附近异常筛查",
    },
    {
        "node_id": "B04",
        "group_code": "B",
        "xy": (0.377, 0.513),
        "object_type": "西南侧集装箱泊位",
        "service": "靠泊区水侧图像/空间数据采集",
    },
    {
        "node_id": "B05",
        "group_code": "B",
        "xy": (0.438, 0.579),
        "object_type": "中段码头前沿",
        "service": "码头前沿连续水线异常筛查",
    },
    {
        "node_id": "B06",
        "group_code": "B",
        "xy": (0.506, 0.642),
        "object_type": "主作业区泊位前沿",
        "service": "岸壁、水线、靠泊区障碍物巡检",
    },
    {
        "node_id": "B07",
        "group_code": "B",
        "xy": (0.585, 0.704),
        "object_type": "东南延伸码头前沿",
        "service": "长岸线水侧异常筛查与近距确认",
    },
    {
        "node_id": "B08",
        "group_code": "B",
        "xy": (0.670, 0.785),
        "object_type": "东南集装箱泊位",
        "service": "泊位前沿和水线附近图像采集",
    },
    {
        "node_id": "B09",
        "group_code": "B",
        "xy": (0.807, 0.865),
        "object_type": "东南端护岸/港池口门",
        "service": "护岸转角、端部水域与口门异常筛查",
    },
    {
        "node_id": "B10",
        "group_code": "B",
        "xy": (0.842, 0.392),
        "object_type": "东北围堰/护岸",
        "service": "围堰边界、水线和疑似冲刷异常筛查",
    },
    {
        "node_id": "B11",
        "group_code": "B",
        "xy": (0.930, 0.573),
        "object_type": "东侧护岸与岸壁",
        "service": "护岸连续边界与水线附近异常筛查",
    },
    {
        "node_id": "B12",
        "group_code": "B",
        "xy": (0.486, 0.424),
        "object_type": "内侧港池岸线",
        "service": "港池内缘岸线、水线和小型结构物筛查",
    },
    {
        "node_id": "C01",
        "group_code": "C",
        "xy": (0.104, 0.522),
        "object_type": "开阔水域漂浮物发现点",
        "service": "UAV广域发现，USV抵近确认位置、尺度和类别",
    },
    {
        "node_id": "C02",
        "group_code": "C",
        "xy": (0.219, 0.626),
        "object_type": "西南航路漂浮物/疑似碍航物",
        "service": "航路邻近异常粗定位与抵近复核",
    },
    {
        "node_id": "C03",
        "group_code": "C",
        "xy": (0.354, 0.745),
        "object_type": "进出港水域疑似漂浮物",
        "service": "UAV发现后由USV确认位置和类别",
    },
    {
        "node_id": "C04",
        "group_code": "C",
        "xy": (0.545, 0.762),
        "object_type": "泊位外侧碍航异常",
        "service": "靠泊区外侧漂浮物、沉没物或搁浅物确认",
    },
    {
        "node_id": "C05",
        "group_code": "C",
        "xy": (0.764, 0.213),
        "object_type": "东北水域施工侵入/临时障碍",
        "service": "临时施工侵入和围堰附近碍航异常筛查",
    },
    {
        "node_id": "C06",
        "group_code": "C",
        "xy": (0.867, 0.217),
        "object_type": "北侧航路疑似异常",
        "service": "航路附近浮标、船迹或漂浮异常复核预留点",
    },
    {
        "node_id": "C07",
        "group_code": "C",
        "xy": (0.948, 0.890),
        "object_type": "东南口门漂浮物/碍航物",
        "service": "港池口门附近异常发现与抵近确认",
    },
]

GROUP_NAMES = {
    "A": "航标与通航保障设施",
    "B": "水侧基础设施",
    "C": "通航异常",
}

GROUP_RISK = {"A": 2, "B": 2, "C": 3}

OUTLINE_GUIDES = [
    {
        "label": "北侧码头前沿",
        "points": [(0.030, 0.083), (0.118, 0.087), (0.215, 0.096), (0.300, 0.135)],
    },
    {
        "label": "桥区与西侧岛体岸线",
        "points": [(0.302, 0.312), (0.346, 0.344), (0.396, 0.403), (0.452, 0.472)],
    },
    {
        "label": "主码头前沿",
        "points": [
            (0.314, 0.423),
            (0.368, 0.497),
            (0.433, 0.568),
            (0.507, 0.636),
            (0.593, 0.700),
            (0.681, 0.770),
            (0.802, 0.858),
        ],
    },
    {
        "label": "东北围堰与护岸",
        "points": [(0.625, 0.257), (0.700, 0.262), (0.800, 0.260), (0.835, 0.210), (0.905, 0.255)],
    },
    {
        "label": "东侧水线",
        "points": [(0.838, 0.392), (0.918, 0.482), (0.958, 0.568), (0.965, 0.692)],
    },
    {
        "label": "港池口门与东南端",
        "points": [(0.765, 0.905), (0.828, 0.881), (0.870, 0.832), (0.910, 0.775), (0.953, 0.730)],
    },
]

WATER_GUIDES = [
    [(0.010, 0.548), (0.170, 0.615), (0.345, 0.718), (0.585, 0.813)],
    [(0.585, 0.225), (0.710, 0.210), (0.865, 0.193), (0.995, 0.155)],
]


def _pick_chinese_font() -> str:
    available = {font.name for font in fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "STSong", "FangSong"):
        if candidate in available:
            return candidate
    return "sans-serif"


_CN_FONT = FontProperties(family=_pick_chinese_font())
plt.rcParams["font.family"] = _pick_chinese_font()
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    grid = load_port_grid(GRID_PATH)
    image = _background(BACKGROUND_PATH)
    nodes = _design_nodes(grid)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "yangshan_satellite_outline_task_nodes.png"
    csv_path = OUTPUT_DIR / "yangshan_candidate_task_nodes.csv"
    json_path = OUTPUT_DIR / "yangshan_candidate_task_nodes.json"
    _write_nodes(nodes, csv_path, json_path)
    _render(image, nodes, png)
    print(png.resolve())
    print(csv_path.resolve())
    print(json_path.resolve())


def _background(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageEnhance.Color(image).enhance(0.82)
    image = ImageEnhance.Contrast(image).enhance(0.98)
    image = ImageEnhance.Brightness(image).enhance(1.03)
    return image


def _design_nodes(grid) -> list[dict[str, Any]]:
    nodes = []
    for definition in NODE_DEFINITIONS:
        x_norm, y_norm = definition["xy"]
        lat, lon = _norm_to_latlon(x_norm, y_norm)
        target_cell = _latlon_to_cell(lat, lon, grid)
        row, col = _nearest_free_cell(target_cell, grid.free_cell_set)
        group_code = definition["group_code"]
        nodes.append(
            {
                "node_id": definition["node_id"],
                "group_code": group_code,
                "group_name": GROUP_NAMES[group_code],
                "object_type": definition["object_type"],
                "service": definition["service"],
                "row": row,
                "col": col,
                "target_row": target_cell[0],
                "target_col": target_cell[1],
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "x_norm": round(x_norm, 4),
                "y_norm": round(y_norm, 4),
                "risk": GROUP_RISK[group_code],
                "model_stage": "UAV筛查 -> 按规则触发USV复核",
                "grid_cell_role": "row/col为最近可达水域接近单元，target_row/target_col为影像目标所在单元",
                "source": "基于卫星底图人工核验的候选点",
            }
        )
    return nodes


def _norm_to_latlon(x_norm: float, y_norm: float) -> tuple[float, float]:
    lon = IMAGE_BBOX["west"] + x_norm * (IMAGE_BBOX["east"] - IMAGE_BBOX["west"])
    lat = IMAGE_BBOX["north"] - y_norm * (IMAGE_BBOX["north"] - IMAGE_BBOX["south"])
    return lat, lon


def _latlon_to_cell(lat: float, lon: float, grid) -> tuple[int, int]:
    bbox = grid.metadata["bbox"]
    row = int((float(bbox["north"]) - lat) / (float(bbox["north"]) - float(bbox["south"])) * grid.height)
    col = int((lon - float(bbox["west"])) / (float(bbox["east"]) - float(bbox["west"])) * grid.width)
    row = max(0, min(grid.height - 1, row))
    col = max(0, min(grid.width - 1, col))
    return row, col


def _nearest_free_cell(cell: tuple[int, int], free_cells: set[tuple[int, int]]) -> tuple[int, int]:
    if cell in free_cells:
        return cell
    return min(free_cells, key=lambda item: (abs(item[0] - cell[0]) + abs(item[1] - cell[1]), item[0], item[1]))


def _norm_to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    return point[0] * width, point[1] * height


def _render(image: Image.Image, nodes: list[dict[str, Any]], output: Path) -> None:
    width, height = image.size
    fig, ax = plt.subplots(figsize=(18, 11), constrained_layout=True)
    ax.imshow(image)
    ax.set_axis_off()

    _draw_guides(ax, width, height)
    _draw_nodes(ax, width, height, nodes)
    _draw_title(ax, nodes)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _draw_guides(ax, width: int, height: int) -> None:
    for guide in OUTLINE_GUIDES:
        pixels = [_norm_to_pixel(point, width, height) for point in guide["points"]]
        xs, ys = zip(*pixels)
        ax.plot(
            xs,
            ys,
            color="#f8fafc",
            linewidth=2.1,
            alpha=0.82,
            zorder=7,
            path_effects=[patheffects.withStroke(linewidth=4.2, foreground="#020617", alpha=0.58)],
        )
    for line in WATER_GUIDES:
        pixels = [_norm_to_pixel(point, width, height) for point in line]
        xs, ys = zip(*pixels)
        ax.plot(
            xs,
            ys,
            color="#facc15",
            linewidth=1.7,
            alpha=0.72,
            linestyle=(0, (5, 5)),
            zorder=6,
            path_effects=[patheffects.withStroke(linewidth=3.2, foreground="#020617", alpha=0.45)],
        )


def _draw_nodes(ax, width: int, height: int, nodes: list[dict[str, Any]]) -> None:
    styles = {
        "A": ("^", "#38bdf8", "#0f172a"),
        "B": ("s", "#fbbf24", "#1f2937"),
        "C": ("o", "#fb7185", "#111827"),
    }
    offsets = {"A": (10, -10), "B": (9, 10), "C": (10, -10)}
    for node in nodes:
        group = node["group_code"]
        marker, color, edge = styles[group]
        x, y = _norm_to_pixel((float(node["x_norm"]), float(node["y_norm"])), width, height)
        ax.scatter(x, y, marker=marker, s=82, color=color, edgecolors=edge, linewidths=0.8, zorder=20)
        dx, dy = offsets[group]
        ax.text(
            x + dx,
            y + dy,
            node["node_id"],
            fontsize=8.7,
            color="#ffffff",
            weight="bold",
            bbox={"facecolor": "#020617", "edgecolor": "none", "alpha": 0.68, "boxstyle": "round,pad=0.18"},
            path_effects=[patheffects.withStroke(linewidth=1.2, foreground="#020617")],
            zorder=22,
        )


def _draw_title(ax, nodes: list[dict[str, Any]]) -> None:
    counts = {code: sum(1 for node in nodes if node["group_code"] == code) for code in ("A", "B", "C")}
    title = (
        "洋山港卫星底图轮廓草图与UAV-USV候选任务节点\n"
        f"A 航标与通航保障 {counts['A']} | B 水侧基础设施 {counts['B']} | C 通航异常 {counts['C']}"
    )
    ax.text(
        0.015,
        0.025,
        title + "\n白线为按影像人工描绘的岸线/码头前沿参考；黄虚线为任务布设用水域走廊参考，非航海图。",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#f8fafc",
        bbox={"facecolor": "#020617", "edgecolor": "#e2e8f0", "linewidth": 0.4, "alpha": 0.70, "boxstyle": "round,pad=0.44"},
        fontproperties=_CN_FONT,
        zorder=40,
    )
    legend = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#38bdf8", markeredgecolor="#0f172a", label="A 航标与通航保障设施"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#fbbf24", markeredgecolor="#1f2937", label="B 水侧基础设施"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fb7185", markeredgecolor="#111827", label="C 通航异常"),
        Line2D([0], [0], color="#f8fafc", lw=2.1, label="影像描绘岸线/码头前沿"),
        Line2D([0], [0], color="#facc15", lw=1.7, linestyle=(0, (5, 5)), label="任务布设水域走廊"),
    ]
    ax.legend(handles=legend, loc="upper right", prop=_CN_FONT, fontsize=9, framealpha=0.82)


def _write_nodes(nodes: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "node_id",
        "group_code",
        "group_name",
        "object_type",
        "service",
        "row",
        "col",
        "target_row",
        "target_col",
        "lat",
        "lon",
        "x_norm",
        "y_norm",
        "risk",
        "model_stage",
        "grid_cell_role",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(nodes)
    json_path.write_text(json.dumps({"image_bbox": IMAGE_BBOX, "nodes": nodes}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
