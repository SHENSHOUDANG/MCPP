from __future__ import annotations

import csv
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "gat_ablation_comparison_2026-05-24.docx"
TRAIN_CSV = ROOT / "outputs" / "gat_ablation" / "course4_train_pool_5pct_summary.csv"
HELDOUT_CSV = ROOT / "outputs" / "gat_ablation" / "course4_heldout_5pct_summary.csv"
GAT_ON_IMAGE = Path(r"E:\test plot\ablation_gat_on\20260522-225540\04-tier-4-20x20-4agents\trajectory.png")
GAT_OFF_IMAGE = Path(r"E:\test plot\ablation_gat_off\20260523-212551\04-tier-4-20x20-4agents\trajectory.png")

NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
)
EMU_PER_INCH = 914400


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["arm"]: row for row in csv.DictReader(handle)}


def pct(value: str) -> str:
    return f"{float(value) * 100:.2f}%"


def num(value: str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def delta(value: str, percent: bool = False) -> str:
    parsed = float(value) * (100 if percent else 1)
    suffix = " pp" if percent else ""
    return f"{parsed:+.{2}f}{suffix}"


def run(text: str, bold: bool = False, size: int = 21) -> str:
    weight = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        f"{weight}<w:rFonts w:ascii=\"Calibri\" w:eastAsia=\"Microsoft YaHei\"/>"
        f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        "</w:rPr>"
        f"<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"
    )


def paragraph(text: str = "", bold: bool = False, size: int = 21, align: str | None = None, spacing_after: int = 100) -> str:
    alignment = f'<w:jc w:val="{align}"/>' if align else ""
    props = f"<w:pPr>{alignment}<w:spacing w:after=\"{spacing_after}\"/></w:pPr>"
    return f"<w:p>{props}{run(text, bold=bold, size=size) if text else ''}</w:p>"


def heading(text: str, level: int = 1) -> str:
    size = 30 if level == 1 else 25
    before = 260 if level == 1 else 180
    return (
        "<w:p><w:pPr>"
        f"<w:spacing w:before=\"{before}\" w:after=\"120\"/>"
        "</w:pPr>"
        f"{run(text, bold=True, size=size)}</w:p>"
    )


def cell(text: str, bold: bool = False, shaded: bool = False) -> str:
    shade = '<w:shd w:fill="D9EAF7"/>' if shaded else ""
    return (
        "<w:tc><w:tcPr>"
        f"{shade}<w:tcMar><w:top w:w=\"80\" w:type=\"dxa\"/><w:start w:w=\"90\" w:type=\"dxa\"/>"
        "<w:bottom w:w=\"80\" w:type=\"dxa\"/><w:end w:w=\"90\" w:type=\"dxa\"/></w:tcMar></w:tcPr>"
        f"{paragraph(text, bold=bold, size=19, spacing_after=0)}</w:tc>"
    )


def table(rows: list[list[str]]) -> str:
    table_rows = []
    for index, values in enumerate(rows):
        table_rows.append("<w:tr>" + "".join(cell(value, bold=index == 0, shaded=index == 0) for value in values) + "</w:tr>")
    return (
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders><w:top w:val=\"single\" w:sz=\"6\" w:color=\"A6A6A6\"/>"
        "<w:left w:val=\"single\" w:sz=\"6\" w:color=\"A6A6A6\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"6\" w:color=\"A6A6A6\"/>"
        "<w:right w:val=\"single\" w:sz=\"6\" w:color=\"A6A6A6\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"D9D9D9\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"D9D9D9\"/></w:tblBorders></w:tblPr>"
        + "".join(table_rows)
        + "</w:tbl>"
    )


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        data = handle.read(24)
    return struct.unpack(">II", data[16:24])


def image_paragraph(rel_id: str, image_id: int, path: Path, width_inches: float = 5.8) -> str:
    pixel_width, pixel_height = png_dimensions(path)
    cx = int(width_inches * EMU_PER_INCH)
    cy = int(cx * pixel_height / pixel_width)
    name = escape(path.name)
    return f"""
<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="100"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{image_id}" name="{name}"/>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:nvPicPr><pic:cNvPr id="{image_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>
</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
"""


def heldout_table(rows: dict[str, dict[str, str]]) -> list[list[str]]:
    on, off, diff = rows["gat_on"], rows["gat_off"], rows["delta_on_minus_off"]
    fields = [
        ("Coverage@100", "coverage_at_100_mean", True),
        ("Coverage@200", "coverage_at_200_mean", True),
        ("Coverage@300", "coverage_at_300_mean", True),
        ("Coverage@500", "coverage_at_500_mean", True),
        ("Coverage-AUC", "coverage_auc_mean", False),
        ("T90 (steps)", "t90_mean_reached", False),
        ("T95 (steps)", "t95_mean_reached", False),
        ("T99 (steps)", "t99_mean_reached", False),
        ("T99 reach rate", "t99_reach_rate", True),
        ("RepeatRatioAfter90", "repeat_ratio_after_90_mean", True),
        ("InterAgentOverlapRatio", "inter_agent_overlap_ratio_mean", True),
        ("StallCoverage@50", "stall_termination_coverage_mean", True),
        ("Completion rate", "completion_rate", True),
    ]
    values = [["指标", "GAT-on", "GAT-off", "on - off"]]
    for label, key, is_percent in fields:
        formatter = pct if is_percent else num
        values.append([label, formatter(on[key]), formatter(off[key]), delta(diff[key], percent=is_percent)])
    return values


def train_pool_table(rows: dict[str, dict[str, str]]) -> list[list[str]]:
    on, off = rows["gat_on"], rows["gat_off"]
    return [
        ["指标", "GAT-on", "GAT-off"],
        ["最终覆盖率", pct(on["coverage_ratio_mean"]), pct(off["coverage_ratio_mean"])],
        ["完成率", pct(on["completion_rate"]), pct(off["completion_rate"])],
        ["Coverage-AUC", num(on["coverage_auc_mean"], 4), num(off["coverage_auc_mean"], 4)],
        ["平均完成/终止步数", num(on["steps_mean"], 1), num(off["steps_mean"], 1)],
        ["总重复率", pct(on["repeat_ratio_mean"]), pct(off["repeat_ratio_mean"])],
        ["90% 后重复率", pct(on["repeat_ratio_after_90_mean"]), pct(off["repeat_ratio_after_90_mean"])],
    ]


def document_xml(train_rows: dict[str, dict[str, str]], heldout_rows: dict[str, dict[str, str]]) -> str:
    body: list[str] = []
    body.append(paragraph("GAT 消融实验对比报告", bold=True, size=38, align="center", spacing_after=80))
    body.append(paragraph("课程四：20x20 未知障碍地图、4 智能体、5% 障碍率", size=24, align="center"))
    body.append(paragraph("记录日期：2026-05-24    用途：周报与实验归档", size=20, align="center", spacing_after=240))

    body.append(heading("1. 实验目的"))
    body.append(paragraph("比较当前局部隐特征 GAT-MAPPO 与不启用 GAT 的 MAPPO baseline 在多智能体在线覆盖任务中的差异。评估重点从严格完成全覆盖转为有限预算覆盖效率，并关注高覆盖阶段的重复运动与停滞。"))

    body.append(heading("2. 公平对照设置"))
    body.append(table([
        ["项目", "GAT-on", "GAT-off"],
        ["地图与 agent", "20x20，4 agents", "20x20，4 agents"],
        ["障碍率", "5%", "5%"],
        ["训练预算", "3,200,000 transitions", "3,200,000 transitions"],
        ["Rollout", "2048", "2048"],
        ["训练 seed 池", "20260440 - 20260447", "20260440 - 20260447"],
        ["主要区别", "4-head GAT 开启", "GAT 关闭"],
    ]))
    body.append(paragraph("评估说明：新增指标均为 checkpoint 训练完成后的离线计算，不会改变已训练策略或消融公平性。", size=19))

    body.append(heading("3. 主要评价指标"))
    body.append(paragraph("主指标包括 Coverage@H（固定步数内覆盖率）、Coverage-AUC（覆盖率-时间曲线面积）、T90/T95/T99（达到覆盖阈值所需步数）与 StallCoverage@50（连续 50 步没有新增覆盖时的覆盖率）。重复率、90% 覆盖后的重复率和 agent 间覆盖重叠率用于解释协作效率。路径是否规整或是否以直线为主仅作次要观察。"))

    body.append(heading("4. 训练 seed 池结果"))
    body.append(table(train_pool_table(train_rows)))
    body.append(paragraph("在见过的地图上，GAT-on 的严格完成率较高，但 Coverage-AUC 低于 GAT-off，且重复率更高。这表明 GAT-on 更容易在训练地图上最终完成任务，但未表现出更好的整体在线覆盖效率。"))

    body.append(heading("5. 未见地图结果（主要依据）"))
    body.append(paragraph("测试地图：10 张未见的 5% 障碍地图，seed 为 20260601 至 20260610。"))
    body.append(table(heldout_table(heldout_rows)))
    body.append(paragraph("未见地图上，两种模型的完成率同为 60%，但 GAT-off 在所有固定预算覆盖率、Coverage-AUC 以及 T90/T95/T99 上均更好；GAT-on 的高覆盖阶段重复率与 agent 间重叠率也更高。"))

    body.append(heading("6. 轨迹示例"))
    body.append(paragraph("下图为课程四自动导出的代表性轨迹。同一评估流程下，GAT-on 完成用时为 185 步、重复率 48.92%；GAT-off 完成用时为 144 步、重复率 34.48%。轨迹图只用于辅助解释效率差异，不作为路径美观评价。"))
    body.append(paragraph("GAT-on 轨迹示例", bold=True, size=20, align="center"))
    body.append(image_paragraph("rId2", 2, GAT_ON_IMAGE))
    body.append(paragraph("GAT-off 轨迹示例", bold=True, size=20, align="center"))
    body.append(image_paragraph("rId3", 3, GAT_OFF_IMAGE))

    body.append(heading("7. 分析结论"))
    for text in (
        "1. 当前 GAT 并非没有效果：在较少智能体或训练 seed 池上，它可能提高严格完成概率。",
        "2. 当前 GAT 未能提高课程四未见地图上的预算覆盖效率，反而加剧高覆盖阶段的重复运动和停滞。",
        "3. 现有 GAT 仅聚合通信范围内的局部隐特征，缺少覆盖地图、未知区域、覆盖意图与冲突关系等任务语义，因此不适合作为最终协作机制。",
        "4. 当前消融结果可作为论文中的基础 baseline，论证仅靠距离邻接的隐特征通信不足以解决未知环境多智能体覆盖问题。",
    ):
        body.append(paragraph(text))

    body.append(heading("8. 后续工作建议"))
    for text in (
        "1. 实现每个 agent 独立维护的显式地图记忆，以及通信范围内的条件地图共享与融合。",
        "2. 设计包含覆盖信息互补、覆盖冲突或覆盖意图的通信输入，使 GAT 获得覆盖任务语义。",
        "3. 持续使用 Coverage@H、Coverage-AUC、T90/T95/T99、StallCoverage@K 作为主要评价指标。",
        "4. 在方法更新后再次设置无通信、基础 GAT、地图共享/覆盖感知 GAT 的分层消融实验。",
    ):
        body.append(paragraph(text))

    body.append(heading("附录：结果文件"))
    for text in (
        r"未见地图汇总：outputs\gat_ablation\course4_heldout_5pct_summary.csv",
        r"训练种子池汇总：outputs\gat_ablation\course4_train_pool_5pct_summary.csv",
        r"GAT-on checkpoint：E:\test plot\ablation_gat_on\20260522-225540\04-tier-4-20x20-4agents\best_policy.pt",
        r"GAT-off checkpoint：E:\test plot\ablation_gat_off\20260523-212551\04-tier-4-20x20-4agents\best_policy.pt",
    ):
        body.append(paragraph(text, size=18))

    body.append(
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1100" w:right="900" w:bottom="1100" w:left="900" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:document {NS}><w:body>{''.join(body)}</w:body></w:document>"
    )


def build_report() -> Path:
    train_rows = load_rows(TRAIN_CSV)
    heldout_rows = load_rows(HELDOUT_CSV)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = document_xml(train_rows, heldout_rows)
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/gat_on_trajectory.png"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/gat_off_trajectory.png"/>
</Relationships>"""
    with zipfile.ZipFile(REPORT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)
        archive.write(GAT_ON_IMAGE, "word/media/gat_on_trajectory.png")
        archive.write(GAT_OFF_IMAGE, "word/media/gat_off_trajectory.png")
    return REPORT_PATH


if __name__ == "__main__":
    print(build_report())
