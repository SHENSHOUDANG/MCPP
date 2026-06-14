# 主实验指标图

图中比较完整四课程训练完成的 GAT-OFF、GAT-ON 与 GAT-CIR，评测口径为覆盖阶段、20 个相同随机地图 seed `20261001-20261020`。

## 图文件

- `three_model_main_metrics_summary.png`：GAT-OFF、GAT-ON、GAT-CIR 的完成率、RepeatRatio、Repeat@90、平均环境步数汇总柱状图。
- `three_model_per_seed_completion_repeat_steps.png`：三模型逐随机地图的完成状态、RepeatRatio 和环境步数曲线。
- `three_model_metric_deltas.png`：OFF→ON 与 ON→CIR 两阶段指标变化图。
- `three_model_coverage_repeat_curves.png`：覆盖率随 step 变化和累计重复率随 step 变化的双子图。
- `three_model_coverage_curve.png`：覆盖率随 step 变化曲线。
- `three_model_repeat_curve.png`：累计重复率随 step 变化曲线。
- `three_model_coverage_repeat_curves_no_shadow.png`：无标准差阴影的覆盖率/累计重复率双子图。
- `three_model_coverage_curve_no_shadow.png`：无标准差阴影的覆盖率曲线。
- `three_model_repeat_curve_no_shadow.png`：无标准差阴影的累计重复率曲线。
- `three_model_coverage_repeat_zoomed_no_shadow.png`：局部放大的三模型覆盖率/累计重复率曲线，用于突出差异。
- `gat_on_vs_cir_zoomed_curves_no_shadow.png`：局部放大的 GAT-ON 与 GAT-CIR 曲线，适合展示 CIR 模块增益。
- `three_model_difference_curves.png`：三模型差值曲线，覆盖率为增益，重复率为下降幅度，正值表示更优。
- `gat_cir_over_on_difference_curves_zoomed.png`：GAT-CIR 相对 GAT-ON 的放大差值曲线，适合强调 CIR 的边际提升。
- `main_metrics_summary.png`：仅 GAT-ON 与 GAT-CIR 的两模型汇总柱状图。
- `per_seed_completion_repeat_steps.png`：仅 GAT-ON 与 GAT-CIR 的逐随机地图曲线。
- `main_metric_deltas.png`：仅 GAT-CIR 相对 GAT-ON 的指标变化图。

同时提供同名 `.svg` 文件，适合插入 Word 后保持清晰度。

## 曲线数据

- `../curves/three_model_coverage_repeat_mean_curves.csv`：每个 step 的覆盖率均值/标准差和累计重复率均值/标准差。
- `../curves/three_model_coverage_repeat_curve_summary.csv`：覆盖完成步数、C@100/C@150/C@200、Repeat@100/150/200 等摘要。
