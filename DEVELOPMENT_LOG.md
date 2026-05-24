# Development Log

本文件用于记录项目代码、配置、实验流程与评估方式的重要变化。日期按实际修改或实验完成日期记录；尚未提交到 Git 的内容会明确标注。

## 2026-05-16

- `d2b1737 Keep random obstacle maps connected`
  - 修改随机障碍物生成逻辑，要求剩余自由区域保持连通。
  - 修复随机障碍导致 agent 被隔离、产生不可覆盖区域的问题。
- `e82d972 Add explicit memory and graph attention`
  - 引入局部显式记忆通道与初步 GAT 通信结构。
  - 后续澄清：这一版本实现的是局部记忆增强，而不是每个 agent 独立维护完整地图并条件共享的去中心化显式建图版本。
- 调整课程训练规划，形成 8x8、13x13、18x18 及更大地图的渐进式课程方案，并补充 README 中的训练与可视化说明。

## 2026-05-21

- `ebbfc35 Enhance GAT communication and ablation setup`
  - 将 GAT 扩展为多头 masked attention，加入相对几何边特征偏置与 residual 输出。
  - 加入 GAT-on / GAT-off 对照配置和 `gat-ablation` 评估命令。
  - 将正式训练最大地图难度收敛为 20x20，并使各课程 `obstacle_ratio` 可独立调整。
  - 更新课程 rollout 大小、地图 seed 池与消融实验流程。
- `882022c Add project IDE metadata and development log`
  - 加入开发日志与项目环境相关文件。

## 2026-05-22

- 完成 GAT-on 课程二、课程三与课程四训练流程的推进。
- 将课程四 GAT-on 训练预算调整为 `3,200,000` agent transitions，以降低单次消融实验耗时。
- 在 README 中补充后续研究记录：
  - 当前模型应称为 `local-memory GAT-MAPPO baseline`。
  - 尚未实现去中心化完整地图记忆、条件地图共享与融合。
  - 记录官方 MAPPO 中可借鉴的稳定性机制和并行环境实现风险。
  - 记录覆盖任务中可能的创新方向：覆盖贡献、重复冲突与任务关系通信。

## 2026-05-23

- GAT-on 课程四训练完成：
  - 配置为 20x20、4 agents、5% 障碍、`3,200,000` agent transitions。
  - 输出位于 `E:\test plot\ablation_gat_on\20260522-225540\04-tier-4-20x20-4agents`。
- 将 GAT-off 课程四预算同步调整为 `3,200,000` agent transitions，保证课程四 GAT 消融对照公平。
- 开始并完成 GAT-off 的课程训练流程，为正式消融对比准备匹配 checkpoint。
- 明确项目评价取向：未知环境在线覆盖不以轨迹外观或强制 100% 完成为唯一判断，而以固定预算覆盖效率为主。

## 2026-05-24

- GAT-off 课程四训练完成：
  - 输出位于 `E:\test plot\ablation_gat_off\20260523-212551\04-tier-4-20x20-4agents`。
- 完成课程四 GAT-on / GAT-off 公平对比分析：
  - 在训练 seed 池上，GAT-on 完成率更高，但重复与 agent 间重叠仍更高。
  - 在 10 张未见 5% 障碍地图上，GAT-off 的 `Coverage@H`、`Coverage-AUC`、`T90/T95/T99`、高覆盖阶段重复率均优于当前 GAT-on。
  - 结论：现有基于距离邻接和局部隐特征传递的 GAT 可作为基础 baseline，但不足以成为最终覆盖协作机制。
- 新增离线评价指标实现，未改变训练或 checkpoint：
  - `Coverage@H`
  - `Coverage-AUC`
  - `T90 / T95 / T99` 与达到率
  - `StallCoverage@K`
  - `RepeatRatioAfter90`
  - `InterAgentOverlapRatio`
- 更新 `evaluate`、`benchmark` 与 `gat-ablation` 输出，并补充自动化测试。
- 生成 GAT 消融实验 Word 记录：
  - `reports/gat_ablation_comparison_2026-05-24.docx`
  - 该文档用于周报整理和实验归档，包含公平性设置、主要指标、轨迹示例、结论与后续改进方向。

## Pending Work

- 实现真正的去中心化显式地图记忆：每个 agent 自主更新已知自由区、障碍区、覆盖区与未知区。
- 在通信范围内进行条件地图共享与融合，避免环境全局真值直接进入 actor。
- 设计具有覆盖任务语义的通信输入，例如地图信息互补、覆盖冲突或覆盖意图。
- 参考官方 MAPPO，逐步补入终止/截断 mask、value normalization 和更稳定的 value loss，并单独验证其效果。
