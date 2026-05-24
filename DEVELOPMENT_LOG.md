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
- 新增 map-intent GAT 公平消融基线实现，正式课程训练尚未开始：
  - 保留已有 `ablation_gat_on.toml` / `ablation_gat_off.toml` 及其结果，不改写历史 baseline。
  - 新增 `configs/ablation_mapmsg_gat_on.toml` 与 `configs/ablation_mapmsg_gat_off.toml`；两臂共同开启去中心化显式地图记忆、通信范围内地图融合和覆盖意图消息，仅在 `use_graph_attention` 上形成核心对照。
  - 每个 agent 独立维护已知自由区、已知障碍区、自己覆盖区与已知团队覆盖区；显式记忆 actor 观测不再直接使用环境构造的真实全局 `team_covered` 局部裁剪。
  - 节点消息加入覆盖状态摘要、近期新增/重复/停滞摘要，以及由自身记忆导出的下一探索方向、目标方向距离和 `3 x 3` 目标探索区域意图。
  - 现有多头距离 masked GAT 结构保持不变：GAT-off 仅编码自身覆盖消息，GAT-on 额外聚合通信范围内邻居覆盖消息。
  - 补充配置、地图隐私/融合、意图消息、消息化 GAT 与训练加载 smoke tests。
- 纠正 actor 信息边界设计：
  - 审计确认已训练的旧 GAT-on/GAT-off actor 同时接收了环境真值生成的局部 `team_covered`，且旧 `uncovered` 与全局 `coverage_ratio` 同样包含团队覆盖真值；旧实验应标注为 legacy truth-observation comparison，不能作为去中心化 actor 的公平 baseline。
  - 新训练默认移除真实团队覆盖通道以及由团队全局覆盖派生的 actor 输入，仅保留自身可知的局部自由区、自身覆盖与最近路径信息；全局覆盖仍只保留在 critic、奖励与评价路径。
  - 修复 map-intent 显式地图：未通信时，agent 即使处于队友历史覆盖位置附近，也不会直接读取队友已覆盖格；队友覆盖知识只能通过通信范围内的地图融合获得。
  - 为旧 checkpoint 保留显式 legacy 重放兼容路径，以便审计既有结果，但禁止将其混称为修正后的去中心化消融结果。
  - 本轮不引入信息素通道；若后续研究信息素启发式，必须作为独立消融变量实现和报告。

## Pending Work

- 依次训练 `ablation_mapmsg_gat_on` 与 `ablation_mapmsg_gat_off` 的四级课程，并在相同未见地图 seed 集上完成公平比较。
- 根据 map-intent 消融结果判断覆盖意图消息是否降低 `RepeatRatioAfter90` 与 `InterAgentOverlapRatio`，而不是提前引入更多通信结构。
- 参考官方 MAPPO，逐步补入终止/截断 mask、value normalization 和更稳定的 value loss，并单独验证其效果。
