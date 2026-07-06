# docs/literature_basis.md

> 规范版本：V1.7

## 1. 本文件用途

本文件区分：

- 直接从文献采用的数学定义；
- 由文献支持的架构选择；
- 为保证实验可解释性而制定的项目公平性规则。

Codex 和论文写作不得把第三类规则伪称为某篇论文的原始模型。

## 2. 等待时间的直接文献来源

Zhang, W. and Ou, H. (2025) 在 *Scientific Reports* 中明确给出：

$$
WT(T_k)=t_{current}-t_{arrival}(T_k)
$$

论文解释该指标为任务在获得调度前保持未分配状态的时间，并使用总等待时间、平均等待时间评价强化学习调度结果。

本项目只做符号映射：

| 文献符号 | 本项目符号 |
|---|---|
| $T_k$ | 港区巡检任务 $j$ |
| $t_{arrival}(T_k)$ | `release_time_j` |
| $t_{current}$ | 当前上层调度时刻 $t$ |
| 任务被调度 | 有效 `ASSIGN_TASK` 正式提交 |

因此：

$$
WT_j(t)=t-release\_time_j
$$

该定义不包含旅行、服务和反馈时间。

## 3. 港口上层时间语义来源

Ren et al. (2024) 的港口拖轮调度模型使用动态任务通知时刻、计划开始时刻和最大等待/缓冲时间描述港口资源调度。该文用于支持：

- 港口任务具有动态通知/到达时刻；
- 等待边界属于上层资源调度；
- 计划开始与任务到达之间的时间可作为港口调度指标。

本项目不照搬其 Stackelberg 博弈、模糊参数或最大等待硬约束。

## 4. 分布式异构任务分配来源

Bagchi, Nair and Das (2024) 提出动态、分散式、能源感知的多机器人任务分配，明确讨论运行时动态任务、冲突解决和平均等待时间等指标。该文支持分布式调度作为独立研究架构，而不是 Centralized PPO 的普通消融。

Dai et al. (2025) 提出面向异构多机器人的分散式强化学习任务分配与调度，采用能力条件和注意力结构，使机器人在执行时反应式选择任务，并考察规模泛化与计算效率。该文支持：

- 异构平台可通过能力特征进入分散策略；
- 分散式学习策略值得与中央调度比较；
- 规模泛化必须通过实验验证，不能预设。

## 5. MAPPO 的 CTDE 来源

MAPPO 遵循集中训练、分布执行：

- Actor 执行时依据各自观测输出动作；
- 集中式 Critic 在训练时使用全局状态；
- 同质/同类型智能体可共享 Actor 参数。

因此论文必须写“CTDE 分布式协同调度”，不能写“完全分布式训练”。

## 6. 项目公平性规则

以下内容不是上述文献的原始公式，而是本项目用于隔离架构差异的实验规则：

- Centralized PPO 与 MAPPO 的核心比较使用完整可行任务集；
- MAPPO 同一任务冲突时全部相关动作无效；
- 不用中央仲裁器选择冲突获胜者；
- 设置自然信息架构对比和信息条件控制消融；
- 未分配任务按窗口结束时刻计算截尾等待；
- 学习型 Top-K 降为核心比较后的扩展。

论文中应明确称其为“统一实验协议”或“公平比较规则”。

## 7. 参考文献

1. Zhang, W., & Ou, H. (2025). Reinforcement learning based multi objective task scheduling for energy efficient and cost effective cloud edge computing. *Scientific Reports*, 15, 41716. DOI: 10.1038/s41598-025-25666-1.
2. Ren, Y., Chen, Q., Lau, Y.-y., Dulebenets, M. A., Li, B., et al. (2024). A multi-objective fuzzy programming model for port tugboat scheduling based on the Stackelberg game. *Scientific Reports*, 14, 25057. DOI: 10.1038/s41598-024-76898-6.
3. Bagchi, M. J., Nair, S. B., & Das, P. K. (2024). On a dynamic and decentralized energy-aware technique for multi-robot task allocation. *Robotics and Autonomous Systems*, 180, 104762. DOI: 10.1016/j.robot.2024.104762.
4. Dai, W., Rai, U., Chiun, J., Cao, Y., & Sartoretti, G. (2025). Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning. *IEEE Robotics and Automation Letters*, 10(3), 2654-2661. DOI: 10.1109/LRA.2025.3534682.
5. Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. *NeurIPS Datasets and Benchmarks*.
6. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

期刊分区会随 JCR/中科院版本变化。提交论文前应使用学校可访问的指定年度数据库复核分区；本规范冻结的是论文中的模型和事实，不把分区标签写成永久属性。
