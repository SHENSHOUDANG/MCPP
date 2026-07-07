# docs/current_task.md

> 规范版本：V1.7

## 当前正式任务

**第十项：冻结 UAV/USV 下层强化学习路径执行器及传统规划临时替代边界。**

第九项已由 `V1.7-A9C` 重新冻结为“Centralized PPO 集中式统一调度与 CTDE-MAPPO 分布式协同调度的并列架构比较”。第十项只确定下层如何在给定平台、任务几何和安全约束下生成旅行路径与服务路径，不得替任何上层架构重新决定任务归属。

## 必须确定的内容

### 下层决策主体与共享方式

- UAV 与 USV 分别使用独立策略、按类型共享策略或类型条件统一策略；
- 点、线、面任务采用统一策略、任务条件分支或不同执行模板；
- 训练阶段允许使用哪些全局信息，执行阶段每个平台能够获得哪些输入。

### 状态、动作与约束

- 平台位置、航向、速度、能源、载荷和当前状态；
- 原始任务几何、剩余几何、执行模板和质量要求；
- 动态障碍、禁行区、通航环境和可见性；
- 移动动作、服务基元、等待、退出、失败上报和返航接口；
- 动力学、安全、碰撞、能源和安全中断点。

### 奖励与完成判定

- 旅行效率、服务进度、任务族相关质量、安全和能耗；
- 到达不等于完成，必须与冻结的 `rho_j` 和 `A_j` 验收函数兼容；
- 奖励不得替代硬安全约束。

### 传统规划临时替代

- 点、线、面分别采用何种临时规划器；
- 临时规划器与最终强化学习策略使用同一输入输出契约；
- 临时规划器不得改变上层任务完成、安全返航或质量验收语义。

## 已冻结的上层约束

- 两种并列上层架构：Centralized PPO 与 CTDE-MAPPO。
- 核心架构比较使用完整可行任务集和相同硬掩码。
- Centralized PPO 基于全局状态自回归生成联合任务分配。
- CTDE-MAPPO 训练时使用全局 Critic，执行时各平台独立选择任务，同类型平台参数共享。
- MAPPO 同任务冲突按“全部相关动作无效、任务保持开放”处理。
- 上层等待时间为 `assignment_time - release_time`，不包含下层旅行、服务或反馈时间。
- 不建立反馈时延、信息年龄、下层执行时延或随机噪声模型。
- 学习型 Top-K 不是核心架构比较的一部分。
- 下层必须向两种上层返回相同口径的预计与实际成本，不得为某一架构提供额外信息。

## 本项不处理

- 重新将 Centralized PPO 指定为唯一最终方法；
- 将 MAPPO 降回普通基线或提前宣称其更适合动态任务；
- 改变 MAPPO 冲突处理规则；
- 为任一架构单独启用学习型 Top-K；
- 增加通信延迟、反馈延迟、信息年龄或下层随机成本模型；
- 最终超参数、统计检验、实验结果、适用边界结论和论文题目。

## 验收条件

第十项只有在一个自洽规范中同时明确以下内容后才算完成：

- UAV/USV 下层决策主体与参数共享方式；
- 点、线、面任务的状态、动作、奖励与质量验收；
- 动力学、安全、能源和中断规则；
- 最终下层强化学习算法及被拒方案；
- 临时传统规划器及其严格接口边界；
- 预计成本、实际反馈、失败码和版本失效机制；
- 对 Centralized PPO 与 CTDE-MAPPO 完全一致的上层接口；
- 可变地图、任务几何和平台配置的泛化测试。

每项结论必须说明依据、备选方案和代码影响。

## 近期工程记录

### 2026-07-05 Yangshan PPO stability profile

- Increased the Yangshan V1.3.3 scheduler training sampling profile after reward-curve inspection: `num_envs = 4`, `env_workers = 4`, and `rollout_steps = 128`.
- Reduced update aggressiveness with `learning_rate = 0.0002` and `clip_ratio = 0.15`.
- Reward weights, `update_epochs`, task lifecycle, candidate-set semantics, state/action/mask semantics, and item-9 algorithm-selection status are unchanged.

### 2026-07-06 Scheduler metrics atomic write

- Hardened scheduler training metric writes after a Windows `Errno 22` failure while overwriting `scheduler_metrics.csv` during a Yangshan stable-profile run.
- `scheduler_metrics.csv` and `scheduler_summary.json` now use temporary-file replacement with short retries.
- Training semantics, reward weights, rollout collection, checkpoint payloads, and item-9 algorithm-selection status are unchanged.

### 2026-07-06 V1.7 documentation sync

- Synced repository-facing specification documents from `Port_UAV_USV_Codex_Spec_Package_V1.7.zip`.
- Added `docs/literature_basis.md` as the required source-separation memory file for V1.7 work.
- This documentation sync records the already-approved `V1.7-A9C` architecture comparison and delay-boundary semantics; it does not change code behavior.

### 2026-07-07 Yangshan wait-cost engineering scan

- Added an engineering-only wait-cost reward shaping path for the historical Yangshan V1.3.3 baseline and related sensitivity configs.
- This is not an `AMENDS` or `REPLACES` to `V1.7-A9C`: formal scheduling-wait metrics remain `assignment_time - release_time`, with unassigned tasks counted by window-end truncation.
- The current-step/open-wait reward load is only a training sensitivity probe on historical data and must not be described as the literature waiting-time definition or as a final architecture conclusion.
- Generated checkpoints, metrics, summaries and raw V1.7 package bundles are kept out of Git; only code, tests, configs and project memory are intended for commit.
