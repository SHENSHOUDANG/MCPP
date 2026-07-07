# docs/experiment_log.md

> 规范版本：V1.7

## 状态标记

- `FROZEN`：当前有效规则。
- `AMENDS`：经批准修改被指明条款，未涉及部分继续有效。
- `REPLACES`：经批准替代被指明条款。
- `PENDING`：未经审查不得形成实现结论。
- `HISTORICAL`：仅作历史对照。

## 冻结记录

### 第一项 — FROZEN

- 采用有限作业窗口下的多周期滚动调度。
- 跨窗口保存任务历史。
- 周期任务与外生事件任务并存。
- 冻结上层调度与下层路径执行职责。
- 最终方法采用上下层双层强化学习；初期允许传统规划器暂代下层。
- 最终使用真实可行路径成本，不长期使用统一欧氏距离。

### 第二项 — FROZEN

- 建立五项任务真实性审查。
- 冻结三类一级任务族。
- 分离任务业务、几何和释放方式。
- 排除普通陆侧对象与常规环境观测任务。
- 主要实证场景由洋山港调整为洛杉矶港。
- 旧洋山任务点降为历史对照数据。

### 第三项 — FROZEN

- 采用“管理对象—调度任务单元—执行基元”。
- 保留点、线、面原始几何。
- 航点、测线、扫描条带和栅格不得默认成为上层任务。
- 任务以服务量与质量完成，不以到达完成。

### 第四至第七项 — FROZEN

- 分离硬能力与执行性能。
- 定义能力候选集 `R_cap`。
- 采用带类型关系的动态任务图。
- 严格区分外生事件、后继任务和中断剩余量。
- 删除 UAV 异常后自动生成 USV 复核。
- 删除按平台预切任务族。
- 删除综合风险分数。
- 分别建模重要性、截止时间、逾期、重访和义务等级。

### 第八项 — FROZEN ACTIVE BASELINE

- 有限窗口内采用事件驱动决策。
- 一次出动可连续执行多个任务。
- 时间和能量必须在同一合法回收点上联合满足。
- 使用物理能源与平台相关安全储备。
- 基准系统采用固定点补能。
- USV 搭载/回收/补能 UAV 仅为设备依赖的可选扩展。

### 第八至第十二项结构 V2 — FROZEN STRUCTURE

后续顺序：

1. 第九项：上层决策机制与求解算法。
2. 第十项：下层强化学习路径执行器及传统规划临时替代边界。
3. 第十一项：接口、训练与联调。
4. 第十二项：实验、基线、创新与题目。

不得跳项。

## 第九项 — FROZEN（当前由 V1.7-A9C 控制；V1.4-A9、V1.5-A9T、V1.6-A9D 的冲突部分已被替代）

原批准日期：2026-06-29。修订批准日期：2026-06-30。

### 当前有效决策主体与算法

- 研究对象为集中式统一调度与 CTDE 分布式协同调度的适用边界，不是 PPO/MAPPO 算法排名。
- Centralized PPO 与 CTDE-MAPPO 为并列主方案，最终方法或分场景适用边界由实验确定。
- Centralized PPO 使用完整全局状态和自回归联合动作。
- MAPPO 使用集中式 Critic、分散 Actor；同类型平台参数共享。
- 核心架构比较使用完整可行任务集；学习型 Top-K 降为后续扩展。

### 当前有效时延与奖励

- 仅保留上层任务等待时间：`WT_j(t)=t-release_time_j`。
- 有效分配时固定为 `assignment_time-release_time`。
- 下层旅行/服务、通信和反馈时间不进入该指标。
- 删除 V1.6 的 `tau_feedback`、信息年龄、发生/送达时间和首次服务开始时延。
- 总/平均/P95等待时间作为次级目标和评价指标；未分配任务按窗口结束截尾。

### 当前有效冲突规则

- Centralized PPO 在自回归动作生成阶段通过掩码避免同批任务冲突。
- MAPPO 多平台同时选择同一非联盟任务时，所有相关动作无效，任务保持开放。
- 禁止概率最大、匹配度最大、固定平台顺序或隐藏中央优化器仲裁。

### 当前代码影响

- 上层环境抽象出统一的 `get_feasible_candidates()`、`validate_action()` 和 `commit_action()`。
- 增加 Centralized PPO 联合解码器和 CTDE-MAPPO 多 Actor 接口。
- 任务结构保留 `first_valid_assignment_time`，删除 V1.6 时延字段。
- 评估器增加截尾平均等待、P95等待、冲突批次率、冲突动作率、推理时间和规模指标。
- Top-K 代码保留但默认不参与核心架构对比。

## V1.7 正式修订记录

### A9C — REPLACES：集中式与 CTDE 调度架构并列比较

- `amendment_id`: `V1.7-A9C`
- `operation`: `REPLACES`
- `target_clause`:
  - V1.5-A9T 中“Centralized PPO 为预设最终方法、MAPPO/HAPPO 仅作基线”的条款；
  - V1.6-A9D 中反馈时延、信息年龄、发生/送达时间和首次服务开始时延条款；
  - V1.4/V1.5 中学习型软锁定 Top-K 作为核心最终方法的角色。
- `approved_on`: `2026-07-05`
- 修订动机：用户明确将研究问题调整为集中式统一调度与 CTDE 分布式协同调度的适用性比较，并要求等待时间严格采用近两年高水平文献中的简单定义，避免下层与通信时延增加复杂性。
- 新冻结：
  1. Centralized PPO 与 CTDE-MAPPO 为并列主方案；
  2. 核心比较双方使用完整可行任务集；
  3. MAPPO 同任务冲突全部判无效，不使用中央仲裁；
  4. 等待时间采用 `WT=t_current-t_arrival`，有效分配后停止；
  5. 未分配任务按窗口结束截尾；
  6. 学习型 Top-K 仅作后续扩展；
  7. 必做自然架构对比和信息条件控制消融。
- 删除：`tau_feedback`、`occurred_at`、`delivered_at`、`information_age`、`first_service_start_time`、等待暴露积分和零反馈时延测试。
- 代码影响：上层策略接口分为 central 与 CTDE 两种；统一候选、掩码、环境提交和评估；新增冲突统计与截尾等待；移除 V1.6 时延字段。
- 受影响测试：联合动作独占、MAPPO 冲突全部无效、等待边界、终局截尾、公平候选、信息消融隔离、训练预算一致和跨规模测试。

## V1.4 正式修订记录

批准日期：2026-06-30。

### A9 — AMENDS：第九项候选机制与策略结构

- `amendment_id`：`V1.4-A9`。
- `operation`：`AMENDS`。
- `target_clause`：`AGENTS.md/第九项上层实现必须`、`README.md/第九项冻结结论与运行闭环`、`docs/current_task.md/已冻结的上层约束`、`docs/model_specification.md/16.1—16.5、16.7—16.11`。
- 修订动机：完整候选集在任务数量增加时导致动作规模增长和短期候选频繁变化；简单 Top-K 又可能隐藏重要任务、造成饥饿和批次锁死。因此引入在完整状态上学习任务组合、具有保护覆盖和安全修复的软锁定任务波次。
- 核心修订：
  1. 将 `top_k = OFF` 改为 `wave_mode = LEARNED_SOFT_LOCKED_TOP_K`；
  2. 全部开放任务仍进入图状态和全局奖励；
  3. 波次选择采用无放回自回归策略，选择顺序不表示执行顺序；
  4. `K` 为名义容量，保护任务可溢出；
  5. 正常完整更新仅在波次初始成员全部关闭后发生；
  6. 新保护任务即时覆盖，永久不可行、阻塞和超时采用显式局部修复或结转；
  7. 新增波次宏动作、波次状态、波次 SMDP 回报和波次管理惩罚；
  8. 完整可行集降为正式基线，而非当前唯一方法。
- 未被修改：中央调度主体、异构图、波次内调度动作类型、硬安全掩码、回收点预留、任务完成语义、事件驱动时间模型和最终双层强化学习路线继续有效。
- 受影响测试：任务排列等变、保护溢出、波次外责任、正常关闭、覆盖插入、局部修复、超时结转、波次版本、波次时间折扣、完整可行集回退。



## V1.6 正式修订记录

批准日期：2026-07-05。

### A9D — AMENDS：简单反馈时延与任务响应时延次级目标

- `amendment_id`：`V1.6-A9D`。
- `operation`：`AMENDS`。
- `target_clause`：`AGENTS.md/第九项上层实现必须、禁止事项与范围状态`、`README.md/核心目标、第九项冻结结论、运行闭环`、`docs/current_task.md/已冻结的上层约束`、`docs/model_specification.md/11、16.3、16.7、16.10、16.11、17`。
- 修订动机：用户希望在 PPO 与 MAPPO 比较中加入简单时延并将其作为次级优化目标。现有动作空间不含通信资源分配，因此纯固定通信时延不可被策略直接优化；需要区分外生反馈时延与可优化任务响应时延，避免与逾期、迟期和总作业时间重复。
- 核心修订：
  1. 增加单一 `tau_feedback`，延迟任务释放通知、平台状态和下层执行反馈的送达；基准不引入丢包、乱序、带宽或拓扑优化；
  2. 所有事件记录 `occurred_at`、`delivered_at`、`source_state_version`，状态记录 `observation_timestamp` 与 `information_age`；
  3. 定义 `D_response = first_service_start_time - release_time`，并按事件真实持续时间累计 `A_response`；
  4. 奖励增加归一化低权重响应时延项，但通过主目标守门保持其“次级目标”地位；
  5. 所有 PPO/MAPPO/HAPPO 与中央 PPO 变体共享时延情景、随机种子和统计口径；
  6. 冻结零时延等价性、过期动作拒绝和时延版本校验测试；
  7. 明确加入时延不重新打开中央 PPO 与 MAPPO 的主体选择，MAPPO/HAPPO 仍为平台级基线。
- 未冻结：`tau_feedback` 的具体数值/分布、`w_delay`、`D_ref`、主目标守门阈值、时延课程、统计检验与最终实验结论。
- 代码影响：事件队列增加发生/送达时间；任务状态增加首次分配与首次服务时间；图状态增加信息年龄与响应等待；奖励增加 `A_response`；评估器增加均值、P95、事件任务响应时延、过期动作失效率和零时延回归。
- 受影响测试：零时延回归、延迟事件排序、状态版本失效、响应时延边界、奖励不重复、基线同源时延采样和主目标守门。

## V1.5 正式修订记录

批准日期：2026-06-30。

### A9T — AMENDS：上层算法、训练协议与基线角色

- `amendment_id`：`V1.5-A9T`。
- `operation`：`AMENDS`。
- `target_clause`：`AGENTS.md/第九项上层实现必须与范围状态`、`README.md/当前状态、第九项冻结结论、上层算法与基线`、`docs/current_task.md/已冻结的上层约束与本项不处理`、`docs/model_specification.md/1、16.1、16.8、16.10、16.11、17`。
- 修订动机：V1.4 已确定学习型任务波次和波次内中央调度，但“双价值头是否强制、波次与调度如何训练、MAPPO/HAPPO 在论文中的角色”仍存在歧义，可能导致代码继续沿用平台级 Actor 或在同一 Critic 中混合两个时间尺度。
- 核心修订：
  1. 正式算法命名为双时间尺度中央异构图 PPO；
  2. 冻结共享异构图编码器、波次 Actor、调度 Actor、波次 Critic 和调度 Critic；
  3. 将双价值输出由建议改为强制，分别使用波次/调度回报、优势和经验缓冲；
  4. 冻结三阶段训练流程：U1 完整可行集调度预训练、U2 冻结编码器与调度分支训练波次、U3 解冻联合微调；
  5. 允许复用现有 MAPPO/HAPPO 的 PPO、GAE、缓冲和评估工程，但必须删除平台级 Actor 语义；
  6. 将 MAPPO 和 HAPPO 固定为平台级基线，不预设二者优劣；
  7. 冻结七类上层核心比较名单，并要求统一任务、约束、奖励、交互预算和随机种子；
  8. MAPPO/HAPPO 的冲突动作不得静默改派，必须按统一环境规则记录无效与等待。
- 未冻结：学习率、裁剪系数、网络层数、训练步数、随机种子数、参数量匹配方式、`K`、超时/阻塞阈值、统计结果和最终创新结论。
- 代码影响：新增 `staged_training_controller`、双价值损失、双 rollout buffer、阶段冻结/解冻、独立检查点、基线适配器、公平比较配置和冲突日志。
- 受影响测试：双 Critic 目标隔离、U1/U2/U3 梯度冻结、阶段检查点回滚、编码器学习率约束、七类基线可运行、MAPPO/HAPPO 冲突不静默改派。

## V1.2 正式修订记录

批准日期：2026-06-29。

### A1 — AMENDS：冻结优先规则

- 目标：`AGENTS.md/规范效力` 与原“前序条款永远优先”表述。
- 修订：未经批准的冲突仍以前序为准；经批准并标注 `AMENDS` 或 `REPLACES` 的后续条款可修改指定前序条款。
- 影响：文档解析、版本审计、变更测试。

### A2 — REPLACES：任务真实性第三项

- 目标：原“移动平台必须提供固定系统无法等效取得的新增信息”。
- 替代：允许无法等效取得，或在安全性、及时性、完整性、运行干扰方面具有可解释优势并形成明确管理数据产品。
- 限制：不得以“更适合”为由将全部 GIS 对象任务化。

### A3 — AMENDS：第八项能源与补能状态机

- 增加任务执行后的实际能源扣减、连续补能函数和换电状态更新。
- 同步更新平台状态、补能开始/完成时刻、可用时刻和设施占用。
- 增加等待能耗，明确 `T_exit` 与 `T_recovery` 边界。

### A4 — AMENDS：最低实现数据结构

- 增加最低平台结构和回收/补能点结构。
- 固定平台状态、任务状态和义务等级枚举。
- 逾期继续作为派生量，不作为任务状态。

### A5 — AMENDS：周期任务字段与更新

- 任务结构加入 `period_interval`、`calendar_anchor`、`last_completion_time`、`next_due_time`、服务窗口和首次巡检初始化字段。
- 区分 `ACTUAL_COMPLETION` 与 `FIXED_CALENDAR` 更新。

### A6 — REPLACES：统一标量质量阈值

- 目标：原 `quality_j >= quality_threshold_j` 完成公式。
- 替代：任务族相关验收函数 `q_pass_j = A_j(y_j,Q_req_j)`。
- 完成：服务量达标且验收通过。
- 归一化质量分数仅可作为可选评价指标。

### A7 — AMENDS：空值、首次巡检与时间指标

- `deadline=null` 时不计算裕度、实时逾期和最终迟期。
- `max_revisit_interval=null` 时不计算重访违约。
- `last_completion_time=null` 时必须使用明确初始化机制或释放首次基线任务。
- 区分实时 `overdue` 与完成后 `lateness`。
- 将任务级裕度改名为 `best_case_slack`，明确其不代表竞争调度结果。

### A8 — AMENDS：数据溯源

正式来源增加：`source_url`、`source_version_or_edition`、`access_date`、`license_or_usage_terms`、`original_crs`、`file_checksum`、`processing_script_version`。

## 冲突审查与处理

### C1. 第一项中的洋山资料与第二项洛杉矶港场景

处理：第一项冻结的是通用对象、数据质量和分层职责，并未锁死最终实证港口。第二项有效确定洛杉矶港为主要场景；洋山港仅保留历史或补充用途。

### C2. 双层强化学习与下层传统规划

处理：最终研究架构明确为上下层均采用强化学习。传统规划方法仅可在研究初期临时替代下层，用于上层先行训练、接口验证和对照基线；必须遵守同一输入输出、任务完成与安全约束，不得被认定为最终下层方案。具体上下层算法及训练方式分别在第九至第十一项冻结，最终论文表述在第十二项冻结。

### C3. 旧“第十二项路径结果”编号

处理：保留“完整可行集需要下层路径结果”的实质要求；按结构 V2 映射为新第十项。

### C4. 第八项 V1 与结构 V2 的重新审查表述

处理：第八项 V1 继续作为实质基线；V1.2 的 A3—A4 为经批准的 `AMENDS`，对能源状态和实现结构作闭合补充；结构 V2 仅控制后续编号与审查边界。

### C5. 已选实证场景但 GIS 制作暂停

处理：场景与数据契约已冻结；大规模数据制作仍暂停，待模型审查顺序允许后恢复。

### C6. V1.3 完整可行候选集与 V1.4 学习型 Top-K 波次

处理：`V1.4-A9` 经批准 `AMENDS` 第九项候选机制。完整任务图、完整可行关系和全局责任核算继续保留；仅将普通 `ASSIGN_TASK` 的近期候选改为学习型软锁定任务波次。完整可行集转为正式基线，不再作为当前唯一动作语义。

## 历史实验边界

基于旧洋山固定点任务集的训练不能证明冻结后最终模型有效。可保留为历史工程基线，但不得作为最终真实港区实验结论。

## 后续实验记录字段

第十项下层算法仍为 `PENDING`。上层三阶段训练协议和核心基线名单已由 `V1.5-A9T` 冻结；第十一项剩余上下层联调方式，以及第十二项 `K`、阈值、超参数、实验结果、统计检验和创新结论仍为 `PENDING`。恢复实验后，每次运行至少记录：

```text
commit id
configuration id
map/data version and checksum
processing script version
random seeds
training/validation/test split
algorithm and hyperparameters
constraint mode
metrics
failure cases
interpretation and next decision
```

## V1.7 工程记录

### 2026-07-05 Yangshan PPO stability profile

- Operation: engineering hyperparameter adjustment only, no `AMENDS` or `REPLACES`.
- Target clause: item 9 algorithm comparison remains open; this does not freeze the final upper-level algorithm, reward function, or comparison conclusion.
- Updated `configs/port_yangshan_training_v133.toml` to increase PPO rollout sampling and reduce update aggressiveness after observing high-variance training reward curves on the historical Yangshan V1.3.3 baseline.
- Sampling now uses `num_envs = 4`, `env_workers = 4`, and `rollout_steps = 128`, increasing nominal per-update samples from `2 * 32 = 64` to `4 * 128 = 512`.
- Update strength is reduced with `learning_rate = 0.0002` and `clip_ratio = 0.15`; `update_epochs = 4`, reward weights, full-candidate action slots, task lifecycle, and scheduler state/action/mask semantics are unchanged.

### 2026-07-06 Scheduler metrics atomic write

- Operation: engineering logging robustness fix only, no `AMENDS` or `REPLACES`.
- Target clause: item 9 training pipeline remains open; this does not change scheduler state, action, mask, reward, rollout collection, checkpoint contents, or algorithm semantics.
- Updated `tools/train_port_scheduler_rl.py` so `scheduler_metrics.csv` and `scheduler_summary.json` are written through a temporary file plus `os.replace`, with short retries for transient Windows I/O failures.
- Motivation: a Yangshan V1.3.3 stable-profile run reached `400384` steps and checkpointed normally, then failed while reopening `scheduler_metrics.csv` for overwrite with Windows `OSError: [Errno 22] Invalid argument`.

### 2026-07-06 V1.7 repository documentation sync

- Operation: documentation synchronization only, no new `AMENDS` or `REPLACES`.
- Source package: `Port_UAV_USV_Codex_Spec_Package_V1.7.zip` and `??UAV-USV??_Codex???????_V1.7.docx`.
- Updated repository-facing `AGENTS.md`, `README.md`, `docs/current_task.md`, `docs/model_specification.md`, and `docs/experiment_log.md` from the V1.7 package, and added `docs/literature_basis.md`.
- Purpose: remove stale V1.2 top-level documentation labels and make the active project memory reflect `V1.7-A9C`.
- Code behavior, training configuration, reward weights, task data, and generated artifacts are unchanged by this sync.

### 2026-07-07 Yangshan wait-cost engineering scan

- Operation: historical engineering sensitivity scan only, no `AMENDS` or `REPLACES`.
- Target clause: `V1.7-A9C` remains active. The formal upper-level scheduling-wait metric remains `WT_j(t)=t-release_time_j`, fixed at effective assignment and truncated at window end for unassigned tasks.
- Code impact: `src/mathbased_mcpp/port_inspection/reward.py` now supports optional scaled current open-wait reward shaping through `wait_time_cost`, `wait_time_scale`, and `wait_time_aggregation`; `configs/port_yangshan_training_v133.toml` enables a historical Yangshan scan weight.
- Test impact: `tests/test_port_reward.py` covers disabled wait cost, default scaled open-wait shaping, and explicit sum aggregation for ablation use.
- Experiment scope: generated Yangshan comparison outputs under wait-cost scan directories are engineering artifacts and are excluded from Git. They may guide follow-up evaluation design but do not establish the final upper-level architecture, final reward formula, statistical result, or paper conclusion.
- Documentation boundary: this current-step/open-wait shaping must not be cited as the Zhang and Ou waiting-time definition; formal reporting should still use assigned and window-truncated waiting metrics from the environment summaries.
