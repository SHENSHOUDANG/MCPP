# docs/current_task.md

> 规范版本：V1.2

## 当前正式任务

**第九项：冻结上层决策机制与求解算法。**

在下列契约完成并获批前，不得选择或实现最终上层算法。最终研究架构为上下层均采用强化学习；本项只冻结上层。

## 必须确定的内容

1. **决策主体**
   - 掌握全局状态的中央调度器；或
   - 每个平台一个策略主体，采用集中训练、分散执行。

2. **状态表示**
   - 活动任务及带类型依赖关系；
   - 任务族、几何引用、释放/截止/重访/义务状态；
   - 平台位置、状态、物理剩余能源、补能状态和实际退出位置；
   - 回收点兼容性、容量、开放窗口和当前占用；
   - 下层估计的路径、时间、能耗和退出信息；
   - 当前窗口时间及继承的未完成状态。

3. **候选集与掩码**
   - 区分硬能力资格与当前完整可行性；
   - 明确哪些条件必须作为硬掩码；
   - 对无截止、无重访或首次巡检状态显式分支，不得以零值替代空值；
   - 明确 Top-K 是不用、仅作效率机制，还是作为消融变量。

4. **动作表示**
   - 平台—任务配对、逐平台顺序选择、联合匹配或其他可变规模结构；
   - 空闲、返航和补能动作必须符合第八项；
   - 明确重复选择任务、重复占用平台和超额占用回收点的消解方式。

5. **目标或奖励**
   - 每一项均对应任务完成、最终迟期、实时逾期、重访违约、资源消耗或安全约束；
   - `best_case_slack` 只表示单任务的最佳可行平台裕度，不得视为竞争条件下的确定结果；
   - 不得恢复综合风险分数；
   - 只有在决策架构确定后才定义信用分配。

6. **求解算法**
   - 比较集中式 PPO、MAPPO/IPPO、图/注意力策略和混合优化；
   - 依据任务与平台数量变化、依赖关系、资源占用和冲突处理进行选择；
   - 不得因旧项目曾使用某算法而直接沿用。

## 已具备前提

- 任务真实性与三类任务族。
- 点、线、面调度任务语义。
- 最低任务、平台和回收点数据结构。
- 能力候选集 `R_cap`。
- 带类型动态任务图和固定任务状态枚举。
- 任务族相关质量验收函数。
- 截止、重访、首次巡检和空值处理规则。
- 事件驱动时间、物理能量状态更新和安全返航规则。
- 上下层预计/实际成本接口。

## 本项不处理

- 点、线、面下层强化学习执行器及其训练方法。
- 研究初期临时传统规划器的具体算法实现；其只需满足既有接口。
- 上下层联合训练、分阶段训练与闭环联调细节。
- 最终 GIS 制作和最终策略训练。
- 最终基线、消融、创新点和论文题目。

## 验收条件

第九项只有在一个自洽规范中同时明确以下内容后才算完成：

- 决策主体；
- 状态/观测结构；
- 候选生成与掩码；
- 动作编码；
- 任务、平台和回收点冲突处理；
- 决策触发时机；
- 目标/奖励；
- 求解算法及被拒方案；
- 可变规模处理；
- 向第十项下层强化学习执行器提出的成本与可行性接口需求；
- 明确上层训练期间可调用接口兼容的传统下层规划器，但不得将其认定为最终方案。

每项结论必须说明依据、备选方案和代码影响。

## 工程执行备注

### 2026-06-29 洛杉矶港训练管线

- 当前正式审查项仍为第九项；洛杉矶港训练实现仅作为 `PENDING` 工程训练管线，不替代第九项算法冻结。
- 默认训练配置已切换到 `configs/port_los_angeles_training_v1.toml`。
- `los_angeles_training_v1` compact 场景包含点、线、面任务，用于验证上层训练代码、候选集、掩码和混合几何加载。
- smoke 训练已可运行；后续若要形成正式实验，仍需按 V1.2 数据契约替换为经核验的官方 GIS 数据和获批第九项算法规范。

### 2026-06-29 Los Angeles official-data correction

- Los Angeles is not treated like Yangshan: Yangshan may retain QGIS/self-defined historical baselines, but Los Angeles training geometry must come from official public chart/port data.
- `los_angeles_training_v1` has been corrected from an engineering seed scene to official NOAA ENC Direct geometry. The checked-in data uses the embedded NOAA official sample snapshot captured on 2026-06-29 because live REST execution was unavailable during this update.
- Training parameters remain `PENDING`: deadlines, risk, service time, release mode, and recovery depot are training assumptions, not official Port of Los Angeles work orders.
- Future LA data work should prefer live NOAA ENC Direct / official port data regeneration and must not reintroduce hand-drawn QGIS geometry for LA.

### 2026-06-29 LA effect figure

- Added a reproducible renderer for a Los Angeles official-geometry training effect figure.
- The figure now uses a NOAA ENC Direct Harbour chart export as the basemap and overlays the current `PENDING` NOAA-derived point, line, and area tasks, grid risk, depot, and provenance summary.
- Generated PNG files belong under ignored `reports/` and must not be treated as final experiment evidence.

### 2026-06-30 Multi-algorithm scheduler training comparison

- Current formal review remains item 9; no final upper-level algorithm is selected by this engineering change.
- `tools/train_port_scheduler_rl.py` now accepts `--algorithm` for `heterogeneous_mappo`, `shared_mappo`, and `centralized_ppo` training candidates.
- `tools/run_port_algorithm_comparison.py` runs the configured candidates on the Los Angeles `PENDING` training scenario and writes comparable JSON/CSV summaries.
- Comparison outputs belong under ignored `data/ports/*/algorithm_comparison/` and are not final baseline, ablation, or innovation evidence.
- IPPO, graph/attention policy, hybrid optimization, and the final rejected/accepted algorithm rationale remain pending item-9 decisions.

### 2026-06-30 User-provided LA task mapping import

- Replaced the compact 7-task NOAA snapshot training set with the user-provided `D:/地图/洛杉矶` Port of Los Angeles Task Mapping V2.0 chart-aligned catalog.
- Current `los_angeles_training_v1` now contains 26 released training tasks: 3 point tasks, 10 corridor tasks, and 13 area tasks.
- Four reinspection records from `reinspection_catalog_v2_0.csv` are preserved as metadata only; they are not automatically released into the current scheduler action space.
- Geometry is labelled `chart_aligned_research_geometry`, not native ENC vector geometry. It remains `PENDING` training data and must not be treated as final GIS or official work-order evidence.
- The import path is reproducible through `tools/import_los_angeles_task_mapping.py`; source CSV/GPKG checksums are embedded in the generated grid/task metadata.

### 2026-06-30 Yangshan user depot update

- User provided Yangshan depot coordinate `30 deg 36.27 min N, 122 deg 5.70 min E`.
- The coordinate is recorded as WGS84 `lat=30.6045`, `lon=122.095`, transformed to EPSG:32651 as `(413246.952064, 3386120.770780)`, and snapped to the current 100 m Yangshan grid cell `[82, 108]`.
- `yangshan_task_initial_v1` remains `HISTORICAL`; this depot update is a baseline data/config correction and does not alter the Los Angeles `PENDING` training scenario or freeze the final algorithm choice.

### 2026-06-30 LA V1.2 direct-service lifecycle alignment

- The Los Angeles `PENDING` training environment now uses `task_lifecycle = "v1_2_direct_service"`.
- Upper-level candidates are platform-task `service` assignments over released V1.2 tasks; they no longer expose screening candidates, review candidates, screening confidence, or an anomaly-triggered review queue.
- Task state transitions in this lifecycle are `ACTIVE -> ASSIGNED -> IN_SERVICE -> COMPLETED`. `deadline = null` remains `None` and is not converted to `0`.
- Yangshan remains explicitly `legacy_screen_review` for historical baseline compatibility only.
- This is an implementation alignment with the frozen V1.2 task-state semantics, not a final item-9 algorithm decision.

### 2026-06-30 HAPPO scheduler candidate interface

- Added HAPPO as an engineering candidate for item-9 upper-level scheduler comparison.
- HAPPO reuses the existing scheduler environment, local observation vectors, action masks, rollout batch, and centralized critic input; no task lifecycle, candidate generation, mask semantics, or environment state transition was changed.
- The HAPPO candidate uses one decentralized actor per platform and a centralized set critic. Training dispatches to a sequential per-agent actor update while MAPPO/PPO candidates keep their existing update path.
- This is not a final algorithm freeze. Existing `heterogeneous_mappo` remains a heterogeneous-actor MAPPO variant, not HAPPO.
