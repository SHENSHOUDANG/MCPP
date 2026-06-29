# docs/experiment_log.md

> 规范版本：V1.2

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

## 历史实验边界

基于旧洋山固定点任务集的训练不能证明冻结后最终模型有效。可保留为历史工程基线，但不得作为最终真实港区实验结论。

## 仓库同步记录

### 2026-06-29 V1.2 规范落库

- 同步 V1.2 规范文件到 `AGENTS.md`、`README.md`、`docs/current_task.md`、`docs/model_specification.md` 和本文件。
- 新增 V1.2 契约代码层，固定枚举、状态转移、空值时间指标和任务记录校验。
- 将 `yangshan_task_initial_v1` 配置和数据说明标记为 `HISTORICAL`，仅保留为工程基线，不作为最终 V1.2 实验依据。
- 历史训练入口需要显式确认 `--allow-historical-baseline` 后才能运行；检查与评估输出写入契约边界元数据。
- 验证：`python -m unittest discover tests` 通过。

### 2026-06-29 旧产物清理

- 删除已跟踪的 Yangshan 历史 checkpoint、训练指标、评估 trace、日志、import summary 和 raw source 包。
- 更新 `.gitignore`，将训练/评估输出、source 包、报告目录和缓存目录排除在版本控制之外。
- 本清理仅降低历史产物对后续上下文分析的干扰，不形成新的模型有效性、基线或实验结论。

### 2026-06-29 洛杉矶港训练管线原型

- 新增 `los_angeles_training_v1` compact 场景、专用平台配置和训练配置，状态为 `PENDING_ENGINEERING_TRAINING`。
- 默认 `check_port_inspection_env.py`、`train_port_scheduler_rl.py` 和评估脚本切换到洛杉矶港训练配置；Yangshan 仍仅作显式历史基线。
- 任务加载器支持 `point_tasks`、`line_tasks` 和 `area_tasks`，用于覆盖导航助航点、航道/泊位走廊和港池/水面区域训练。
- 已完成 smoke 训练：`tools/train_port_scheduler_rl.py --config configs/port_los_angeles_training_v1.toml --steps 8 --device cpu`，checkpoint 输出位于 ignored 的 `data/ports/los_angeles_training_v1/smoke_training/scheduler_rl/`。
- 验证：`python -m unittest discover tests` 通过。该结果仅说明训练管线可运行，不构成最终算法、基线或实验结论。

## 后续实验记录字段

最终基线、消融和创新结论在第十二项冻结前均为 `PENDING`。恢复实验后，每次运行至少记录：

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
### 2026-06-29 Los Angeles official-data correction

- User clarified that Los Angeles must not follow the Yangshan QGIS/self-defined scene pattern because NOAA and other official public chart/port datasets are available.
- Replaced the LA training scene provenance from `PENDING_ENGINEERING_TRAINING` engineering seed geometry to `PENDING_OFFICIAL_GEOMETRY_TRAINING` official NOAA ENC Direct geometry.
- Added a live NOAA REST generator path and an explicit `--use-embedded-official-snapshot` path. The checked-in data uses the 2026-06-29 embedded official NOAA sample snapshot because live REST execution was blocked by environment usage limits during this update.
- The generated tasks contain official geometry provenance fields (`source_dataset`, `source_agency`, `source_url`, `source_date`, `source_version_or_edition`, `access_date`, `original_crs`, `file_checksum`, and `processing_script_version`). Scheduling parameters remain training assumptions and do not represent official work orders.
