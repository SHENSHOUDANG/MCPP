# docs/model_specification.md

> 规范版本：V1.7

## 1. 研究边界

系统服务于港区运营、通航安全、海道维护和水侧设施检查，不研究陆侧通用巡检、无差别全覆盖或端到端底层控制。

最终研究架构为双层强化学习：上层比较集中式统一调度与集中训练—分布执行协同调度两种组织方式，下层学习点、线、面任务的旅行与服务路径执行。研究初期可用传统规划器临时替代下层，以先行验证统一接口和上层公平比较。上层不预设唯一最终算法：Centralized PPO 与 CTDE-MAPPO 为并列主方案，最终结论由任务规模、动态程度、信息条件、冲突和调度效率实验确定。学习型 Top-K 不参与核心架构比较，仅作为后续扩展。

## 2. 正式修订与冲突处理

未经明确批准的后续内容与前序冻结条款冲突时，以前序条款为准。

经用户批准并标注为 `AMENDS` 或 `REPLACES` 的后续条款，可以修改或替代被指明的前序条款；未被明确替代的内容继续有效。正式修订必须写入 `docs/experiment_log.md`，不得通过代码默认值或无记录重构改变模型语义。

## 3. 时间模型

### 3.1 跨周期状态

每个作业窗口结束后至少保留：

- `last_completion_time`；
- `next_due_time`；
- `remaining_work`；
- `unfinished_status`；
- `abnormal_status`；
- 平台能源、状态和实际退出位置；
- 回收点占用与可用状态；
- 固定日历实例状态。

周期任务、计划任务和一次性事件任务并存。

### 3.2 有限作业窗口

```text
H = [T_start, T_end]
```

上层统一以分钟记录时间，但采用事件驱动决策，不强制每分钟决策。任务释放、完成、中断、后继触发、平台恢复、补能开始/完成、回收点占用变化和下层计划失效均可触发重调度。

平台一次出动可连续执行多个任务；当补能时长、剩余窗口和安全条件允许时，一个窗口内可多次出动。

## 4. 空间与任务层级

```text
管理对象 -> 调度任务单元 -> 执行基元
```

### 4.1 管理对象

真实存在的航道、港池、泊位前沿、防波堤、航标、水中桥梁构件或其他合格水侧对象。

### 4.2 调度任务单元

能够独立分配、排序、记录进度并验收的最小业务作业包。

划分优先级：

```text
管理边界与业务属性
> 设施结构
> 几何连续性
> 服务工作量与预计服务时间
> 固定长度或面积
```

不得将统一固定长度或面积切分作为主方案。

### 4.3 执行基元

航点、测线、扫描条带、覆盖栅格、局部进出路径等下层元素，不直接进入上层动作空间。

## 5. 任务真实性与分类

任务必须同时满足：

1. 位于合格水域、水陆交界或明确水侧检查面；
2. 对应真实运营、通航、测量、维护或异常处置需求；
3. 移动平台能够获得固定系统或陆侧方式无法等效取得的信息，或者能够以更安全、更及时、更完整、对港区运行干扰更小的方式形成具有明确管理用途的数据产品；
4. 具有明确数据产品；
5. 具有明确完成条件和管理用途。

第三项中的优势必须有可解释依据，不能据此把所有 GIS 对象重新定义为任务。

一级任务族：

```text
HYDROGRAPHIC_SURVEY
SURFACE_SAFETY_PATROL
WATERSIDE_ASSET_INSPECTION
```

几何形式：

```text
TARGET | CORRIDOR | AREA
```

释放方式：

```text
PERIODIC | SCHEDULED | EVENT
```

任务族、几何和释放方式相互独立。“目标确认”属于既有任务族中的后继任务实例，不新增第四类任务族。

风、浪、潮位、流速、能见度、AIS 和 VTS 通常是环境输入。只有形成明确管理处置请求时，才释放为任务。

## 6. 实证数据契约

主要实证场景：洛杉矶港。

正式空间对象至少保留：

```text
source_dataset
source_agency
source_date
source_url
source_version_or_edition
access_date
license_or_usage_terms
original_id
original_crs
file_checksum
processing_script_version
processing_note
```

数据必须分层：

1. 原始官方数据；
2. 清洗后的管理对象；
3. 上层调度任务；
4. 下层执行几何；
5. 仿真事件情景。

仿真事件必须标记：

```text
release_mode = EVENT
scenario_generated = true
```

旧洋山港 219 个固定点和 3 个事件种子不再是最终真实任务集合。

## 7. 平台能力与最低平台结构

基准系统采用一种 UAV 配置和一种 USV 配置，同类允许多个实例。只有现有类型无法满足真实硬能力需求时，才增加平台类型。

对平台 `i` 与任务 `j`：

```text
F_cap[i,j] = 1(c_H[i] >= r_H[j])
R_cap[j] = {i | F_cap[i,j] = 1}
```

`R_cap` 仅表示硬能力资格，不等于当前完整可行集。

执行适配可考虑：

```text
预计旅行时间
平台相关的预计剩余服务时间
预计能耗
预计服务质量
平台可用状态
```

声学测深、海床地形和水下碍航物巡测仅向实际配置对应测量设备的 USV 开放。水面可见异常和部分水线外观检查可在满足最低能力和质量条件时由 UAV 或 USV 柔性执行。

只有单个平台均无法满足硬能力与同步服务要求，而某个平台集合能够满足时，才建立联盟任务。仅有潜在效率提升不足以建立联盟。

UAV 筛查后由 USV 近距检查通常表示为先后/触发关系，不是同步联盟任务。

最低平台数据结构：

```yaml
platform_id: string
platform_type: UAV|USV
configuration_id: string
current_position: geometry_or_coordinate
current_status: enum
hard_capability_vector: vector
energy_capacity: number
remaining_energy: number
reserve_energy: number
legal_recovery_point_ids: list
replenishment_mode: string
replenishment_rate_or_duration: number|object
available_time: number
current_task_id: string|null
actual_exit_position: geometry_or_coordinate|null
```

平台状态固定为：

```text
AVAILABLE
TRAVELING
SETUP
IN_SERVICE
WAITING
RETURNING
REPLENISHING
UNAVAILABLE
```

任何状态变化必须记录发生时刻和触发事件。

## 8. 动态任务表示与状态

活动任务图：

```text
G_t = (T_t, E_pre, E_trig, E_hier, M_sub)
```

- `T_t`：已释放且未关闭的任务。
- `E_pre`：确定性先后关系。
- `E_trig`：条件触发关系。
- `E_hier`：父子或从属关系。
- `M_sub`：互斥、替代或等效集合。

只有 `E_pre` 与 `E_trig` 构成的执行依赖图必须无环。周期重复通过跨窗口状态更新实现，不通过依赖环实现。

必须区分：

- 外生事件任务；
- 内生后继任务；
- 原任务中断后的剩余工作量。

UAV 异常不会自动生成 USV 任务。仅当后继任务能够取得新增、不可等效的信息且对应明确管理需求时才释放。第一阶段采用确定性异常结果和确定性触发规则。

义务等级固定为：

```text
MANDATORY
PENALIZED
OPTIONAL
```

任务状态固定为：

```text
UNRELEASED
ACTIVE
ASSIGNED
IN_SERVICE
INTERRUPTED
COMPLETED
CANCELLED
SUBSTITUTED
```

允许的主状态转移：

```text
UNRELEASED -> ACTIVE
ACTIVE -> ASSIGNED | CANCELLED | SUBSTITUTED
ASSIGNED -> IN_SERVICE | ACTIVE | CANCELLED
IN_SERVICE -> COMPLETED | INTERRUPTED
INTERRUPTED -> ASSIGNED | ACTIVE | CANCELLED | SUBSTITUTED
```

逾期不是任务状态，而是由当前时间和截止时间派生的标志与数值。

## 9. 最低任务数据结构

```yaml
task_id: string
parent_object_id: string
task_family: enum
object_type: string
geometry_mode: TARGET|CORRIDOR|AREA
geometry_ref: string
execution_template_ref: string
release_mode: PERIODIC|SCHEDULED|EVENT
release_time: number
importance_class: string
hard_capability_requirement: vector
required_work: number
completed_work: number
remaining_work: number
estimated_remaining_service_time_by_platform: map
work_threshold: number
quality_requirement: object
quality_acceptance_ref: string
deadline: number|null
service_window_start: number|null
service_window_end: number|null
max_revisit_interval: number|null
last_completion_time: number|null
next_due_time: number|null
period_interval: number|null
calendar_anchor: number|null
calendar_update_mode: ACTUAL_COMPLETION|FIXED_CALENDAR|null
revisit_initialization_mode: TRUSTED_HISTORY|COMMISSIONING_TIME|STUDY_START|INITIAL_INSPECTION_REQUIRED|null
revisit_initialization_time: number|null
obligation_level: MANDATORY|PENALIZED|OPTIONAL
parent_task_id: string|null
predecessor_ids: list
trigger_rule: object|null
substitution_set_id: string|null
status: enum
status_history: list
provenance: object
scenario_generated: bool
```

约束：

- 周期任务必须具有 `period_interval`、`next_due_time` 和 `calendar_update_mode`。
- `FIXED_CALENDAR` 必须具有 `calendar_anchor`。
- `importance_class`、`replenishment_mode` 和 `recovery_mode` 的取值必须来自获批配置注册表；规范未冻结时不得自行补默认值。
- `service_window_start/end = null` 表示没有额外服务窗口限制，不得按零处理。
- 非周期任务的周期字段应为 `null`。
- 可缓存质心坐标作为特征，但不得替代 `geometry_ref`。

周期更新：

```text
ACTUAL_COMPLETION:
next_due_time <- completion_time + period_interval

FIXED_CALENDAR:
next_due_time <- previous_next_due_time + period_interval
```

固定日历不因实际完成时刻整体平移。每个到达的日历时点均释放一个独立任务实例；若事件步长跨越多个日历时点，必须按时间顺序补放全部实例，不得静默跳过、合并或取消。

## 10. 进度、质量与完成

```text
rho_j(t) = min(1, completed_work_j(t) / required_work_j)
q_pass_j = A_j(y_j, Q_req_j) in {0,1}
completed_j = 1 iff rho_j >= work_threshold_j and q_pass_j = 1
```

其中：

- `A_j`：由任务族和执行模板指定的验收函数；
- `y_j`：实际服务结果；
- `Q_req_j`：多维质量要求。

质量要求可以包含测线间距、覆盖率、定位精度、数据完整性、图像清晰度、目标观测角度或有效采样数量。可另行返回归一化质量分数作为评价指标，但所有任务不得共用一个标量阈值决定关闭。

到达不等于完成；服务量达标但质量验收未通过时不得关闭任务。

任务中断后保留原任务标识、剩余工作量和剩余几何，不得伪造为新的外生事件任务。

## 11. 重要性、截止时间、逾期与重访

以下属性分别建模：

- 重要性；
- 释放时间；
- 截止时间；
- 重访间隔；
- 巡检年龄；
- 实时逾期；
- 最终迟期；
- 义务等级。

不得将其合并为缺乏依据的综合风险分数。

当 `deadline_j != null` 时：

```text
slack[i,j,t] = deadline_j - t
               - estimated_travel[i,j,t]
               - estimated_remaining_service[i,j,t]

best_case_slack[j,t] = max over i in R_feas[j,t] of slack[i,j,t]
overdue[j,t] = max(0, t - deadline_j)
lateness[j] = max(0, completion_time_j - deadline_j)
```

`best_case_slack` 只表示至少一个当前可行平台独立执行该任务时的最佳情形，不保证多任务竞争下按时完成。无可行平台时为负无穷。

当 `deadline_j = null` 时：

- `slack`、`best_case_slack`、`overdue` 和 `lateness` 均为 `NA/null`；
- 不进入截止约束或相关奖励；
- 禁止按零截止时间计算。

当 `max_revisit_interval_j != null` 且存在有效历史参考时：

```text
revisit_age[j,t] = t - last_completion_time_j
revisit_violation[j,t] = max(0, revisit_age[j,t] - max_revisit_interval_j)
```

当 `max_revisit_interval_j = null` 时，不计算重访违约。

当 `last_completion_time_j = null` 时，必须采用并记录以下一种机制：

1. `TRUSTED_HISTORY`：使用可信历史检查记录初始化；
2. `COMMISSIONING_TIME`：使用设施投运时刻初始化；
3. `STUDY_START`：使用研究起始时刻初始化；
4. `INITIAL_INSPECTION_REQUIRED`：释放首次基线任务，完成前不进行时间减法。

禁止由代码自行把空值设为零。

逾期是状态和惩罚，不自动取消任务。`MANDATORY` 表示任务释放后持续构成硬义务，直至完成、正式取消或由认可等效任务替代；不等于必须在当前窗口完成。

只有具有正式依据或明确实验定义时，截止时间和重访间隔才设为硬约束；否则作为软违约并进行敏感性分析。


### 11.1 上层任务调度等待时间

本项严格采用 Zhang 与 Ou（2025）在强化学习任务调度中给出的任务等待时间定义：

$$
WT_j(t)=t-r_j
$$

其中：

- $r_j$ 为任务 $j$ 的到达/释放时刻；
- $t$ 为当前上层调度时刻；
- $WT_j(t)$ 表示任务已进入待调度池但仍未获得有效资源分配的等待时间。

在本项目中，任务 $j$ 于时刻 $s_j$ 获得有效 `ASSIGN_TASK` 后，其最终调度等待时间为：

$$
d_j^{sch}=s_j-r_j
$$

“有效分配”必须同时满足：

- 平台与任务状态合法；
- 硬能力、路径、时间、能源和同一回收点安全返航约束通过；
- 任务未被其他平台占用；
- 对 MAPPO 而言，本次选择未发生同任务冲突；
- 环境正式将任务状态由 `ACTIVE` 或可恢复的 `INTERRUPTED` 更新为 `ASSIGNED`。

若任务在作业窗口结束时 $H$ 仍未获得有效分配，则用于最终评价的截尾等待时间为：

$$
d_j^{sch}(H)=H-r_j
$$

该处理防止算法通过长期不分配困难任务来降低“仅统计已分配任务”的平均等待时间。

等待时间边界严格限定为：

- **起点**：任务正式释放进入上层待调度任务池；
- **终点**：有效平台分配正式生效；
- **不包括**：下层旅行时间、服务时间、退出时间、通信时间和执行反馈时间。

本阶段不设置 `tau_feedback`、`information_age`、`occurred_at/delivered_at` 或下层随机执行时延。原 V1.6 中相关字段与测试由 `V1.7-A9C` 正式撤销。

任务数据结构补充：

```yaml
first_valid_assignment_time: number|null
```

当前等待时间可由 `current_time - release_time` 动态计算，不重复存储为可漂移状态字段。

## 12. 时间、退出边界与能耗

预计任务占用时间：

```text
T_op_hat[i,j,t] = T_travel_hat + T_setup_hat
                  + T_wait_hat + T_service_remaining_hat
                  + T_exit_hat
```

无预计等待时 `T_wait_hat = 0`。

边界定义：

- `T_exit`：从最后一个服务基元离开任务内部几何，抵达任务退出状态；
- `T_recovery`：从任务退出状态前往一个具体合法回收点；
- 若最后服务位置直接作为退出位置，则 `T_exit = 0`，不得与返航重复计算。

任务能耗：

```text
E_op[i,j] = E_travel + E_setup + E_wait
            + E_service + E_exit + E_payload
```

无等待时 `E_wait = 0`。

能源单位保持平台物理含义。归一化能源可作观测，但安全约束和状态更新必须使用物理能源。

厂家标称最大续航仅表示能力上限，不等于安全可调度续航。

## 13. 回收点、安全返航与补能状态更新

最低回收/补能点数据结构：

```yaml
recovery_point_id: string
geometry_ref: string
compatible_platform_types: list
recovery_mode: string
capacity: integer
availability_window: object
service_duration_or_rate: number|object
current_occupancy: list[string]
```

合法回收点集合：

```text
D_i(t) = 与平台类型兼容、处于开放窗口、容量可用且平台依法可到达的回收点
```

平台—任务组合只有在同一个回收点 `d` 同时满足以下条件时才安全可行：

```text
remaining_energy_i >= E_op_hat[i,j,t]
                      + E_recovery_hat[i,j,d,t]
                      + E_reserve[i,t]

t + T_op_hat[i,j,t] + T_recovery_hat[i,j,d,t] <= T_end
```

不得分别使用一个回收点通过能量校验、另一个回收点通过时间校验。

每项任务结束后，平台可直接执行下一任务，但必须根据实际退出位置和实际剩余能源重新校验时间、能量和返航。

执行后能源状态：

```text
e_i(t+) = e_i(t) - E_i_actual
```

必须满足 `0 <= e_i(t+) <= E_i_max`。若实际能耗导致能源为负，应记录安全违约与失败事件，不得通过静默截断掩盖。

连续充电或加油：

```text
e_i(t+) = min(E_i_max, e_i(t) + G_i(delta_t))
```

其中 `G_i(delta_t)` 为平台与补能模式相关的恢复函数。

换电：

```text
e_i(t+) = E_i_swap
```

`E_i_swap` 为配置中定义的换电后能源水平，不得隐含假设为任意常数。

补能事件必须同步更新：

- 平台状态；
- 补能开始时刻；
- 补能完成时刻；
- `available_time`；
- 回收点容量与当前占用；
- 补能完成后的实际能源。

固定点充电、换电或加油属于资源恢复活动，不属于巡检任务。恢复函数必须平台相关。

只有具备明确着舰、锁定和充电/换电能力，并加入会合、同步、容量、等待和占用约束时，移动 USV 才可成为 UAV 回收/补能点。该机制仅为可选扩展，不是基准设定或创新结论。

## 14. 上下层接口与阶段性替代

### 14.1 上层发送

```text
平台及配置
调度任务及原始几何
服务与质量要求
起始/退出上下文
时间、能源、回收点和安全约束
```

### 14.2 下层预计返回

```text
路径可行性
预计旅行时间
预计准备时间
预计等待时间
预计剩余服务时间
预计退出时间
预计任务能耗及分项
预计退出位置
各合法回收点的预计返航时间/能耗
```

### 14.3 下层实际反馈

```text
实际路径
实际旅行/准备/等待/服务/退出时间
实际能耗及分项
实际退出位置
平台剩余能源
完成度
任务族相关质量验收结果
可选归一化质量分数
平台状态
失败原因
中断时的剩余几何
```

上层必须使用实际反馈更新任务、平台和回收点状态，不得继续沿用过期估计值。

### 14.4 下层实现阶段

```text
研究初期：接口兼容的传统规划器
最终方法：下层强化学习策略
```

两者必须使用同一输入输出契约、任务完成标准、安全约束和成本口径。传统规划器用于上层先行研究、环境校验和基线比较，不得替代最终下层强化学习研究。

## 15. 当前完整可行集

`R_feas[j,t]` 必须同时满足：

- 硬能力资格；
- 下层路径可行性；
- 时间可行性；
- 物理能量可行性；
- 同一回收点上的安全返航可行性；
- 平台状态、任务状态和回收点容量兼容。

旧冻结稿中的“第十二项路径结果”按新结构映射为**新第十项：下层路径执行器**。实质依赖不变，仅更新编号。

## 16. 第九项：上层决策机制与求解算法

### 16.1 冻结的研究问题与方法角色

```text
research_target = CENTRALIZED_VS_CTDE_SCHEDULING_ARCHITECTURE
architecture_A = CENTRALIZED_PPO
architecture_B = CTDE_MAPPO
final_architecture = DETERMINED_BY_EXPERIMENT
core_candidate_mode = FULL_FEASIBLE_SET
learned_top_k_role = POST_COMPARISON_EXTENSION
upper_delay_metric = TASK_WAITING_TIME
communication_delay = OUT_OF_SCOPE
feedback_delay = OUT_OF_SCOPE
lower_execution_delay_uncertainty = OUT_OF_SCOPE
```

研究不回答“PPO 与 MAPPO 哪个算法更先进”，而回答：

> 集中式统一调度与 CTDE 分布式协同调度在 UAV-USV 港区巡检中的任务完成、等待、冲突、计算开销和规模适应性边界分别是什么？

两种方法是并列主方案，不预设胜负。实验可以得出：

- 某一架构在全部情景中更优；
- 两种架构分别适合不同任务规模或动态程度；
- 信息条件而非决策结构是主要差异来源；
- 二者性能相近，但工程实施代价不同。

任何结论均须由实验给出。

### 16.2 共同事件驱动调度过程

两种架构共享相同的事件驱动 SMDP。允许触发上层决策的事件包括：

- 作业窗口开始；
- 任务释放、完成、取消、替代或中断；
- 平台变为可用、故障、返航到达或补能完成；
- 回收点开放、关闭、容量或占用变化；
- 下层计划失效或实际执行反馈；
- 经批准的安全中断决策点。

同一物理时刻的事件先合并，再执行：

1. 更新任务、平台、回收点和下层成本估计；
2. 生成完整可行关系与硬动作掩码；
3. 调用 Centralized PPO 或 CTDE-MAPPO；
4. 校验动作、处理冲突并提交有效分配；
5. 推进到下一物理事件。

动作生成本身不推进物理时间。两种架构必须使用同一事件排序和状态转移代码。

### 16.3 共同状态实体与完整可行任务集

完整全局状态表示为：

$$
s_t=\left[X_t^{task},X_t^{platform},X_t^{recovery},X_t^{global},E_t^{typed}\right]
$$

任务特征至少包括：

- 任务族、几何模式、释放方式、义务等级和状态；
- 剩余工作量、完成度和质量验收模板；
- 释放时刻、当前等待时间、截止时间、实时逾期、重访年龄与违约；
- 服务窗口、依赖关系、替代关系和几何摘要；
- 平台相关的预计旅行、服务、能耗和质量可行性。

平台特征至少包括：

- 平台类型、配置、硬能力和载荷；
- 当前/退出位置、状态、可用时刻；
- 物理能源、储备能源、当前任务和合法回收点。

对平台 $i$，完整可行任务集定义为：

$$
\mathcal C_i(t)=\left\{j\mid i\in R_{feas}[j,t],\;status_j\in\{ACTIVE,INTERRUPTED\},\;j\text{未被占用}\right\}
$$

动作集为：

```text
A_i(t) = C_i(t) ∪ {
  WAIT, RETURN, START_REPLENISH,
  CONTINUE_CURRENT, INTERRUPT_CURRENT
}
```

并非所有平台在每个事件都拥有所有非任务动作；环境仍按平台状态和前置条件生成硬掩码。

核心架构对比中，Centralized PPO 与 MAPPO 必须从同一完整可行关系生成候选任务。不得只为其中一方预先裁剪任务。

### 16.4 架构 A：Centralized PPO 集中式统一调度

#### 16.4.1 决策主体和观测

一个中央调度策略观察完整全局状态 $s_t$，统一生成当前所有待决策平台的动作。平台是中央策略管理的异构运输服务资源。

中央状态编码器可使用异构图注意力，但图网络不是独立研究变量；核心比较应尽量与 MAPPO 使用相同的任务、平台和关系特征编码模块。

#### 16.4.2 自回归联合动作

设当前待决策平台集合为 $\mathcal I_t$。中央策略不使用一个具有指数类别数的平铺联合动作，而是自回归生成平台—动作对：

$$
\nu_q=(i_q,a_t^{i_q})
$$

$$
\pi_{\theta}^{C}(\mathbf a_t\mid s_t)
=\prod_{q=1}^{|\mathcal I_t|}
\pi_{\theta}^{C}\left(\nu_q\mid s_t,\nu_{1:q-1}\right)
$$

每选择一个 $\nu_q$ 后立即：

- 从待决策集合中移除平台 $i_q$；
- 临时占用被选择的非联盟任务；
- 更新回收点预留和剩余动作掩码；
- 再选择下一平台—动作对。

因此，Centralized PPO 在动作生成阶段即可消除同批任务独占冲突。平台选择顺序由策略或排列等变候选解码器产生，不使用固定平台 ID 优先级。

#### 16.4.3 PPO 更新

中央价值函数为：

$$
V_{\phi}^{C}(s_t)
$$

采用标准 PPO 裁剪目标：

$$
L_{clip}^{C}=\mathbb E_t\left[
\min\left(r_t^{C}\hat A_t^{C},
\operatorname{clip}(r_t^{C},1-\epsilon,1+\epsilon)\hat A_t^{C}\right)
\right]
$$

其中：

$$
r_t^{C}=\frac{\pi_{\theta}^{C}(\mathbf a_t\mid s_t)}{\pi_{\theta_{old}}^{C}(\mathbf a_t\mid s_t)}
$$

自回归子动作的对数概率求和形成联合动作对数概率。

### 16.5 架构 B：CTDE-MAPPO 分布式协同调度

#### 16.5.1 Actor 与 Critic

每个平台是一个执行时决策主体。联合策略因子化为：

$$
\pi_{\theta}^{D}(\mathbf a_t\mid\mathbf o_t)
=\prod_{i\in\mathcal I_t}
\pi_{\theta_{\tau(i)}}\left(a_t^i\mid o_t^i\right)
$$

其中 $\tau(i)\in\{UAV,USV\}$ 为平台类型。

- 同配置 UAV 共享 UAV Actor 参数；
- 同配置 USV 共享 USV Actor 参数；
- 可采用类型条件统一 Actor，但必须通过平台类型、能力和载荷特征区分异构性；
- `platform_id` 不进入神经网络。

训练阶段使用集中式 Critic：

$$
V_{\phi}^{D}(s_t)
$$

执行阶段不需要 Critic，各平台只依据冻结的信息协议获得 $o_t^i$ 并独立输出动作。

#### 16.5.2 局部观测协议

自然架构对比中的 $o_t^i$ 至少包含：

- 自身平台状态、位置、能源、能力和当前任务；
- 自身完整可行候选任务的属性与成本；
- 环境允许广播的任务状态摘要；
- 可达回收点与资源状态；
- 不包含其他平台的未来动作或中央联合分配结果。

为区分“信息不完整”与“独立动作因子化”的影响，必须增加信息条件控制消融：向 MAPPO Actor 提供与中央策略相同的全局任务/平台摘要，但仍保持每个平台独立输出动作。

#### 16.5.3 MAPPO 更新

对平台 $i$：

$$
r_t^i=\frac{\pi_{\theta_{\tau(i)}}(a_t^i\mid o_t^i)}{\pi_{\theta_{old,\tau(i)}}(a_t^i\mid o_t^i)}
$$

$$
L_{clip}^{D}=\mathbb E_{t,i}\left[
\min\left(r_t^i\hat A_t,
\operatorname{clip}(r_t^i,1-\epsilon,1+\epsilon)\hat A_t\right)
\right]
$$

所有平台共享团队奖励，集中式 Critic 基于全局状态估计价值。若使用按类型 Actor，优势和价值口径仍须统一。

### 16.6 MAPPO 任务冲突与动作提交

设：

$$
c_{j,t}=\sum_{i\in\mathcal I_t}\mathbf 1(a_t^i=j)
$$

对于非联盟任务，当 $c_{j,t}>1$ 时：

- 所有满足 $a_t^i=j$ 的动作均标记为 `CONFLICT_INVALID`；
- 任务 $j$ 保持开放，不进入 `ASSIGNED`；
- 相关平台本轮保持原可决策/等待状态；
- 环境记录冲突任务数、冲突平台动作数和由此产生的额外等待；
- 下一调度事件重新决策。

禁止以下仲裁：

- 选择策略概率最大的 Actor；
- 选择平台—任务匹配度最高者；
- 固定 UAV 或 USV 优先；
- 按实体 ID、Actor 调用顺序或随机顺序选出获胜者；
- 冲突后静默改派第二选择任务。

上述机制会在 MAPPO 后额外加入中央调度逻辑，破坏架构比较。

### 16.7 上层奖励与文献等待时间目标

共同基础奖励保持为：

$$
\begin{aligned}
r_k^{base}={}&w_{prog}\sum_j\omega_j\Delta\rho_{j,k}
+w_{comp}\sum_j\omega_j I_{complete,j,k}\\
&-w_{over}\sum_j\omega_j A_{overdue,j,k}
-w_{late}\sum_j\omega_j I_{complete,j,k}\,lateness_j\\
&-w_{revisit}\sum_j\omega_j A_{revisit,j,k}
-w_{energy}\sum_i\frac{E_{i,k}^{actual}}{E_i^{ref}}\\
&-w_{interrupt}N_{interrupt,k}
-w_{failure}N_{failure,k}
-w_{invalid}N_{invalid,k}.
\end{aligned}
$$

等待时间不再采用 V1.6 的事件暴露积分。严格依据文献中的 $WT_j(t)=t-r_j$，在任务获得有效分配的决策事件中加入：

$$
r_k^{wait}=-w_{wait}\sum_{j\in\mathcal J_k^{new\_assigned}}\omega_j WT_j(t_k)
$$

其中 $\mathcal J_k^{new\_assigned}$ 为本决策事件首次获得有效分配的任务集合。

作业窗口结束时，对仍未分配任务加入终局等待责任：

$$
r_H^{wait}=-w_{wait}\sum_{j\in\mathcal J_H^{unassigned}}\omega_j(H-r_j)
$$

因此一个窗口内累计的等待项等价于最小化全部释放任务的加权总等待时间：

$$
TWT=\sum_{j\in\mathcal J^{assigned}}\omega_j(s_j-r_j)
+\sum_{j\in\mathcal J_H^{unassigned}}\omega_j(H-r_j)
$$

总奖励为：

$$
r_k=r_k^{base}+r_k^{wait}+r_k^{conflict}
$$

对 MAPPO：

$$
r_k^{conflict}=-w_{conflict}N_{conflict\_actions,k}
$$

Centralized PPO 的冲突动作数理论上应为零；若非零说明硬掩码或提交逻辑错误，不应作为可接受策略行为。

等待时间是次级目标，不能通过提高 $w_{wait}$ 牺牲 MANDATORY 任务完成、安全和质量。具体权重及多目标选择规则留待第十二项统一敏感性实验，但两种架构必须使用相同数值。

### 16.8 Top-K 与任务波次的控制边界

#### 16.8.1 核心比较

核心比较固定：

```text
candidate_mode = FULL_FEASIBLE_SET
```

Centralized PPO 与 MAPPO 均直接基于 $\mathcal C_i(t)$ 选择任务。

#### 16.8.2 共同规则 Top-K 控制实验

若任务规模过大，可增加环境级、非学习的共同规则 Top-K：

$$
\mathcal K_t=f_{rule}(\mathcal T_t)
$$

两种架构使用完全相同的 $\mathcal K_t$，且保护任务、逾期责任和终局责任仍覆盖完整任务集。

#### 16.8.3 学习型软锁定 Top-K

原 V1.4/V1.5 的学习型软锁定波次不删除代码设计，但其角色改为：

- 核心架构比较完成后的扩展方法；
- 不能用于决定 Centralized PPO 与 MAPPO 哪种调度架构更优；
- 若由中央波次选择器为两种架构共同生成任务集合，论文只能声称比较“共同中央候选组织下的波次内联合/分布式分配”；
- 若只接入 Centralized PPO，则只能评价中央架构的扩展收益。

### 16.9 两层公平比较实验

#### 16.9.1 自然架构对比

- Centralized PPO：完整全局状态、中央联合动作；
- MAPPO：规定局部观测、CTDE训练、分布执行。

该实验比较真实的信息组织与决策架构整体，不属于纯算法控制变量比较。

#### 16.9.2 信息条件控制消融

- Centralized PPO 保持全局状态；
- MAPPO Actor 增加统一广播的全局任务和平台摘要；
- MAPPO 仍独立输出动作，冲突规则不变。

该实验用于判断性能差异主要来自信息可见性，还是来自中央联合动作与分布式独立动作本身。

### 16.10 必须报告的指标

#### 任务与时间指标

$$
R_{complete}=\frac{N_{complete}}{N_{released}}
$$

$$
R_{mandatory}=\frac{N_{mandatory,complete}}{N_{mandatory,released}}
$$

$$
R_{overdue}=\frac{N_{overdue}}{N_{tasks\ with\ deadline}}
$$

已分配任务平均等待：

$$
\bar d_{assigned}^{sch}=\frac{1}{N_{assigned}}
\sum_{j\in\mathcal J_{assigned}}(s_j-r_j)
$$

全部释放任务截尾平均等待：

$$
\bar d_{all}^{sch}=\frac{1}{N_{released}}\left[
\sum_{j\in\mathcal J_{assigned}}(s_j-r_j)
+\sum_{j\in\mathcal J_H^{unassigned}}(H-r_j)\right]
$$

同时报告 `P50/P90/P95` 等待时间和 EVENT/MANDATORY 任务分组等待时间。

#### 协同指标

$$
R_{conflict\_batch}=\frac{N_{decision\ batches\ with\ conflict}}{N_{decision\ batches}}
$$

$$
R_{conflict\_action}=\frac{N_{conflicting\ agent\ actions}}{N_{agent\ actions}}
$$

$$
R_{invalid}=\frac{N_{invalid\ actions}}{N_{agent\ actions}}
$$

#### 计算与学习指标

- 平均与 P95 单次调度推理时间；
- 每秒可处理决策事件数；
- 模型参数量和峰值显存；
- 达到给定完成率所需环境交互数量；
- 多随机种子均值、标准差和置信区间；
- 训练崩溃或明显不收敛种子比例。

#### 规模测试

至少覆盖：

- 2、4、6 个平台，并扩展一个更大规模档；
- 16、32、64 个任务，并扩展一个更大规模档；
- 不同 UAV/USV 数量比例；
- 低、中、高动态任务到达率；
- 训练规模内测试与跨规模测试。

### 16.11 允许的研究假设与禁止的预设结论

可以在实验前提出：

- 中央联合动作可能在小规模下冲突更少；
- MAPPO 的因子化动作可能降低单个 Actor 的动作维度；
- MAPPO 可能产生更多任务竞争；
- 中央自回归解码的推理时间可能随平台和候选数量增长；
- 参数共享和可变规模编码可能改善 MAPPO 的跨规模能力。

不得提前写成结论：

- MAPPO 一定更适合动态任务；
- MAPPO 一定具有更低等待时间或更强韧性；
- Centralized PPO 一定无法扩展；
- Centralized PPO 一定在所有情景完成率更高；
- 单点故障或通信依赖已经通过实验得到验证。

### 16.12 Codex 最低实现与测试

#### 共同环境测试

- 两种算法从同一状态快照生成候选集；
- 硬能力、路径、时间、能源和安全返航掩码完全一致；
- 相同动作提交后产生相同环境状态转移；
- `null` 时间字段不被当作零；
- 任务未分配时等待时间等于 `current_time - release_time`；
- 有效分配后等待时间停止增长；
- 窗口结束未分配任务计入截尾等待。

#### Centralized PPO 测试

- 每个待决策平台在一个批次中恰好获得一个动作；
- 同一非联盟任务不会被分配两次；
- 自回归概率和联合对数概率计算一致；
- 平台排列变化不应由 ID 特征导致系统性偏差。

#### MAPPO 测试

- Actor 执行时不读取集中式 Critic 输入；
- 同类型平台参数共享正确，类型特征有效；
- 多平台选择同一任务时所有相关动作无效；
- 冲突后不静默改派、不选择概率最大者；
- 冲突任务保持开放并继续累计等待；
- 自然观测与信息控制消融的接口明确分离。

#### 公平性回归测试

- 相同任务释放序列与随机种子；
- 相同下层成本缓存和估计版本；
- 相同奖励权重与终局责任；
- 相同环境交互预算；
- 完整可行集核心对比中不存在架构专属 Top-K；
- 评估器同时输出完成、等待、冲突、无效、计算和规模指标。

### 16.13 文献依据与采用边界

直接采用：

- Zhang, W. and Ou, H. (2025), *Reinforcement learning based multi objective task scheduling for energy efficient and cost effective cloud edge computing*, Scientific Reports, 15, 41716. DOI: 10.1038/s41598-025-25666-1。直接采用其公式 $WT(T_k)=t_{current}-t_{arrival}(T_k)$ 以及总/平均等待时间评价思想。
- Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS Datasets and Benchmarks。用于 MAPPO 的 CTDE、集中式价值函数和分散 Actor 基础结构。
- Bagchi, Nair and Das (2024), *On a dynamic and decentralized energy-aware technique for multi-robot task allocation*, Robotics and Autonomous Systems, 180, 104762. DOI: 10.1016/j.robot.2024.104762。用于支持动态任务到达条件下的分散式任务分配、冲突解决需求和等待时间评价。
- Dai et al. (2025), *Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning*, IEEE Robotics and Automation Letters, 10(3), 2654-2661. DOI: 10.1109/LRA.2025.3534682。用于支持异构多机器人分散式策略、能力条件决策和规模泛化的研究方向。
- Ren et al. (2024), *A multi-objective fuzzy programming model for port tugboat scheduling based on the Stackelberg game*, Scientific Reports, 14, 25057. DOI: 10.1038/s41598-024-76898-6。用于支持港口动态任务通知、计划开始时间与等待/缓冲时间的上层调度语义。

项目公平性协议而非文献原公式：

- MAPPO 同任务冲突全部判无效；
- 核心比较使用完整可行任务集；
- 设置自然架构对比和信息控制消融；
- 未分配任务采用窗口结束截尾等待。

这些规则用于隔离架构差异，不应在论文中伪称为上述论文的原始算法。

## 17. 后续冻结门

### 第十项

冻结点、线、面任务的 UAV/USV 下层强化学习执行器、传统规划临时替代和对两种上层架构一致的成本接口。

### 第十一项

冻结上下层联调、临时规划器替换、估计版本失效、失败反馈和数据隔离。不得引入本版已排除的反馈时延或随机执行成本，除非正式 `AMENDS`。

### 第十二项

冻结两种上层架构的公平超参数、模型容量、训练预算、随机种子、任务规模、动态到达率、信息条件消融、等待时间权重、统计检验、适用边界结论、扩展 Top-K 结果、最终创新点和论文题目。
