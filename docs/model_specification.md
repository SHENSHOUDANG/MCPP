# docs/model_specification.md

> 规范版本：V1.2

## 1. 研究边界

系统服务于港区运营、通航安全、海道维护和水侧设施检查，不研究陆侧通用巡检、无差别全覆盖或端到端底层控制。

最终研究架构为双层强化学习：上层学习任务分配、排序、返航与补能决策，下层学习点、线、面任务的旅行与服务路径执行。研究初期可用传统规划器临时替代下层，以先行训练和验证上层、稳定成本接口并建立对照基线；该替代不改变最终双层强化学习路线。具体算法名称、训练方式和论文题目仍需后续冻结。

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
first_valid_assignment_time: number|null
```

约束：

- 周期任务必须具有 `period_interval`、`next_due_time` 和 `calendar_update_mode`。
- `FIXED_CALENDAR` 必须具有 `calendar_anchor`。
- `importance_class`、`replenishment_mode` 和 `recovery_mode` 的取值必须来自获批配置注册表；规范未冻结时不得自行补默认值。
- `service_window_start/end = null` 表示没有额外服务窗口限制，不得按零处理。
- 非周期任务的周期字段应为 `null`。
- 可缓存质心坐标作为特征，但不得替代 `geometry_ref`。
- `first_valid_assignment_time` 仅在任务获得有效上层平台分配时写入；未获得有效分配时保持 `null`。

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

本项目按 V1.7-A9C 仅保留上层任务等待时间：

```text
WT_j(t) = t - release_time_j
d_sch_j = first_valid_assignment_time_j - release_time_j
```

有效分配必须同时满足平台与任务状态合法、硬能力、路径、时间、能源和同一回收点安全返航约束通过，且环境正式将任务状态由 `ACTIVE` 或可恢复的 `INTERRUPTED` 更新为 `ASSIGNED`。有效分配后等待时间停止增长。

若任务在作业窗口结束 `H` 时仍未获得有效分配，则最终评价使用截尾等待：

```text
d_sch_j(H) = H - release_time_j
```

等待时间边界严格限定为：

- 起点：任务正式释放进入上层待调度任务池；
- 终点：有效平台分配正式生效；
- 不包括：下层旅行时间、服务时间、退出时间、通信时间和执行反馈时间。

本阶段不设置 `tau_feedback`、`information_age`、`occurred_at/delivered_at`、`first_service_start_time` 或下层随机执行时延。上述 V1.6 字段已由 V1.7-A9C 撤销，不得在代码、数据或测试中恢复。

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

## 16. 后续冻结门

### 第九项

上层状态、动作、掩码、冲突处理、目标/奖励和求解算法。

### 第十项

UAV/USV 点、线、面任务的下层强化学习状态、动作、奖励、约束、训练方法与执行策略；同时规定研究初期传统规划器的临时替代边界和对照用途。

### 第十一项

上下层分阶段训练与联合联调、临时规划器向下层强化学习策略的替换、成本调用、失败反馈和数据隔离。

### 第十二项

基线、消融、指标、最终创新点和最终论文题目。
