# 洋山港任务信息补齐报告

## 处理结果

- 输入GeoPackage：`洋山港全港区正式GIS训练任务数据_V2(1).gpkg`
- 补齐后的GeoPackage：`洋山港全港区正式GIS训练任务数据_补齐版.gpkg`
- 固定任务CSV：`洋山港固定巡检任务节点_补齐版.csv`
- 动态任务种子CSV：`洋山港动态任务种子_补齐版.csv`
- 更新后的QGIS工程：`洋山港任务工程_补齐版.qgz`

检测到39个新增空属性节点。

## 数据规模

- 航标/通航保障固定任务：14个
- 水侧结构固定任务：205个
- 其中人工新增并补齐属性：39个
- 固定任务CSV记录数：219条

## 新增节点属性补齐规则

- `BW_MANUAL_01`：breakwater，15个节点，任务编号 B180—B194。
- `BW_MANUAL_02`：breakwater，4个节点，任务编号 B195—B198。
- `BP_MANUAL_01`：bridge_pier_or_bridge_water_structure，13个节点，任务编号 B199—B211。
- `MF_MANUAL_01`：berth_or_mooring_facility，5个节点，任务编号 B212—B216。
- `PJ_MANUAL_01`：pier_or_jetty，1个节点，任务编号 B217—B217。
- `PJ_MANUAL_02`：pier_or_jetty，1个节点，任务编号 B218—B218。

所有新增节点统一补齐了：

- `task_id`
- `task_class`
- `facility_type`
- `side_zone`
- `source_dataset`
- `source_feature_id`
- `verification_level`
- `parent_length_m`
- `initial_visible`
- `source_kind`
- `generation_method`
- `parent_id`
- 经纬度与UTM坐标
- `manual_added`
- `attribute_status`
- `remark`

## 坐标更新

所有水侧结构节点（包括你移动过的旧节点）均重新依据当前几何计算：

- `longitude`
- `latitude`
- `x_utm`
- `y_utm`

计算采用：

- 地理坐标：EPSG:4326
- 米制坐标：EPSG:32651

## 自动检查

- 水侧结构任务ID重复数：0
- 必填字段NULL统计：`{"task_id": 0, "task_class": 0, "facility_type": 0, "side_zone": 0, "source_dataset": 0, "source_feature_id": 0, "verification_level": 0, "parent_length_m": 0, "initial_visible": 0, "source_kind": 0, "generation_method": 0, "parent_id": 0, "longitude": 0, "latitude": 0, "x_utm": 0, "y_utm": 0, "attribute_status": 0}`
- 文本必填字段空字符串统计：`{"task_id": 0, "task_class": 0, "facility_type": 0, "side_zone": 0, "source_dataset": 0, "source_feature_id": 0, "verification_level": 0, "source_kind": 0, "generation_method": 0, "parent_id": 0, "attribute_status": 0}`
- GeoPackage图层数：10

## 需要你重点人工复核的内容

以下分类依据新增节点的空间分组和排列形态推断，不是官方设施台账：

1. `BP_MANUAL_01` 被暂定为桥墩或桥梁临水结构；
2. `MF_MANUAL_01` 被暂定为泊位或系泊设施；
3. `PJ_MANUAL_01`、`PJ_MANUAL_02` 被暂定为独立突堤或引桥式设施。

如果你在QGIS中确认其中某组属于不同设施，只需修改该组的：

- `facility_type`
- `parent_id`
- `remark`

坐标和任务编号无需再次修改。
