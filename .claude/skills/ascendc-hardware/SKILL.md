---
name: ascendc-hardware
description: Ascend 芯片硬件规格索引与设计影响归纳。**权威源是 `AscendC_platform/*.ini`，本 skill 不维护第二份平行规格真值表**。用于定位目标 SoC 对应 ini、理解关键字段（ai_core_cnt / l0 / l1 / ub / fb0 / l2 / memory_size 等）及其对 Tiling、buffer 预算、分核、Kernel 优化的影响。当要确定 tile 尺寸、估算 buffer 占用、判断某硬件特性是否可用、或对比不同 SoC/variant 做迁移决策时调用。
---

# Ascend 芯片硬件规格（权威源：`AscendC_platform/*.ini`）

本 skill **不再维护第二份手写规格真值表**。芯片原始数字的唯一权威源是项目根目录下的：

- `AscendC_platform/*.ini`

本 skill 的职责是：
1. 帮你找到目标 SoC 对应的 ini
2. 告诉你该看哪些 section / field
3. 解释这些字段对 Tiling / Kernel 设计意味着什么

> 规则：**原始数字只认 ini，不认记忆里的旧表。** 如果 memory、旧文档、历史讨论和 ini 冲突，一律以当前 ini 为准。

## 1. 先找对 SoC / variant

不要笼统地问“950 是多少核”。A5 / 950 有多个 variant（如 `950DT_*`、`950PR_*`），不同 ini 的数字可能不同。

常见入口示例：
- A2A3 / 910B 系列 → `Ascend910B*.ini`
- A3 910_93 系列 → `Ascend910_93*.ini`
- A5 / 950 DT 系列 → `Ascend950DT_*.ini`
- A5 / 950 PR 系列 → `Ascend950PR_*.ini`

**第一步永远是先确认本次需求的目标 SoC / variant。**

## 2. 重点 section

典型 ini 里优先看这些 section：

- `[SoCInfo]`
  - SoC / core 总体信息
- `[AICoreSpec]`
  - AI Core / Cube / Vector / buffer / tile 相关关键规格
- `[AICoreMemoryRates]`
  - 片上数据通路速率信息
- `[SoftwareSpec]`
  - 软件/路径相关特征
- `[DtypeMKN]`
  - 某些 dtype/cube 形状相关信息（需要时看）

## 3. 关键字段索引

### 3.1 核数 / 执行单元

优先看：
- `ai_core_cnt`
- `cube_core_cnt`
- `vector_core_cnt`
- `cube_vector_combine`
- `cube_freq`
- `vec_calc_size`

**设计影响：**
- 决定总并行度上限
- 影响分核策略和是否值得做特殊 split
- 决定“更多核摊开”还是“每核做大 tile”的倾向

### 3.2 片上 buffer / tile 上限

优先看：
- `l0_a_size`
- `l0_b_size`
- `l0_c_size`
- `l1_size`
- `ub_size`
- `ubblock_size`
- `ubbank_size`
- `fb0_size`
- `fb1_size`
- `bt_size`

**设计影响：**
- 决定 tile 尺寸上限
- 决定双缓冲 / 三缓冲是否可行
- 决定 TilingData 里 buffer 预算能否站得住
- `fb0_size` 等字段也会提示某些特性/通路是否存在，不要只凭旧印象说“某代有/没有某能力”

### 3.3 外存 / cache / 总量

优先看：
- `memory_size`
- `l2_size`
- `l2_type`

**设计影响：**
- 决定大 shape 下的数据复用压力
- 决定是更强调片上复用还是可接受更多外存往返
- 影响某些跨 tile 策略是否划算

### 3.4 cube 形状 / 指令粒度

优先看：
- `cube_m_size`
- `cube_n_size`
- `cube_k_size`
- `vec_calc_size`

**设计影响：**
- 决定最自然的 tile 对齐方式
- 影响 kernel / tiling 在 M/N/K 方向的切分粒度
- 影响某些 dtype/path 的 fast path 设计

### 3.5 数据通路速率

优先看：
- `l1_to_l0_a_rate`
- `l1_to_l0_b_rate`
- `l1_to_ub_rate`
- `l0_c_to_ub_rate`
- `ub_to_l2_rate`
- `ub_to_l1_rate`
- `l2_read_rate`
- `l2_write_rate`

**设计影响：**
- 帮助判断瓶颈更像算力不足还是搬运不足
- 帮助判断某个融合/直连/中转方案是否值得
- 对 kernel 优化和 pipeline 判断很关键

## 4. 怎么把 ini 读成设计结论

### 4.1 给 `ascendc-tiling-expert`
重点关心：
- `l0_*_size`
- `l1_size`
- `ub_size`
- `ai_core_cnt`
- `cube_*_size`

输出要落到：
- tile 尺寸
- split / 分核
- TilingData 字段预算
- TilingKey 选择依据
- 双/三缓冲能否站住

### 4.2 给 `ascendc-kernel-expert`
重点关心：
- buffer 容量
- 数据通路速率
- cube/vector 规模
- 特定通路相关字段（如 `fb0_size`）

输出要落到：
- pipeline 是否可能打满
- 哪类中转/融合/fast path 有意义
- 某个优化是正优化还是负优化

### 4.3 给 `ascendc-architect`
重点关心：
- variant 差异
- 通路/容量约束
- 是否需要在方案里显式写 scope in/out

输出要落到：
- 方案约束
- 迁移风险
- 哪些层需要协同（host / tiling / kernel / semantics）

## 5. 使用原则

- **不要写死单一“950=某数字”**，先确认 variant
- **不要把 skill 当真值表**，skill 只是索引和解释层
- **不要用旧 memory 覆盖当前 ini**
- **先读 ini，再给 tile / split / kernel 结论**

## 6. 与其他 skill 的边界

- 具体 API 用法 → `ascendc-api`
- dtype / quant / broadcast / transpose 语义 → `ascendc-data-context`
- 开发流程 / A5 范式 / 迁移方法 → `ascendc-development`
- 算子类 split / TilingKey / pipeline 先例 → `ascendc-operators`
- profiling / cannsim / msprof → `ascendc-debug`

本 skill 只负责：
> **把 `AscendC_platform/*.ini` 变成可用于设计判断的硬件真值索引与解释层。**