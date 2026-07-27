# QBMM 迁移至 omni-ops — 文档索引

> 状态：迁移主体已完成（v3/v4 源码 + torch binding + golden 多分支已验）。本文档为**设计/索引**文档，进度状态以 [`施工进度.md`](施工进度.md) 为准。
> 最后更新：2026-07-27

## 文档列表

| 文档 | 说明 |
|------|------|
| [项目需求.md](项目需求.md) | 项目需求文档，包含功能/非功能需求、验收标准、风险限制 |
| [项目结构变动分析.md](项目结构变动分析.md) | matmul 从 ops-nn 到 omni-ops 的结构变化分析 |
| [迁移计划.md](迁移计划.md) | 完整迁移计划（设计态），包含决策汇总、逐层修改、命名落地 |
| [修改清单.md](修改清单.md) | 详细修改清单（设计态），包含每个文件的具体修改点 |
| [施工进度.md](施工进度.md) | **实际施工进度**：已完成项+证据等级 / disabled+根因 / 待办编排 / 通路覆盖矩阵 |
| [nn仓aclnn分支差异分析.md](nn仓aclnn分支差异分析.md) | nn 仓 aclnn 文档分支与 omni-ops 已迁内容差异分析 |

---

## 项目概述

**目标**: 将 QBMM (QuantBatchMatmul V3/V4) 算子从 ops-nn 迁移至 omni-ops

**芯片**: A2A3 only (Ascend 910B/910C)

**参考**: ai_infra_matmul (已迁移并发货)

---

## 快速参考

### 命名变化

| 原名 | 新名 |
|------|------|
| QuantBatchMatmulV3 | AiInfraQuantBatchMatmulV3 |
| aclnnQuantMatmulV3 | aclnnAiInfraQuantMatmulV3 |

### 关键决策

- 只支持 A2A3，**整删** A5(arch35,对 A2A3 死代码)/310P(arch20)/Kirin
- 加 AiInfra 前缀，全改名（⚠ tiling 注册 key = 算子名 `"QuantBatchMatmulV3"`，非类名）
- 删 arch20 4 处 + arch35 3 处 tiling.cpp 挂钩；删 prio1 须同步 `registerList{0,1,2}→{0,2}`（否则 operator[] 崩）
- kernel 无 `add_kernel_sources`（靠 func.cmake glob）；common 在 `op_host/op_api/`；torch `m.def` 进 `ops_def_registration.cpp`
- v3/v4 def config 结构相反（v4 留 config_910BC）；KB (runtime_kb) 只 v3 有，改名 sed
- **torch 前置**：aclnn 编完即接，边迁边对 golden 验证（P4 验 V3 路径 / P6 验 V4 路径）

### 主要删除

- arch20/ (310P)
- arch35/ (A5)
- ascend950/kirin/ascend310p/ascend350 config

### 新增 common

- log_format_util.h
- quant_matmul_v4.{h,cpp}

---

## 验证清单

> 实际验证状态以 [`施工进度.md`](施工进度.md) §1-§2 为准。下表为设计期验证清单，保留作 design intent。

- [x] cmake configure 通过
- [x] cmake --build --target optiling 通过
- [x] bisheng kernel 编译通过
- [x] grep 无残留旧名
- [~] UT 运行通过 (N>0)（omni-ops 下 v3/v4 无 tests/ 目录，待补；golden.py 已代行验证）
- [x] torch 早接后 P4 验 V3 路径、P6 验 V4 路径对 golden（源 ops-nn v3/v4 tests/assets/golden.py）— V3 全分支已通，V4 perblock 单例已通

---

*最后更新: 2026-07-27*
