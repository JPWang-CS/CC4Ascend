---
name: ascendc-development
description: AscendC 算子开发全流程与 A2A3→A5 跨平台迁移方法论（含 A5 四大范式：Regbase / SIMT / Cube-Vector 融合 / CCU）。涵盖 AI Core 算子开发全流程（工程创建→算子定义→Tiling→Kernel→aclnn 适配→编译部署→UT 验证）、AI CPU 算子开发、图模式适配、以及跨平台迁移指南（硬件能力变更、搬运/计算/存储单元差异、迁移步骤、A5 性能 FAQ）。当要新建算子、写 Tiling/Kernel 骨架、做图模式交付件、把算子从 910B 迁到 950、或用 Regbase/SIMT/CV 融合/CCU 适配时调用。（本 skill 是跨平台迁移通用方法论 + A5 范式；某类算子的具体分核/流水/TilingKey 细节见 ascendc-operators）
---

# AscendC 算子开发与 A2A3→A5 迁移

本 skill 覆盖算子开发全流程和跨平台迁移方法论。真实源 = `ops-transformer_AI/docs/zh/develop/`。

> 硬件规格（核数/L0C/UB 容量）以 `AscendC_platform/*.ini` 为准，见 `ascendc-hardware`。

## 真实文档映射

| 真实文档 | 本 skill 对应 | 内容 |
|---|---|---|
| `docs/zh/develop/aicore_develop_guide.md` (1068行) | [AI Core 开发指南](aicore_develop_guide.md) | 工程创建→算子定义→Tiling→Kernel→aclnn→编译→UT 全流程 |
| `docs/zh/develop/aicpu_develop_guide.md` | （并入开发指南说明） | AI CPU 算子开发，`--genop_aicpu` |
| `docs/zh/develop/graph_develop_guide.md` | （见 ascendc-operator-invocation 的 GE graph 通路） | 图模式交付件、InferShape/InferDataType、REG_OP |
| `docs/zh/develop/cross_platform_migration_guide.md` (453行) | [跨平台迁移指南](cross_platform_migration_guide.md) | **A5 四大范式真身都在这里**：硬件变更、迁移步骤、SWAT、SIMT、Regbase、CV 融合、CCU、核间同步 |

## A5 四大范式速查（内容真身均在迁移指南内）

> 以下 4 个速查文件是从迁移指南**抽取的快速查阅版**，权威以迁移指南为准：

- [Regbase 编程范式](regbase编程范式.md) — MicroAPI/RegTensor/MaskReg/LoadDist；速查摘录，详见迁移指南 Regbase 章节
- [SIMT 编程范式](simt编程范式.md) — 线程级并行、gather_v2 选路（尾轴≤2048 用 SIMT）；速查摘录
- [Cube-Vector 融合通路](cv融合通路.md) — UB2L1 / L0C2UB / FIXP→UB 直连、核间同步 mode 3；速查摘录
- [CCU 通信适配](ccu通信适配.md) — AICPU→CCU1.0、Eager/Graph 双模式适配；速查摘录

## 开发 vs 迁移 vs 范式的边界

- **新建算子**：先看 AI Core 开发指南的全流程
- **迁移到 A5**：先看迁移指南的硬件变更表 + 迁移步骤，再按算子类查具体范式
- **某类算子的分核/流水/TilingKey 先例**：见 `ascendc-operators`（本 skill 不重复）
- **具体 API 怎么调**：见 `ascendc-api`
- **编译/安装/UT 命令**：见 `ascendc-install`