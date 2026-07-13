---
name: ascendc-development
description: AscendC 算子开发全流程与 A5（Ascend950）新编程范式、以及 A2A3→A5 跨平台迁移方法论。涵盖 AI Core 算子开发流程（工程创建/算子定义/Tiling/Kernel/aclnn 适配/编译/UT 验证）、跨平台迁移指南（硬件能力变更、搬运/计算/存储单元差异、迁移步骤、A5 性能调优 FAQ）、以及 A5 专属四大范式：Regbase（MicroAPI/RegTensor 寄存器编程）、SIMT（线程级并行，gather/scatter 离散访存）、Cube-Vector 融合通路（UB2L1/L0C2UB 直连）、CCU 通信适配（CCU1.0 替代 AICPU）。当要新建算子、写 Tiling/Kernel 骨架、把算子从 910B 迁到 950、或用 Regbase/SIMT/CV 融合/CCU 优化时调用。（本 skill 是跨平台迁移通用方法论 + A5 范式；某类算子的具体迁移/分核/流水细节见 ascendc-operators）
---

# AscendC 算子开发与 A5 编程范式

本 skill 覆盖算子开发全流程，以及从 A2A3 迁移到 A5 时需要掌握的新编程范式与适配方法。新建算子或做迁移/优化时按下表查阅。

> 芯片硬件规格（核数/L0C/UB/Bias 容量、Fixpipe/Sparsity 支持）见 `ascendc-hardware` skill 与 agent memory `hardware-specs.md`。

## 明细文档

### 通用开发与迁移
- [AI Core 算子开发指南](aicore_develop_guide.md) — 开发全流程、工程目录结构、算子定义/Tiling/TilingKey/TilingData、Kernel 入口与算子类、Process 主流程、aclnn 适配、编译部署、UT 验证模板（A2A3 + A5 通用）。
- [算子跨平台迁移指南](cross_platform_migration_guide.md) — A2A3 vs A5 硬件对比、搬运/计算/存储单元变更与适配方案、迁移步骤、A5 性能不升反降排查 FAQ。

### A5 专属编程范式
- [Regbase 编程范式](regbase编程范式.md) — MicroAPI/RegTensor 寄存器编程、Membase vs Regbase 对比、掩码/搬运/Gather API、LoadDist 分发模式、适用场景。
- [SIMT 编程范式](simt编程范式.md) — SIMT vs SIMD 编程模型与对比、gather_v2 选路（尾轴 ≤2048 用 SIMT）、迁移注意事项。
- [Cube-Vector 融合通路](cv融合通路.md) — UB2L1 / L0C2UB 直连通路、切 K 累加消除 GM 往返、后处理融合管线、同步时序建议。
- [CCU 通信适配](ccu通信适配.md) — A5 CCU1.0 替代 AICPU、Eager/Graph 模式适配、核间同步严格匹配要求与死锁排查、mode 3 新同步模式。
