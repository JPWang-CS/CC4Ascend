# AscendC Expert Memory Index

- [核间同步专家技巧](AscendC-expert/expert-cross-core-sync.md) — A5 CrossCoreSetFlag/WaitFlag 死锁排查与同步模式
- [性能调优专家技巧](AscendC-expert/expert-performance-tuning.md) — A5 性能反模式排查清单与通用分析流程
- [Cube-Vector 融合专家技巧](AscendC-expert/expert-cv-fusion.md) — L0C2UB/UB2L1 直连通路使用与同步时序
- [Tiling 策略专家技巧](AscendC-expert/expert-tiling-patterns.md) — TilingKey 设计原则、双缓冲判定、分核策略、量化 Tiling
- [常见陷阱速查](AscendC-expert/expert-common-pitfalls.md) — 类型推导、A5 迁移、命名约定、aclnn 调用陷阱

## A2A3 专家技巧
- [A2A3 分核策略](AscendC-expert/a2a3/a2a3-core-split.md) — GMM 对角线分核、Attention S1/S2/B 多维分核、QuantGMM 2D Split
- [A2A3 流水同步](AscendC-expert/a2a3/a2a3-pipeline-sync.md) — 三缓冲流水线、ProcessVec1 事件链、GM Workspace 双缓冲布局
- [A2A3 双缓冲](AscendC-expert/a2a3/a2a3-double-buffer.md) — BigDoubleBuffer 控制、TQue 双缓冲、Triple Buffer 场景、Pre-deferred MMCompute
- [A2A3 Tiling 优化](AscendC-expert/a2a3/a2a3-tiling-optimization.md) — TilingKey 编码、MTE2 最小化分核、GEMV 阈值切换、workspace-split

## A5 专家技巧
- [A5 分核策略](AscendC-expert/a5/a5-core-split.md) — 三种分核模式（顺序/对称/正倒序循环）、ASWT 对角线分组、容量反推 Tiling
- [A5 流水同步](AscendC-expert/a5/a5-pipeline-sync.md) — 四阶流水线、CrossCoreSetFlag/WaitFlag、cgmct 框架、TSCM 双缓冲
- [A5 Regbase 优化](AscendC-expert/a5/a5-regbase-optimization.md) — VF 计算模式、LoadDist/StoreDist、Norm+RoPE+Cache 融合
- [A5 SIMT 优化](AscendC-expert/a5/a5-simt-optimization.md) — SIMT vs SIMD 对比、线程独立分支、适用场景

## 迁移专家技巧
- [Attention 迁移](AscendC-expert/migration/attention-migration.md) — FlashAttention TilingKey 简化、分核模式变化、Buffer 容量、CV 通路、IFA Regbase 化
- [GMM 迁移](AscendC-expert/migration/gmm-migration.md) — cgmct 框架、通信 AICPU→CCU、SwiGLU V1→V2、模板参数扩展
- [PosEmbedding 迁移](AscendC-expert/migration/posembedding-migration.md) — Membase→Regbase、TilingKey 体系变化、融合算子流水线、迁移 Checklist
