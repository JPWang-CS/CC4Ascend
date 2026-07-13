---
name: ascendc-operators
description: ops-transformer 七大类算子的通用设计范式、分核策略、流水同步、TilingKey 体系与 A2A3/A5 差异。涵盖 Attention（FlashAttention/MLA/IFA/BlockSparse 分核与流水、TilingKey 位域、Softmax 机制，含 A5 Regbase/MX 全量化实现细节）、GMM 分组矩阵乘（对角线/ASWT 分核、量化 scale UB 预算、三阶流水、Atomic Add）、MoE（gating→routing→permute→compute 数据流、arch35 分布）、MC2 通信计算融合（通信+MatMul、Eager/Graph、CCU）、FFN、MHC（Sinkhorn）、PosEmbedding（RoPE 三模式、broadcast 后缀、20000+ TilingKey、融合算子）。当要设计/改造/迁移这些算子、查某类算子的分核或流水或 Tiling 范式时调用。（本 skill 是各算子的具体范式/迁移细节；通用迁移方法论与 Regbase/SIMT/CV/CCU 编程范式见 ascendc-development）
---

# ops-transformer 算子通用范式

本 skill 收录七大类算子的架构范式：分核策略、流水同步、TilingKey 体系、量化/UB 预算、A2A3 vs A5 差异。设计、改造或迁移某类算子时先读对应通用范式文档。

## 明细文档

### 按算子类
- [Attention 通用范式](attention通用范式.md) — 53 个 Attention 算子：A2A3 四种拆分体系 + A5 三种新分核、三/四阶流水同步事件链、Same-AB 跨核同步、A2A3(64bit)/A5(62bit) TilingKey 位域、Softmax 特殊机制、IFA Regbase 化、GM Workspace 与 UB 布局。
- [A5 Attention 实现细节](a5-attention实现细节.md) — A5 arch35 模块化结构（kernel/service/matmul_modules/vector_api）、Regbase 入口、MX FullQuant、原生双架构算子 vs A5 专属算子清单。
- [GMM 通用范式](gmm通用范式.md) — 分组矩阵乘：对角线分核(A2A3)/ASWT(A5)/2D Split、GEMV 阈值、量化 scale UB 预算公式、三阶流水(A8W4 MSD)、Pre-deferred MMCompute、Atomic Add、V1→V2 演进。
- [MoE 通用范式](moe通用范式.md) — 24 个 MoE 算子：gating→routing→permute→compute→unpermute→finalize 数据流、arch35 分布特征、多版本迭代、permute 变体、量化支持。
- [MC2 通用范式](mc2通用范式.md) — 36+ 通信计算融合算子：3rd 共享计算库、arch22/arch31/arch35 拆分、Tiling 命名约定、Eager vs Graph 双模式、CCU 适配、V 版本演进、量化算子。
- [FFN 通用范式](ffn通用范式.md) — 6 个 FFN 算子：MoE/普通 FFN、Swin 系列、MatMul+Activation+Quant 高度融合、无 arch 子目录平铺特征。
- [MHC 通用范式](mhc通用范式.md) — 9 个 MHC/Sinkhorn 算子：Sinkhorn-Knopp 双随机矩阵投影、几乎全 A5、唯一双架构算子 mhc_post_backward。
- [PosEmbedding 通用范式](posembedding通用范式.md) — 9 个位置编码算子：RoPE 三模式(HALF/INTERLEAVE/QUARTER)+DeepSeek/Partial、broadcast 后缀编码、A2A3 Membase vs A5 Regbase 20000+ TilingKey、融合算子(kv_rms_norm_rope_cache 等)、迁移 Checklist。
