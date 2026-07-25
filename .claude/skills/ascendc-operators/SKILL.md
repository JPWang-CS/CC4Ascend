---
name: ascendc-operators
description: ops-transformer 七大类算子的结构索引与范式速查。按算子类组织：Attention / GMM / MoE / MC2 / FFN / MHC / PosEmbedding。覆盖每类的真实算子清单、arch22/arch35/arch38 代码分布、关键代码路径、分核/流水/TilingKey 范式索引。当要查某类算子有哪些实现、某算子代码在哪、某类算子的分核或 Tiling 范式时调用。（本 skill 是各算子的结构/范式索引；通用迁移方法论与 A5 编程范式见 ascendc-development；具体优化偏离见 ascendc-kernel-optimization）
---

# ops-transformer 算子结构索引

本 skill 是七大类算子的**结构索引 + 范式速查**。真实源 = `ops-transformer_AI/{attention,gmm,moe,mc2,ffn,mhc,posembedding}/`。

## 算子数量（2026-07 核对）

| 类 | 算子数 | 旧 skill 旧数（漂移） |
|---|---|---|
| attention | 61 | 53 |
| mc2 | 35 | 36+ |
| moe | 28 | 24 |
| posembedding | 12 | 9 |
| gmm | 7 | — |
| mhc | 8 + common | 9 |
| ffn | 6 | 6 |

## arch 分布关键事实（已验证）

- **arch22** = A2A3，**arch35** = A5，**arch31** = 310P，**arch38** = 新架构（IFA 独有）
- **MC2 有 arch22 + arch31 + arch35 三架构**（tiling 层也按 arch 拆）
- **attention 部分算子双架构**（arch22+arch35），IFA 有 **arch38** + `attn_infra/arch` 共享设施
- **GMM 特殊**：不用 arch22 子目录，A2A3 代码平铺 + `a16w4_msd/` + `gmm_infra/arch/` 抽象层；A5 在 `arch35/`
- **MoE / PosEmbedding 抽样 arch35 only**（A2A3 可能平铺，逐算子确认）
- ⚠️ **"无 arch22 子目录" ≠ "不支持 A2A3"**：A2A3 代码常平铺在 op_kernel/ 或用 infra 抽象

## 明细文件

- [Attention](attention通用范式.md) — 61 算子清单、arch 分布、IFA arch38、attn_infra 结构
- [A5 Attention 实现](a5-attention实现细节.md) — arch35 模块化结构
- [GMM](gmm通用范式.md) — 7 算子、全部 arch35 only
- [MoE](moe通用范式.md) — 28 算子清单、arch35 分布
- [MC2](mc2通用范式.md) — 35 算子、三架构 arch22/31/35、Eager/Graph
- [FFN](ffn通用范式.md) — 6 算子
- [MHC](mhc通用范式.md) — Sinkhorn 系列、几乎全 A5
- [PosEmbedding](posembedding通用范式.md) — 12 算子、RoPE 变体、融合算子

## 边界

- 通用迁移方法论 / A5 四范式 → `ascendc-development`
- 某算子的 kernel 优化 base 模板与偏离 → `ascendc-kernel-optimization`
- 具体 API → `ascendc-api`
- 硬件规格 → `ascendc-hardware` → `AscendC_platform/*.ini`