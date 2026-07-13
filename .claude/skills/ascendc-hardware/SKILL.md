---
name: ascendc-hardware
description: Ascend 芯片硬件规格速查，用于 Tiling 设计与性能估算。涵盖 A2A3（Ascend 910B/910C）与 A5（Ascend 950）的 AI Core 核数、Cube/Vector 单元数量、L0A/L0B/L0C/L1/UB/Bias 片上缓冲容量、HBM/L2 容量、Fixpipe 与 Sparsity 硬件支持、以及每项规格对 Tiling 粒度和分核策略的设计影响。当要确定 tile 尺寸、估算 buffer 占用、判断某硬件特性（Fixpipe/Sparsity/同地址并行）是否可用、或对比两代芯片做迁移决策时调用。
---

# Ascend 芯片硬件规格

本 skill 收录 A2A3（910B/910C）与 A5（950）的硬件规格表与设计要点，供 Tiling 设计、buffer 预算、迁移决策时查阅。

> 说明：同样的芯片规格数据也记录在 agent memory `hardware-specs.md`（含跨芯片对 Tiling 影响的归纳）。本 skill 提供可被 Skill 工具触发的入口；两处内容保持一致，规格更新时同步两边。

## 关键差异速览

| 维度 | A2A3 (910B) | A5 (950) |
|------|-------------|----------|
| AI Core 数量 | 24 | 36 (+50%) |
| L0C 大小 | 128 KB | 256 KB (+100%) |
| UB 大小 | 192 KB | 248 KB (+29%) |
| Bias 大小 | 1 KB | 4 KB |
| Vector 核数 | 48 | 72 |
| Fixpipe | 支持 | 不支持 |
| Sparsity | 支持 | 不支持 |

## 明细文档

- [A2A3 (910B/910C) 硬件规格](a2a3硬件规格.md) — AIC/CCEC 版本、Cube/Vector 单元规格、L0/UB/Bias 容量、Fixpipe/Sparsity 支持、同地址冲突与错位分核、设计要点。
- [A5 (950) 硬件规格](a5硬件规格.md) — AI Core/Cube/Vector 数量与容量、L1 512KB/L0C 256KB/UB 248KB、Fixpipe/Sparsity 不可用的替代策略、L2 缩小/HBM 翻倍的设计影响。
