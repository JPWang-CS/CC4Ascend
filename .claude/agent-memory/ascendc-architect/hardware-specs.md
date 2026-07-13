---
name: hardware-specs
description: A2A3 (910B) 和 A5 (950) 芯片硬件规格速查 — AI Core/UB/L0/L1/L2/HBM 容量、Cube/Vector 核数、Fixpipe/Sparsity 支持
metadata:
  type: reference
---

# 芯片硬件规格速查

## A2A3 (Ascend 910B/910C)

| 规格项 | 参数 |
|--------|------|
| AI Core 总数 | 24 |
| HBM 容量 | 64 GB |
| L2 容量 | 192 MB |
| Cube 核数量 | 24 |
| AIC 版本 | AIC-C-220 / dav-c220-cube |
| L1 大小 | — |
| L0A 大小 | 64 KB |
| L0B 大小 | 64 KB |
| L0C 大小 | 128 KB |
| Bias 大小 | 1 KB |
| Vector 核数量 | 48 |
| CCEC_AIV 版本 | dav-c220-vec |
| UB 大小 | 192 KB |
| Fixpipe 支持 | 是 |
| Sparsity 支持 | 是 |

## A5 (Ascend 950)

| 规格项 | 参数 |
|--------|------|
| AI Core 总数 | 36 |
| HBM 容量 | 128 GB |
| L2 容量 | 128 MB |
| Cube 核数量 | 36 |
| L1 大小 | 512 KB |
| L0A 大小 | 64 KB |
| L0B 大小 | 64 KB |
| L0C 大小 | 256 KB |
| Bias 大小 | 4 KB |
| Vector 核数量 | 72 |
| UB 大小 | 248 KB |
| Fixpipe 支持 | 否 |
| Sparsity 支持 | 否 |

## 关键差异速记

- **A5 核数 +50%** (24→36 Cube, 48→72 Vector)
- **A5 L0C 翻倍** (128→256KB) → tile 粒度可放大
- **A5 UB 扩大** (192→248KB, +29%) → 双缓冲阈值上调
- **A5 L2 缩小** (192→128MB, -33%) → 需更注意 L2 复用
- **A5 无 Fixpipe** → 软件流水替代，CrossCoreSetFlag/WaitFlag
- **A5 无 Sparsity** → 走 Dense 计算路径
- **A2A3 需错位分核** 避免同地址冲突，**A5 硬件支持同地址并行**
- **A5 有 L1** (512KB)，A2A3 无 L1

## 对 Tiling 的影响

| 设计决策 | A2A3 | A5 |
|----------|------|-----|
| 单 tile 最大字节 (受限于 L0C) | 128KB | 256KB |
| 双缓冲 tile 门限 (受限于 UB) | tile ≤ 96KB | tile ≤ 124KB |
| Cube:Vector 配比 | 1:2 (24:48) | 1:2 (36:72) |
| 分核策略 | 错位/对角线 | 规则滑动/顺序/对称/TND |
