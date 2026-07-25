---
name: hardware-specs
description: 硬件规格指向 ini + 对 Tiling/Kernel 设计影响归纳（原始数字真值在 AscendC_platform/*.ini 与 ascendc-hardware skill）
metadata:
  type: reference
---

# 硬件规格 — 指向 ini + 设计影响

> **原始数字真值 = `AscendC_platform/*.ini`**。本 memory 不维护规格表（旧表曾漂移：A5 核数 36/32/28 不一致）。规格索引与字段解释见 skill `ascendc-hardware`。

## 查规格的正确路径
1. 确认目标 SoC / variant（如 950DT_9575 vs 950PR）
2. 读对应 `AscendC_platform/Ascend*.ini` 的 `[AICoreSpec]`
3. 字段含义与设计影响见 `ascendc-hardware` skill

## 对 Tiling/Kernel 设计的影响归纳（跨代稳定，非漂移数字）

- **L0C**：决定 tile 累加块上限（A2A3 128KB / A5 256KB）
- **UB**：决定双/三缓冲 tile 门限（A2A3 192KB / A5 ~248KB）
- **L1**：A2A3 无 / A5 512KB，A5 可做更多片上复用
- **Cube:Vector = 1:2**：每 AI Core 1 Cube + 2 Vector
- **同地址并行**：A2A3 不支持（分核需错位规避）/ A5 支持（可简化）
- **Fixpipe/Sparsity**：A2A3 支持 / A5 通路不同（见 ini fb0_size + intrinsic）
- **核数**：因 variant 而异，**一律以 ini 为准**（勿写死）

## 相关
- skill `ascendc-hardware`（SKILL.md + a2a3/a5 索引）
- kernel-optimization `common/general-techniques.md`（同地址冲突、双缓冲等）