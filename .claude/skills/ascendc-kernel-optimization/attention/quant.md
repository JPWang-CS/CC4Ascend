# Attention 量化优化

内容均经真实代码 trace。源：`attention/common/op_kernel/arch35/`（A5 共享量化模块）。

## A5 量化三档（已验证文件名）

A5 attention 量化在 `common/op_kernel/arch35/` 按 **noquant / fullquant_gqa / fullquant_mx** 三档提供共享模块，分 block 级与 kernel 级：

| 量化档 | block_cube | block_vec | kernel | tiling_data |
|---|---|---|---|---|
| **noquant** | fia_block_cube_noquant_gqa | fia_block_vec_noquant_gqa | fia_kernel_noquant_gqa | fia_tiling_data_noquant_gqa |
| **fullquant_gqa** | fia_block_cube_fullquant_gqa | fia_block_vec_fullquant_gqa | fia_kernel_fullquant_gqa | fia_tiling_data_fullquant |
| **fullquant_mx** | fia_block_cube_fullquant_mx | fia_block_vec_fullquant_mx | fia_kernel_fullquant_mx | — |

另有 `fia_block_vec_flashdecode` / `fia_block_vec_flashdecode_fullquant`（flashdecode 专用 vec 模块）。

## 设计理由

- **noquant**：FP16/BF16 标准路径，base 形态
- **fullquant_gqa**：GQA 场景全量化（FP8/INT8 input + scale apply），Cube 侧量化搬入、Vec 侧量化后处理
- **fullquant_mx**：MX Microscaling 全量化（MXFP4/MXFP8），cube 与 vec 都需处理 per-group scale

## block vs kernel 级

- **block 级**（fia_block_*）：分核后单 block 的 Cube/Vec 计算
- **kernel 级**（fia_kernel_*）：整核的算子级入口（含 noquant/fullquant 分发）

## 量化对流水/UB 的影响

- fullquant 需额外 scale buffer（per-token/per-channel scale 进 UB）
- MX 量化 scale 为 per-group（K/32），UB 预算更紧
- 量化搬入与 MatMul 可在 Cube 侧融合（减一次 GM 往返），但需调整 MTE2 同步

## A2A3 vs A5 量化

- A2A3 attention：量化路径在 arch22 平铺文件，变体较少
- A5：模块化三档 + MX 支持，是 A5 相对 A2A3 的主要量化增强

## 相关文件
- [base 模板](base-template.md)
- [流水掩盖](pipeline.md)