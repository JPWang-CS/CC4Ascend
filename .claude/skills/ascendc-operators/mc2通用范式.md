# MC2（通信计算融合）类算子范式

真实源：`ops-transformer_AI/mc2/`。共 **35 个算子**（最大算子类，不含 3rd/common/tools）。核心 = 通信（AllReduce/AllToAll/ReduceScatter）+ MatMul/GroupedMatMul 融合。arch 分布已全量验证。

## 算子清单（真实，35 个）

- **MatMul+通信**：`matmul_all_reduce` / `matmul_all_reduce_add_rms_norm` / `matmul_allto_all` / `matmul_reduce_scatter`(+v2) / `inplace_matmul_all_reduce_add_rms_norm`
- **AllGather/AllToAll+MatMul**：`all_gather_matmul`(+v2) / `allto_all_matmul` / `allto_all_all_gather_batch_mat_mul` / `allto_allv_grouped_mat_mul` / `allto_allv_quant_grouped_mat_mul` / `batch_mat_mul_reduce_scatter_allto_all`
- **GroupedMatMul+通信**：`grouped_mat_mul_all_reduce` / `grouped_mat_mul_allto_allv`
- **量化**：`quant_all_reduce` / `quant_reduce_scatter` / `quant_grouped_mat_mul_allto_allv`
- **MoE 分布**：`moe_distribute_dispatch`(+setup+teardown+v2+v3) / `moe_distribute_combine`(+setup+teardown+add_rms_norm+v2+v3) / `mega_moe` / `moe_update_expert`
- **Attn/FFN 桥接**：`attention_to_ffn` / `ffn_to_attention`
- **Barrier**：`distribute_barrier` / `distribute_barrier_extend`
- **新增**：`engram_fetch` / `engram_fetch_wait`（旧 skill 无）

## arch 分布（已验证全量，四架构）

| 架构 | 含义 | 代表算子 |
|---|---|---|
| **arch22** | A2A3 | `matmul_all_reduce` / `allto_all_matmul` / `attention_to_ffn` / `mega_moe` / `moe_distribute_*` 全系列 |
| **arch31** | 310P | `matmul_all_reduce` / `all_gather_matmul_v2`(tiling) / `matmul_reduce_scatter_v2`(tiling) |
| **arch35** | A5 | 大多数算子 |
| **arch38** | 新架构 | `all_gather_matmul_v2`(tiling) / `matmul_reduce_scatter_v2`(tiling) |

> MC2 是唯一四架构共存（arch22/31/35/38）的算子类。tiling 层也按 arch 拆（`op_host/op_tiling/arch*/`）。

## 3rd 共享计算库

`mc2/3rd/`：`mat_mul_v3` / `batch_mat_mul_v3` / `quant_batch_matmul_v3` / `weight_quant_batch_matmul_v2` / `grouped_matmul` / `rms_norm` / `norm_common` / `template_linear_algebra`。多个 MC2 算子共享这些核心计算库。

## Eager vs Graph 双模式 + CCU

A5 中 CCU 通信路径两模式适配（详见 ascendc-development §CCU）：
- **Eager**：`NnopbaseSetHcclServerType(executor, CCU)`
- **Graph**：`CreateCcuTask(context, ccuGroups)` + `ccu server`/`ccu_stream`

## Tiling 命名约定

`op_tiling/${op}_tiling_${soc}.cpp`（如 `_910` `_910b` `_950`）+ `${op}_tiling_key.h`。TilingKey 常见：`HIGH_PERFORMANCE_KEY` / `QUANT_KEY` / `HIGH_PRECISION_KEY`。

## V 版本演进

- MoE Dispatch/Combine：V1 → `_v2` → `_v3`（+setup/teardown 变体）
- MatMul ReduceScatter / AllGather MatMul：V1 → `_v2`（双架构）

## 来源
- `ops-transformer_AI/mc2/`（ls + find arch* 全量验证）