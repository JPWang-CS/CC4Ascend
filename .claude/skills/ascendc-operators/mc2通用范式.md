# MC2（通信计算融合）类算子通用范式

## 算子全景

36+ 个算子，最大的算子分类，核心是 **通信（AllReduce/AllToAll/ReduceScatter）+ 矩阵乘（MatMul/GroupedMatMul）融合**。

## 代码组织

### 3rd 共享库

```
mc2/3rd/
├── batch_mat_mul_v3/       # 批矩阵乘 V3
├── mat_mul_v3/             # 矩阵乘 V3
├── quant_batch_matmul_v3/  # 量化批矩阵乘
├── weight_quant_batch_matmul_v2/
├── grouped_matmul/
├── rms_norm/               # RMS Norm 基础库
├── norm_common/
└── template_linear_algebra/
```

这些是多个 MC2 算子共享的核心计算库。

### 架构拆分

| 架构 | 数量 | 典型算子 |
|------|:---:|------|
| arch22 only | 7 | `all_gather_matmul`, `attention_to_ffn` |
| arch35 only | 4 | `moe_distribute_dispatch_v2` |
| **both arch22+arch35** | 6 | `matmul_all_reduce`, `all_gather_matmul_v2` |
| 无拆分 | 17+ | `moe_distribute_combine` 系列 |

### arch31 特例

`matmul_all_reduce` 同时拥有 arch22/arch31/arch35 三个架构目录。arch31 对应 Ascend 310P。

## 核心范式

### Tiling 文件命名约定

```
op_tiling/
├── matmul_all_reduce_tiling_910.cpp       # A2
├── matmul_all_reduce_tiling_910b.cpp      # A2
├── matmul_all_reduce_tiling_950.cpp       # A5
├── weight_quant_matmul_all_reduce_tiling_950.cpp
├── quant_matmul_all_reduce_tiling_950.cpp
└── matmul_all_reduce_tiling_key.h         # TilingKey 枚举
```

TilingKey 枚举通用模式：`HIGH_PERFORMANCE_KEY`, `QUANT_KEY`, `HIGH_PRECISION_KEY`。

### Eager vs Graph 双模式

大多数 MC2 算子同时支持两种调用方式。A5 中 CCU 通信路径在两种模式下适配不同：

- **Eager**：`NnopbaseSetHcclServerType(executor, CCU)`
- **Graph**：`CreateCcuTask(context, ccuGroups)` + `ccu server`/`ccu_stream`

### V 版本演进

| 类别 | V1 | V2 | V3 |
|------|-----|-----|-----|
| MoE Dispatch | `moe_distribute_dispatch` | `_v2` (A5专用) | `_v3` |
| MoE Combine | `moe_distribute_combine` | `_v2` (A5专用) | `_v3` |
| MatMul ReduceScatter | `matmul_reduce_scatter` | `_v2` (双架构) | — |
| AllGather MatMul | `all_gather_matmul` | `_v2` (双架构) | — |

## 量化支持

6 个专用量化算子 + 非量化算子的量化 Tiling 变体：

```
quant_all_reduce/
quant_reduce_scatter/
quant_grouped_mat_mul_allto_allv/
allto_allv_quant_grouped_mat_mul/   # arch35 only
```

## 来源
- Agent 分析 `mc2/` 全部 36+ 算子目录
