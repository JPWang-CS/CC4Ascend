# MoE（专家混合）类算子通用范式

## 算子全景（24 个）

覆盖 MoE 完整生命周期：Gating → Routing → Permute → Compute → Unpermute → Finalize

## MoE 数据流

```
Input Tokens
  → moe_gating_top_k[_softmax]     # 选 TopK 专家
  → moe_init_routing               # 路由分发
  → moe_token_permute              # Token 重排
  → [GroupedMatMul 专家计算]        # 在 gmm/ 或 mc2/ 中
  → moe_token_unpermute            # 恢复排列
  → moe_finalize_routing           # 合并输出
  → Output
```

## 架构特征

- **14 个算子 arch35 only** — MoE 大量新算子仅 A5
- **0 个算子 arch22 only** — A2A3 的 MoE 都在 op_kernel 平铺
- **无算子同时有 arch22+arch35**

## 多版本迭代规律

| 基础算子 | V2 改进 | V3 改进 |
|------|------|------|
| `moe_init_routing` | `_v2` (新 API) | `_v3` (支持不量化+动态量化) |
| `moe_finalize_routing` | `_v2` + `_v2_grad` | — |
| `moe_gating_top_k_softmax` | `_v2` (renorm模式) | — |

## Permute 变体

全部 12 个 permute/unpermute 算子（含 grad）：

| 基础 | 扩展名 | 说明 |
|------|------|------|
| `moe_token_permute` | — | 基础 permute |
| | `_grad` | 反向 |
| | `_with_ep` | 带专家并行范围切片 |
| | `_with_ep_grad` | 反向+EP |
| | `_with_routing_map` | 带路由表映射 |
| | `_with_routing_map_grad` | 反向+路由表 |

unpermute 同理。

## 量化支持

- `moe_init_routing_quant` / `moe_init_routing_quant_v2`
- 仅对 routing 结果量化，不对 token 数据量化

## 来源
- Agent 分析 `moe/` 全部 24 个算子目录
