# MoE（专家混合）类算子范式

真实源：`ops-transformer_AI/moe/`。共 **28 个算子**（旧 skill 误记 24）。RoPE 等范式已对照代码验证。

## MoE 数据流

```
tokens → moe_gating_top_k[_softmax] → moe_init_routing → moe_token_permute
  → [GroupedMatMul 专家计算（gmm/ 或 mc2/）]
  → moe_token_unpermute → moe_finalize_routing → output
```

## 算子清单（真实，28 个）

- **Gating/Routing**：`moe_gating_top_k`(+backward) / `moe_gating_top_k_softmax`(+v2) / `moe_fused_topk` / `moe_re_routing`
- **Init Routing**：`moe_init_routing`(+v2+v2_grad+v3) / `moe_init_routing_quant`(+v2)
- **Finalize**：`moe_finalize_routing`(+v2+v2_grad)
- **Permute**：`moe_token_permute`(+grad+with_ep+with_ep_grad+with_routing_map+with_routing_map_grad)
- **Unpermute**：`moe_token_unpermute`(+with_routing_map_grad)
- **其他**：`moe_compute_expert_tokens`

## arch 分布（已验证全量）

**13 个算子有 arch35 子目录**（gating/init_routing/finalize/re_routing/permute_with_routing_map 等），其余**平铺无 arch 子目录**。

> 无 arch22 子目录 ≠ 不支持 A2A3，平铺算子的 A2A3 实现可能在 op_kernel/ 根。MoE 新算子倾向 arch35。

## 多版本迭代

- `moe_init_routing` → `_v2`(新 API) → `_v3`(不量化+动态量化)
- `moe_finalize_routing` → `_v2`(+grad)
- `moe_gating_top_k_softmax` → `_v2`(renorm 模式)

## 量化支持

`moe_init_routing_quant`(+v2)：仅对 routing 结果量化，不对 token 数据量化。

## Permute 变体

`moe_token_permute` 扩展：`_grad` / `_with_ep`(专家并行切片) / `_with_routing_map`(路由表映射)，各有 grad。unpermute 同理。

## 来源
- `ops-transformer_AI/moe/`（ls + find arch* 全量验证）