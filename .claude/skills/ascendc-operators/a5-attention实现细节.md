# A5 Attention 实现细节索引

真实源：`ops-transformer_AI/attention/*/op_kernel/arch35/` + `attention/common/op_kernel/arch35/`。

## A5 (arch35) 结构特征（已验证）

- **模块化子目录**：`kernel/` / `service/` / `matmul_modules/` / `vector_api/` / `vf/`
- **Regbase 入口**：`*_entry_regbase.h` 文件名模式
- **MX FullQuant**：`*_mx_fullquant.h` 文件名模式
- **`_apt.cpp` 后缀**：Ascend Parallel Template（A5 专属）
- **共享设施**：`attention/common/op_kernel/arch35/`（50+ 文件，全量化/非量化变体）

## 原生双架构算子（arch22 + arch35 共存）

`flash_attention_score` / `flash_attention_score_grad` / `fused_infer_attention_score` / `incre_flash_attention` / `lightning_indexer` / `mla_prolog` 等（详见 [attention索引](attention通用范式.md) arch 分布表）

## A5 专属算子（有 arch35，抽样未见 arch22）

`attention_update` / `flash_attn` / `fused_causal_conv1d` / `nsa_*` 系列 / `rain_fusion_attention` / `recurrent_gated_delta_rule` / `ring_attention_update`

> 注意：无 arch22 子目录不等于不支持 A2A3，可能平铺在 op_kernel/。逐算子确认。

## 量化增强（A5 vs A2A3）

- MX FullQuant（Microscaling 全量化）
- FP8 / HiFLOAT8 数据通路
- 更多 AntiQuant 变体

## 来源
- `ops-transformer_AI/attention/*/op_kernel/arch35/`（find + ls 验证）