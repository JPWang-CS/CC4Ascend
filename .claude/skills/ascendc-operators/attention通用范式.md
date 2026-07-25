# Attention 类算子范式

真实源：`ops-transformer_AI/attention/`。共 **61 个算子**（旧 skill 误记 53）。范式内容已对照真实代码验证。

## 算子清单（真实，按系列）

- **FlashAttention**：`flash_attention_score`(+grad) / `prompt_flash_attention` / `flash_attn`(+metadata) / `rain_fusion_attention`
- **InferAttention / KV Cache**：`fused_infer_attention_score` / `incre_flash_attention` / `gather_pa_kv_cache` / `scatter_pa_cache` / `scatter_pa_kv_cache` / `indexer_quant_cache`
- **Sparse / BlockSparse**：`sparse_flash_attention`(+grad) / `block_sparse_attention`(+grad) / `kv_quant_sparse_flash_attention` / `sparse_flash_mla`(+grad+metadata) / `mixed_quant_sparse_flash_mla`(+metadata)
- **MLA**：`mla_preprocess`(+v2) / `mla_prolog`(+v2+v3)
- **Lightning Indexer / NSA**：`lightning_indexer`(+grad+v2+metadata) / `quant_lightning_indexer`(+v2) / `nsa_compress`(+attention+infer+grad+with_cache) / `nsa_selected_attention`(+grad+infer) / dense/sparse kl_loss 变体
- **Conv / Delta**：`fused_causal_conv1d` / `inplace_fused_causal_conv1d` / `masked_causal_conv1d`(+backward) / `chunk_gated_delta_rule` / `recurrent_gated_delta_rule`
- **Worker**：`attention_update` / `attention_worker_combine` / `attention_worker_scheduler` / `ring_attention_update`
- **其他**：`fused_floyd_attention`(+grad) / `kv_compress_epilog` / `swin_attention_score_quant`

## arch 分布（已验证）

- **双架构**（arch22+arch35）：`flash_attention_score` / `fused_infer_attention_score` / `lightning_indexer` / `mla_prolog`
- **arch38**（新架构）：`fused_infer_attention_score` 独有
- **A5 专属**（arch35 only + attn_infra）：`block_sparse_attention` 等
- `attn_infra/arch/`：部分算子的共享基础设施目录

## FlashAttention A2A3 分核（已验证，文件名直证）

arch22 下四种拆分 Kernel 类（文件名即证据）：

| 文件 | Kernel 类 | 拆分母 |
|---|---|---|
| `flash_attention_score_s1s2_bn2gs1.h` | S1s2Bn2gs1 | S1+S2+D |
| `flash_attention_score_s1_bn2gs1.h` | S1Bn2gs1 | 仅 S1 |
| `flash_attention_score_bn2gs1s2_b.h` | Bn2gs1s2B | Batch |
| `flash_attention_score_s1s2_bn2gs1_sab.h` | S1s2Bn2gs1SameAB | S1 + AIC-AIV 配对 |

TilingKey 中 UB0/UB1/Block 控制拆分维度：UB0=S1,UB1=S2→S1+S2；UB0=S1,UB1=D→仅S1；Block=B→Batch。

## FlashAttention A5 分核（已验证 grep）

在 `CalcRealCoreIdx()` 中实现三种模式：
1. **顺序分核**：s1 块依次分发，提高 L2 复用
2. **对称分核**：N 上半顺序 + 下半对称
3. **TND（正倒序循环）**：`varlenCycleCoreNums = coreNum × 2`

## 流水同步

- **A2A3 三缓冲**：taskId 流转 IterateBmm1 → ProcessVec1 → SetFlag(MTE3_MTE2) → IterateBmm2 → ProcessVec2
- **A5 四阶流水**：CrossCoreSetFlag/WaitFlag 跨核同步（详见 ascendc-development 核间同步章节）

## A5 结构特征

- 模块化：`kernel/` `service/` `matmul_modules/` `vector_api/` `vf/`
- Regbase 入口：`*_entry_regbase.h`
- MX FullQuant：`*_mx_fullquant.h`
- `_apt.cpp` 后缀 = A5

## TilingKey 位域 / UB 布局 / Softmax 机制

线级细节以真实代码为准，入口：
- `flash_attention_score/op_kernel/arch22/flash_attention_score_template_tiling_key.h` — A2A3 TilingKey
- `attention/common/op_kernel/arch35/` — A5 共享设施（50+ 文件）

## 来源
- `ops-transformer_AI/attention/`（ls + find + grep 验证文件名/类名/分核模式）