# FFN（前馈网络）类算子范式

真实源：`ops-transformer_AI/ffn/`。共 **6 个算子**（已验证，无 arch 子目录）。

## 算子清单（真实）

| 算子 | 说明 |
|---|---|
| `ffn` | MoE FFN + 普通 FFN（aclnn + Graph） |
| `swin_attention_ffn` | Swin Attention + FFN（Graph only） |
| `swin_transformer_ln_qkv` | Swin LayerNorm→QKV 投影（Graph only） |
| `swin_transformer_ln_qkv_quant` | + 量化（Graph only） |
| `ffn_worker_scheduler` | Attn/FFN 分离→FFN 数据扫描（AI CPU） |
| `ffn_worker_batching` | Attn/FFN 分离→Token 重排 |

## 特征（已验证）

- **全部无 arch 子目录**：6 个算子均直接在 op_kernel/ 下实现，不按 arch22/arch35 拆分
- **融合深度高**：FFN 内部融合 MatMul + Activation + Quant
- **Swin 系列**是特殊架构专用
- **worker 系列**走 AI CPU 路径

## 来源
- `ops-transformer_AI/ffn/`（ls + find arch* 验证）