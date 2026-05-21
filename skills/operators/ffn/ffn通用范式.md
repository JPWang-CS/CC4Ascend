# FFN（前馈网络）类算子通用范式

## 算子列表（6 个）

| 算子 | 功能 | 模式 |
|------|------|------|
| `ffn` | MoE FFN + 普通 FFN | aclnn + Graph |
| `swin_attention_ffn` | Swin Attention + FFN | Graph only |
| `swin_transformer_ln_qkv` | Swin LayerNorm→QKV 投影 | Graph only |
| `swin_transformer_ln_qkv_quant` | Swin LN→QKV + 量化 | Graph only |
| `ffn_worker_scheduler` | Attn/FFN 分离→FFN 数据扫描 | aclnn + Graph (AI CPU) |
| `ffn_worker_batching` | Attn/FFN 分离→Token 重排 | aclnn + Graph |

## 特征

- **全部无 arch 子目录** — 6 个算子均直接在 op_kernel/ 下单文件实现
- **融合深度高**：FFN 内部融合了 MatMul + Activation + Quant
- **Swin 系列**是特殊架构专用

## 来源
- Agent 分析 `ffn/` 全部 6 个算子目录
