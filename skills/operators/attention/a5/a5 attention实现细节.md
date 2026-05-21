# A5 Attention 实现细节

## 架构特征

- **模块化**：kernel/service/matmul_modules/vector_api/vf 子目录
- **Regbase 入口**：`*_entry_regbase.h` 替代传统模板核函数
- **MX FullQuant 支持**：`*_mx_fullquant.h` 文件名
- **_apt.cpp 后缀**：Ascend Parallel Template

## 5 个原生双架构算子（arch35 存在）

| 算子 | arch35 文件数 | 关键特征 |
|------|:---:|------|
| `flash_attention_score` | 5 | `_entry_regbase.h`, `_kernel_train.h` |
| `flash_attention_score_grad` | — | 训练反向 |
| `incre_flash_attention` | ~10+ | 9种量化模式，PA支持 |
| `block_sparse_attention` | — | `_kernel_arch35_regular.h` |
| `common` | 50+ | 共享全部量化/非量化变体 |

## A5 专属算子（仅有 arch35，无 arch22）

| 算子 | 说明 |
|------|------|
| `attention_update` | A5 才有的更新算子 |
| `dense_lightning_indexer_softmax_lse` | A5 新增稠密索引 |
| `flash_attn` | A5 专用 FlashAttention |
| `fused_causal_conv1d` | A5 才支持 |
| `gather_pa_kv_cache` | A5 PagedAttention KV 收集 |

## Regbase 入口模式

```cpp
// arch35/flash_attention_score_entry_regbase.h
// 使用 RegTensor + MicroAPI 替代传统 TPipe/TQue
// 寄存器直接操作，无需显式 buffer 管理
```

## 量化增强

A5 相比 A2A3 新增：
- **MX FullQuant**（Microscaling 全量化）
- **FP8/HiFLOAT8 数据通路**
- 更多 AntiQuant 变体组合

## 来源
- Agent 分析 `attention/*/op_kernel/arch35/` 全部目录
