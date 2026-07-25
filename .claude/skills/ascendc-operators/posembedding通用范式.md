# PosEmbedding（位置编码）类算子范式

真实源：`ops-transformer_AI/posembedding/`。共 **12 个算子**（旧 skill 误记 9）。RoPE 模式/TilingKey 已对照代码验证。

## 算子清单（真实，12 个）

| 算子 | 说明 |
|---|---|
| `apply_rotary_pos_emb` | QK 融合 RoPE（主算子） |
| `rotary_position_embedding`(+grad) | 单路 RoPE（旧版） |
| `interleave_rope` | 仅 Interleave RoPE |
| `rope_with_sin_cos_cache` | 带 position_id + cos/sin cache |
| `rope_quant_kvcache` | RoPE + 量化 KV Cache |
| `dequant_rope_quant_kvcache` | Dequant + RoPE + Quant |
| `qkv_rms_norm_rope_cache` | QKV Split + RMSNorm + RoPE + Cache |
| `qkv_rms_norm_rope_cache_with_k_scale` | **新增**，带 K scale |
| `kv_rms_norm_rope_cache` | KV Split + RMSNorm + RoPE + Cache |
| `norm_rope_concat`(+grad) | Norm + RoPE + Concat |
| `inplace_partial_rotary_mul`(+grad) | **新增**，原地 partial rotary |

## arch 分布（已验证全量）

9 个算子有 **arch35 子目录**（apply_rotary_pos_emb / rotary_position_embedding(+grad) / rope_with_sin_cos_cache / kv_rms_norm_rope_cache / inplace_partial_rotary_mul(+grad) / qkv_rms_norm_rope_cache_with_k_scale）。其余平铺。

> PosEmbedding 抽样为 arch35（A5 Regbase）；A2A3（Membase）实现可能平铺在 op_kernel/ 根，逐算子确认。

## 三种 RoPE 模式（已验证 grep：HALF/INTERLEAVE/QUARTER/dSplitCoef）

| 模式 | dSplitCoef | 说明 |
|---|---|---|
| **HALF** (Mode 1) | 2 | D 分两半旋转 |
| **INTERLEAVE** (Mode 2) | 1 | 偶奇维度对旋转 |
| **QUARTER** (Mode 3) | 4 | D 分四等份 |
| **DEEPSEEK_INTERLEAVE** | — | Half-size interleave，BF16 dequant + DeInterleave（已验证 grep DEEPSEEK） |
| **Partial RoPE** | — | realDim < headDim 时仅前 realDim 旋转，其余 passthrough（已验证 isPartialRope） |

## Broadcast 模式与 TilingKey 后缀

Cos/Sin broadcast pattern 编码为 tiling 后缀（`_ab`/`_aba`/`_ba`/`_bab`/`_a`/`_b`），对应不同 layout（BNSD/BSND/SBND/TND）和 broadcast 语义。

## A5 TilingKey 20000+ 系列

A5 Regbase 用 20000+ 范围（20010 ABA / 20011 BA / 20020 BAB / 20030 AB / 20040-41 A/B），各 variant 独立 tiling 文件，`REGISTER_OPS_TILING_TEMPLATE` 注册带 priority。

## 融合算子

- `kv_rms_norm_rope_cache`（A5 Regbase）：FullLoad(10000) / Recompute(20000) 两 variant + 5 种 Cache Mode（NORM_CACHE/PA_CACHE/PA_NZ_CACHE/PA_BLK_BNSD_CACHE/PA_BLK_NZ_CACHE）
- `qkv_rms_norm_rope_cache`（A2A3 Membase）：Split → Q(RoPE) / K(RoPE→Cache) / V(RMSNorm→Cache)
- `norm_rope_concat`：RopeOperation → NormOperation → Concat
- `dequant_rope_quant_kvcache`：INT32 → Dequant → RoPE → Quant int8 cache

## A5 Regbase 实现要点

LoadDist/StoreDist 模式（`DIST_UNPACK_B16` BF16 解包 / `DINTLV_B32` De-Interleave / `DIST_PACK4_B32` int8 打包）；寄存器级 RoPE 计算，无 UB 中间缓冲。

## 迁移 Checklist

1. 加 `_apt` 后缀（`*_apt.cpp`）
2. Config 注册 `ascend950` / `mc62cm12a`
3. TilingKey 切到 20000+
4. UB Buffer → 向量寄存器
5. `__global__ __aicore__` → `__simd_vf__`
6. DataCopy → LoadDist/StoreDist

## 来源
- `ops-transformer_AI/posembedding/`（ls + find arch* + grep HALF/INTERLEAVE/QUARTER/DEEPSEEK/dSplitCoef 验证）