# PosEmbedding（位置编码）类算子通用范式

## 算子列表（9 个全面分析）

| 算子 | 功能 | A5 | A2A3 |
|------|------|:---:|:---:|
| `apply_rotary_pos_emb` | QK 融合 RoPE（主算子） | Regbase | Membase |
| `rotary_position_embedding` | 单路 RoPE（旧版） | Regbase | Membase |
| `rotary_position_embedding_grad` | RoPE 反向 | Regbase (6 variants) | Membase |
| `interleave_rope` | 仅 Interleave RoPE | — | Membase |
| `rope_with_sin_cos_cache` | 带 position_id + cos/sin cache | Regbase | Membase |
| `rope_quant_kvcache` | RoPE + 量化 KV Cache | — | Membase |
| `dequant_rope_quant_kvcache` | Dequant + RoPE + Quant | — | Membase |
| `qkv_rms_norm_rope_cache` | QKV Split + RMSNorm + RoPE + Cache | — | Membase |
| `kv_rms_norm_rope_cache` | KV Split + RMSNorm + RoPE + Cache | Regbase | Membase |
| `norm_rope_concat` | Norm + RoPE + Concat + Grad | — | Membase |

## 三种 RoPE 模式

### HALF (Mode 1)
D 分成两半。`dSplitCoef = 2`：
```
qOut[0..halfD-1] = q[0..halfD-1]*cos[0..halfD-1] - q[halfD..D-1]*sin[0..halfD-1]
qOut[halfD..D-1]  = q[halfD..D-1]*cos[halfD..D-1] + q[0..halfD-1]*sin[halfD..D-1]
```

### INTERLEAVE (Mode 2)
偶奇维度对旋转。`dSplitCoef = 1`：
```
out[evenIdx] = cos[even]*in[even] - sin[even]*in[odd]
out[oddIdx]  = cos[odd]*in[odd] + sin[odd]*in[even]
```

### QUARTER (Mode 3)
D 分成 4 等份。`dSplitCoef = 4`：
```
qOut[0]=q[0]*cos[0] - q[1]*sin[0], qOut[1]=q[1]*cos[1] + q[0]*sin[1]
qOut[2]=q[2]*cos[2] - q[3]*sin[2], qOut[3]=q[3]*cos[3] + q[2]*sin[2]
```

### DEEPSEEK_INTERLEAVE
Half-size interleave: LoadDist::DIST_UNPACK_B16 加载时 BF16 dequant → DeInterleave → cos*in - sin*neg_swapped + cos*swapped + sin*in

### Partial RoPE (isPartialRope)
当 cos/sin 维度 (realDim) < headDim (D) 时，仅前 realDim 个元素旋转，其余 passthrough。

## Broadcast 模式与 TilingKey 后缀

Cos/Sin broadcast pattern 编码为 tiling 后缀：

| 后缀 | Layout | Cos/Sin Broadcast | A5 TILING_KEY | 含义 |
|------|--------|-------------------|---------------|------|
| `_ab` | SBND, BSND, TND | B-matched or B-broadcast | 20030 | 标准 B 对齐 |
| `_aba` | BNSD | cos_b == b_（无 B broadcast） | 20010 | Q/K B 轴对齐 |
| `_ba` | BNSD | cos_b == 1（B broadcast） | 20011 | B 广播到所有 |
| `_bab` | BSND | cos_b == 1（1s1d broadcast） | 20020 | S/D 双重广播 |
| `_a` | — | NO_BROADCAST | — | 全 B*S*N 广播 |
| `_b` | — | BROADCAST_BSN | — | B 维度广播 |

## TilingKey 体系

### A2A3 (Membase)
```
TILING_KEY 1: SMALL (单 batch per core)
TILING_KEY 3: AB (多 batch 拆分)
TILING_KEY 4: ABCast (Cos/Sin B 广播)
```

### A5 (Regbase) — 20000+ 系列
```
TILING_KEY 20010: ABA (BNSD, cos_b==b_)
TILING_KEY 20011: BA  (BNSD, cos_b==1)
TILING_KEY 20020: BAB (BSND, cos_b==1, 1s1d)
TILING_KEY 20030: AB  (SBND/BSND/TND)
TILING_KEY 20040/20041: A/B (NO_BROADCAST / BROADCAST_BSN)
```

每个 variant 独立的 tiling 文件 / tiling 类，通过 `REGISTER_OPS_TILING_TEMPLATE` 注册，带 priority 值。

## 融合算子详解

### kv_rms_norm_rope_cache（A5 Regbase, 1471 行 base class）

**FullLoad variant (TILING_KEY 10000)**：
```
CopyIn(K) → Rope(K) → ScatterUpdateK → RmsNorm(V) → ScatterUpdateV
```

**Recompute variant (TILING_KEY 20000)**：
```
RMSNorm re-read x² from GM（不保留在 UB）
Binary-tree reduction → PostWithMul → Quant
```

5 种 Cache Mode: NORM_CACHE(0) / PA_CACHE(1) / PA_NZ_CACHE(2) / PA_BLK_BNSD_CACHE(3) / PA_BLK_NZ_CACHE(4)

模板参数：`<T_KV, T_K_CACHE, T_V_CACHE>`（KV 输入 dtype, K-cache quant type, V-cache quant type）

VF 计算函数命名约定：
- `RmsNormBasicComputeVF` / `RmsNormPostWithMul` / `RmsNormSymQuantWithKvVF` / `RmsNormAsymQuantWithKvVF`
- `RopeSymQuantVF` / `RopeAsymQuantVF`：同时 RoPE + 量化 KV cache

### qkv_rms_norm_rope_cache（A2A3 Membase）
```
Input QKV → Split → Q(RoPE→output) / K(RoPE→Cache) / V(RMSNorm→Cache)
```

### norm_rope_concat
基础类层次：
- `RopeOperation<RopeType>`：PreProcess(cos/sin load) → Rotate(Gather/Scatter) → Collect(Mul+MulAddDst)
- `NormOperation`：LayerNorm 或 RMSNorm（Sum → mean → x-E(x) → var → sqrt → div → affine）
- `NormRopeConcat`：ProcessOne() 迭代 (S,B,N) 做 normOp → ropeOp → concat(BEFORE_AFTER encoder)

### dequant_rope_quant_kvcache (981 lines)
```
INT32 input → Dequant(weight_scale+bias+activation_scale) → fp32 → fp16 →
RoPE(HALF, Q+K) → Div(scale) → Add(offset) → Cast int8 cache
```

## A5 Regbase 实现要点

### LoadDist / StoreDist 模式

```cpp
LoadDist::DIST_UNPACK_B16     // BF16 解包加载
LoadDist::DINTLV_B32          // De-Interleave 加载（INTERLEAVE RoPE）
StoreDist::DIST_PACK4_B32     // 4 元素打包存储（int8 量化输出）
```

### RoPE VF 计算（寄存器级）

```cpp
// 从 GM 加载 cos/sin
LoadDist<T, LoadDist::DINTLV_B32> cosReg, sinReg;
// 寄存器内计算（无 UB 中间缓冲）
Mul(rotatedReg, cosReg, inReg);
Mul(swappedReg, sinReg, swappedInReg);
Sub(outReg, rotatedReg, swappedReg);
// 直接写回 GM
StoreAlign<T, StoreDist::DIST_PACK4_B32> outGm, outReg;
```

## 迁移 Checklist

1. Kernel 文件名加 `_apt` 后缀（`apply_rotary_pos_emb_apt.cpp`）
2. Config 注册加 `ascend950` / `mc62cm12a`
3. TilingKey 切换到 20000+ 范围
4. 中间数据 UB Buffer → 向量寄存器
5. Kernel 限定 `__global__ __aicore__` → `__simd_vf__`
6. DataCopy → LoadDist / StoreDist（简单搬移场景）

## 来源
- Agent 深度分析 `posembedding/` 全部 9 个算子目录、67+ 个文件
