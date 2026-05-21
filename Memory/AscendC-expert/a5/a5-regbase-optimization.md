# A5 Regbase 优化

## Regbase 核心概念

Regbase 用**向量寄存器**代替 UB Buffer 进行中间数据传递，消除 UB 读写开销。

## Regbase 在代码仓中的应用

### FlashAttention Regbase Entry

```cpp
#define REGBASE_COPY_TILING_DATA_ASCEND950_ANTIQUANT_BASEAPI(tiling)
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling);
```

TilingKey 中新增 `Regbase` 标志位（1 bit）。

### IFA（IncreFlashAttention）Regbase 转换

A5 IFA 全面 Regbase 化：
- Entry: `incre_flash_attention_entry_regbase.h`
- 核心模板参数新增：`quantMode`, `KvLayoutType`, `isFd`, `enableKVPrefix`, `enableS1OutSplit`
- 模块化处理器：`mm1_processor.h`, `mm2_processor.h`, `vec1_processor.h`, `vec2_processor.h`
- TilingData 使用 `IncreFlashAttentionTilingDataRegbase`

### RoPE Regbase VF 计算

```cpp
// 从 GM 加载 cos/sin 到向量寄存器
LoadDist<T, LoadDist::DINTLV_B32> cosReg, sinReg;

// 寄存器级 RoPE 计算
// (cos*in - sin*swapped, cos*swapped + sin*in)
Mul(rotatedReg, cosReg, inReg);
Mul(swappedReg, sinReg, swappedInReg);
Sub(outReg, rotatedReg, swappedReg);

// 直接写回 GM
StoreAlign<T, StoreDist::DIST_PACK4_B32> outGm, outReg;
```

整个计算链在向量寄存器中完成，无 UB 中间缓冲。

### Norm + RoPE + Cache 融合中的 Regbase

`kv_rms_norm_rope_cache_regbase_full_load.h`：
- RMSNorm: `RmsNormBasicComputeVF` 做 binary-tree reduction
- Rope: `LoadDist::DINTLV_B32` de-interleave → 寄存器级旋转
- Quant: `StoreDist::DIST_PACK4_B32` 量化写入

## Regbase vs Membase 对比

| 方面 | Membase | Regbase |
|------|---------|---------|
| 中间存储 | UB Buffer（需要 AllocTensor/FreeTensor） | 向量寄存器（无需分配） |
| 数据搬移 | DataCopy → UB → 计算 → DataCopy 回 GM | LoadDist → 寄存器计算 → StoreDist |
| Kernel 限定 | `__global__ __aicore__` | `__simd_vf__` |
| 同步 | TQue 自动管理 | 手动 SetFlag/WaitFlag |
| UB 压力 | 需要为中间结果预留 UB | 寄存器直接传递，UB 压力小 |
| 代码量 | 较简单 | 较复杂，需精确管理寄存器 |

## VF 函数命名约定

Regbase 计算函数命名：
- `*AlignVF`：对齐访问模式
- `*BasicComputeVF`：基础计算（通常在 VL_FP32 下）
- `*PostWithMul`：后处理带 affine multiply
- `*SymQuantVF` / `*AsymQuantVF`：对称/非对称量化

## LoadDist / StoreDist 模式

```cpp
LoadDist::DIST_UNPACK_B16    // BF16 解包加载
LoadDist::DINTLV_B32         // De-Interleave 加载
StoreDist::DIST_PACK4_B32    // 4元素打包存储（int8 量化）
```

## TilingKey 中的 Regbase 标志

在 A5 FlashAttention TilingKey 中：
```
bit 61: Regbase (1 bit) - 是否使用 Regbase 内核
```

在 A5 FlashAttentionScoreGrad TilingKey 中新增：
- `IsRegbase` flag
- `Fp8OpenTscm` flag（FP8 + TSCM 双缓冲）
