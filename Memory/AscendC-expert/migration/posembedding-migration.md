# PosEmbedding A2A3 → A5 迁移实例

## 架构迁移：Membase → Regbase

### A2A3 (Membase)
```
Kernel 位置: op_kernel/（无子目录）
数据流: GM → DataCopy → UB → Vector 计算 → UB → DataCopy → GM
中间存储: TQue + TBuf
Kernel 限定: __global__ __aicore__
Config: ascend910b, ascend310p
```

### A5 (Regbase/arch35)
```
Kernel 位置: op_kernel/arch35/
数据流: GM → LoadDist → 向量寄存器 计算 → StoreDist → GM
中间存储: 向量寄存器（无需 UB Buffer 分配）
Kernel 限定: __simd_vf__
Config: ascend950, mc62cm12a
TilingKey: 20000+ 范围
```

## apply_rotary_pos_emb 迁移

### 配置注册变化
```cpp
// A2A3
ascend910b → Membase (apply_rotary_pos_emb)
ascend310p → Membase (fp16/fp32 only)

// A5  
ascend950 → Regbase (apply_rotary_pos_emb_apt)
mc62cm12a → Regbase
```

`_apt` 后缀 = Ascend Parallel Template（A5 kernel 命名约定）。

### TilingKey 变化

| TilingKey | 变体 | 平台 |
|-----------|------|------|
| 1, 3, 4 | SMALL / AB / ABCast | A2A3 |
| 20010 | ABA | A5 |
| 20011 | BA | A5 |
| 20020 | BAB | A5 |
| 20030 | AB | A5 |

### Tiling 实现变化

A2A3: 单个 tiling 文件处理所有变体
```cpp
// apply_rotary_pos_emb_tiling.cpp
TILING_KEY 1/3/4 分支
```

A5: 每个变体独立 tiling 文件
```cpp
// apply_rotary_pos_emb_tiling_ab_arch35.cpp
// apply_rotary_pos_emb_tiling_aba_and_ba_arch35.cpp
// apply_rotary_pos_emb_tiling_bab_arch35.cpp
```

## kv_rms_norm_rope_cache 迁移

### A2A3
```
Kernel: op_kernel/kv_rms_norm_rope_cache_b16_*.h
- Membase UBuffer RMSNorm
- 多个 layout/cache 格式变体：b1sd, pa, pa_blk_bnsd, pa_blk_nz, pa_nz, mtp
```

### A5
```
Kernel: op_kernel/arch35/
- kv_rms_norm_rope_cache_regbase_full_load.h (TILING_KEY 10000)
- kv_rms_norm_rope_cache_regbase_recompute.h (TILING_KEY 20000)
- Regbase RMSNorm: binary-tree reduction + PostWithMul
- Regbase Rope: LoadDist::DINTLV_B32 + 寄存器级旋转
- 5 种 cache mode: NORM_CACHE(0), PA_CACHE(1), PA_NZ_CACHE(2), 
                    PA_BLK_BNSD_CACHE(3), PA_BLK_NZ_CACHE(4)
```

### 关键差异

| 方面 | A2A3 | A5 |
|------|------|-----|
| RMSNorm | UBuffer 中 x^2 → sum → sqrt → div | VF binary-tree ReduceSum |
| RoPE | UB 中 DeInterleave → Mul → Add | 寄存器级 LoadDist::DINTLV_B32 |
| Quant | UB Cast 链 | StoreDist::DIST_PACK4_B32 |
| Cache 更新 | DataCopy scatter | ScatterUpdate with S_MTE3 sync |

## 融合算子流水线变化

### A2A3（qkv_rms_norm_rope_cache）
```
单一 KernelQkvRmsNormRopeCache 类
UB-based 三阶段：Norm → Rope → Cache Update
每个阶段独立 UB buffer 分配
```

### A5（kv_rms_norm_rope_cache regbase）
```
模块化类层次：
  KvRmsNormRopeCacheRegbase (base)
  ├── FullLoad variant (一次性加载)
  └── Recompute variant (re-read x² from GM)

VF 计算链在寄存器中完成：
  RmsNormBasicComputeVF → RmsNormPostWithMul → RopeVF → QuantVF
```

### 精度变化

A5 Regbase 默认在 VL_FP32（全浮点向量长度）下计算，比 A2A3 的 FP16 UB 中间结果精度更高，特别是 RMSNorm 的 x² sum 和 RoPE 的 sin/cos multiply 路径。

## 迁移通用 Checklist

1. Kernel 文件名加 `_apt` 后缀
2. Config 注册加 `ascend950` / `mc62cm12a`
3. TilingKey 切换到 20000+ 范围
4. 中间数据从 UB Buffer 迁移到向量寄存器
5. Kernel 限定从 `__global__ __aicore__` 改为 `__simd_vf__`
6. LoadDist/StoreDist 替代 DataCopy（对简单搬移场景）
