# QuantBatchMatmul — Ascend 950 (arch35) 算子原理深度分析

> 基于 `ops-transformer_AI` v9.1.0 源码，聚焦 Ascend 950 (arch35, `__CCE_AICORE__ == 220`)
> 代码路径: `mc2/3rd/quant_batch_matmul_v3/op_kernel/arch35/`

---

## 1. 算子定位

QuantBatchMatmul 在 950 上的核心公式不变：

```
dequant_weight = weight_q × antiquantScale + antiquantOffset
output = activation @ dequant_weight + bias
```

但 **硬件架构和编程范式发生了根本性变化**。910B (arch22) 是 **Membase 范式** —— 数据流向由 Cube→Fixpipe→GM 硬件固定；950 (arch35) 是 **Regbase 范式** —— Cube 输出可以走到 UB，由 Vector Core 用寄存器级指令灵活处理。

这意味着 950 上可以实现 910B 做不到的事情：**Cube 输出不经过 GM workspace 直接被 Vector Core 消费**，省掉一次 HBM 往返。

---

## 2. 代码结构

### 2.1 双层引擎（同 910B）

```
weight_quant_batch_matmul_v2/  ← ACLNN API 层  (Tiling dispatch: RegBase / ASW / BasicBlock)
quant_batch_matmul_v3/         ← AscendC 内核引擎 (arch35/ 下的模板类)
```

### 2.2 arch35 内核文件全景

```
quant_batch_matmul_v3/op_kernel/arch35/
├── qbmm_cube_on_the_fly.h              ← 内核 A: AIC-Only, Fixpipe 反量化
├── qbmm_cube_on_the_fly_al1_full_load.h ← 内核 A': A 矩阵全载入 L1
├── qbmm_mix_online_dynamic.h           ← 内核 B: AIC+AIV 混合流水线
├── qbmm_mix_online_dynamic_al1_full_load.h ← 内核 B': A 矩阵全载入 L1
├── qbmm_mix_perblock.h                 ← 内核 C: Per-Block 量化 (AIC+AIV)
├── qbmm_perblock_api_utils.h           ← 内核 C 详细实现 (784 行)
├── qbmm_perblock_api_param_utils.h     ← Nd2Nz 参数 / L1 offset 计算
├── qbmm_asw_block.h                    ← Adaptive Sliding Window 调度器
├── qbmm_api_utils.h                    ← Batch 循环 / CopyInA1 工具
├── quant_batch_matmul_v3_tiling_data.h ← Tiling 数据结构 (arch35)
└── mm_extension_interface/
    ├── qbmm_custom_mm_policy.h         ← 自定义 MatMul Policy (DUAL_DST_SPLIT_M)
    └── qbmm_copy_cube_out.h            ← 自定义 Cube 输出 (Fixpipe+quant)
```

---

## 3. Regbase vs Membase：950 的根本变化

| 维度 | 910B (arch22) | 950 (arch35) |
|------|---------------|--------------|
| 编程范式 | **Membase** | **Regbase + SIMT** |
| Cube 输出目标 | Fixpipe → GM (固定) | Fixpipe → **UB** (可编程) |
| L0C 直出 | 不支持 | 支持 (`supportL0c2Out`) |
| L1 作为 B 矩阵源 (bf16) | 不支持 | 支持 (`supportL12BtBf16`) |
| AIC/AIV 混合模式 | CV Ping-Pong (粗粒度) | CrossCoreSetFlag 流水线 (细粒度) |
| 向量寄存器宽度 | 128-bit | **256-bit** |
| MatmulConfig | `GetMDLConfig(...)` | `GetMDLConfig(..., openUnitFlag=true)` |

**最关键的差异**：arch22 的 BF16 和 Per-token 路径必须让 Cube 输出先写到 GM workspace，再由 AIV 读回 UB 做反量化 —— 这意味着每个 tile 多一次 HBM 写+读。而 arch35 的 `Mc2QuantBmmPertokenRegbaseKernel` 让 Cube 输出直接走到 UB (`TPosition::VECIN`)，AIV 直接从 UB 取数做寄存器级反量化。

---

## 4. Adaptive Sliding Window (ASW) 分块调度

### 4.1 问题

矩阵乘 `[M, K] × [K, N]` 需要分配给 N 个 AI Core。简单的 round-robin 分配会导致：
- 尾块负载不均衡
- 跨 batch 的 cache 抖动

### 4.2 ASW 策略 (`qbmm_asw_block.h:93-262`)

```
M×N 矩阵分解为 mCnt × nCnt 个 baseM × baseN 块

初始化 (Init, 行 124-146):
  mCnt = ceil(M / baseM)
  nCnt = ceil(N / baseN)
  totalCnt = mCnt × nCnt
  round = ceil(totalCnt / usedCoreNum)   ← 每个 core 处理多少轮
  mCoreNum = min(4, mCnt)                ← 滑动窗口宽度 (最多 4)
  mainRow = mCnt / mCoreNum - 1          ← 完整窗口行数
  mTailCoreNum = mCnt - mCoreNum × mainRow  ← 尾行 M 块数

每轮分配 (UpdateBasicIndex, 行 148-166):
  rowIdx < mainRow (完整窗口):
    mIndex = row × mCoreNum + index % mCoreNum
    nIndex = (index / mCoreNum) % nCnt
  rowIdx == mainRow (尾行):
    mIndex = mainRow × mCoreNum + tailIndex % mTailCoreNum
    nIndex = (tailIndex / mTailCoreNum) % nCnt

  蛇形翻转 (行 163-165):
    if rowIdx & 1:  nIndex = nCnt - 1 - nIndex  ← 奇数行 N 方向反转
```

**设计意图**：
- WINDOW_LEN=4 是因为 950 的 core 间共享 L2，window 太大会导致 L2 cache 竞争
- 蛇形翻转让相邻 core 的访存方向交替，减少 L2 bank conflict
- 尾块拆分 (行 185-216)：最后一轮的块可以被多个 core 协作处理 (`mTailTile × nTailTile` way split)

### 4.3 尾块拆分 (`UpdateBlockParams`, 行 176-217)

```cpp
if (roundIdx == params_.round - 1) {
    // 最后一轮: 将 tail block 拆分给多个 core
    singleCoreMSplit = (singleCoreM + mTailTile - 1) / mTailTile;
    singleCoreNSplit = (singleCoreN + nTailTile - 1) / nTailTile;
    // NZ 格式需要额外对齐
    if constexpr (formatX2 != ND) {
        singleCoreNSplit = Align(singleCoreNSplit, bTrans ? 16 : 32);
    }
    // 计算当前 core 负责的子块偏移
    mSplitAddrOffset = (blockIdx % totalTailTile) % mTailTile * singleCoreMSplit;
    nSplitAddrOffset = (blockIdx % totalTailTile) / mTailTile * singleCoreNSplit;
}
```

---

## 5. 内核 A: Pure-Cube On-The-Fly (`qbmm_cube_on_the_fly.h`)

### 5.1 类结构

```cpp
template <x1Type, x2Type, inputScaleType, biasType, yType,
          formatX1, formatX2, formatY, aTrans, bTrans>
class MatMulASWKernel {
    MmType mm_;   // matmul::MatmulImpl (MX 或非 MX 版本)
    Mc2QuantBmmAswBlock block_;  // ASW 调度器
};
```

### 5.2 模板条件分派 (Init, 行 60-76)

```cpp
// MX 格式 → MatmulTypeWithScale (带 scale tensor)
// 非 MX  → MatmulType (不带 scale tensor)
using MmType = Conditional<
    IsMxType<scaleType>(),
    MatmulImpl<..., MatmulWithScalePolicy>,    // MX: 硬件 scale lookup
    MatmulImpl<..., default_policy>            // 非 MX: SetQuantScalar/Vector
>::type;
```

### 5.3 Scale 处理三种路径 (`UpdateGlobalAddr`, 行 95-135)

```
1. MX 格式 (fp8_e8m0 scale):
   scaleAGlobal_ = perTokenScale  (A 矩阵的 per-token scale)
   scaleBGlobal_ = scale          (B 矩阵的 per-channel/block scale)
   → SetTensorScaleA / SetTensorScaleB 将 scale 送入 Matrix Engine

2. DoubleScale 模式:
   deqScale = scale × perTokenScale  (两个 float 标量相乘)
   scaleScalar = uint32(deqScale) & 0xFFFFE000  ← 取高 19 位给 Fixpipe
   → SetQuantScalar(scaleScalar)

3. PerTensor / PerChannel:
   PerTensor: scaleScalar = *scale (bf16→左移16→uint32 / uint32直读)
   PerChannel: scaleGlobal_ 指向 scale GM buffer
   → SetQuantScalar(scaleScalar) 或 SetQuantVector(scaleGlobal_[offset])
```

**关键细节 `DEQ_SCALE_MUL = 0xFFFFE000`** (行 29): Fixpipe 硬件只取高 19 位作 scale。这意味着 float32 scale 被截断到约 4.8 位十进制精度。DoubleScale 模式预先将两个 scale 乘好再截断，减少了精度损失。

### 5.4 算子执行 (`SetMMParaAndCompute`, 行 216-238)

```cpp
// 1. 配置 scale (MX / scalar / vector)
if (MX)  mm_.SetTensorScaleA(...), mm_.SetTensorScaleB(...);
else if (perTensor || doubleScale)  mm_.SetQuantScalar(scaleScalar);
else  mm_.SetQuantVector(scaleGlobal_[offsetScale]);

// 2. 配置 bias (可选)
if (isBias)  mm_.SetBias(biasGlobal_[offsetBias]);

// 3. 执行
mm_.SetTensorA(aGlobal_[offsetA], aTrans);
mm_.SetTensorB(bGlobal_[offsetB], bTrans);
mm_.Iterate();
mm_.GetTensorC(cGlobal_[offsetC]);
```

**数据流**: `GM(A,B) → L1 → L0A/L0B → Cube(MAC) → L0C(int32) → Fixpipe(×scale,+bias,cast) → GM(C)`

### 5.5 AIC-Only

```cpp
void Init(...)  { if ASCEND_IS_AIV { return; } }   // 行 85-87
void Process()  { if ASCEND_IS_AIV { return; } }   // 行 140-142
```

AIV core 完全空闲。这种模式适用于 scale 处理足够简单（标量或 per-channel 向量）、Fixpipe 可以直接完成反量化的场景。

---

## 6. 内核 B: AIC+AIV 混合在线动态 (`qbmm_mix_online_dynamic.h`)

这是 950 上**最复杂、最重要**的内核变体。适用于 per-token 量化、W8A8 双端量化、BF16 输出等需要额外 Vector 处理的场景。

### 6.1 类结构与模板参数

```cpp
template <aType, bType, scaleType, biasType, ptScaleType, cType,
          aFormat, bFormat, cFormat, aTrans, bTrans, l0cDtype,
          blockType, mmCfg>
class Mc2QuantBmmPertokenRegbaseKernel {
    // Matmul 输出到 VECIN (UB), 不是 GM
    using cT = MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, l0cDtype>;
    MatmulImpl<aT, bT, cT, biasT, mmCfg, ..., QBmmCustomMatmulPolicy> mm;
};
```

**关键**: `cT = MatmulType<TPosition::VECIN, ...>` —— Cube 输出目标不是 GM，而是 UB (`VECIN`)。这是 Regbase 范式的核心能力。

### 6.2 双核同步机制

```cpp
// 行 97-108: 硬同步原语
void NotifyCube()   { CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG); }
void WaitForVector(){ CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
                      CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX); }
void NotifyVector() { CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIC_FLAG);
                      CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIC_FLAG + FLAG_ID_MAX); }
void WaitForCube()  { CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG); }
```

- **两对 flag**: base (`AIC_SYNC_AIV_FLAG=8`, `AIV_SYNC_AIC_FLAG=6`) 和 base+FLAG_ID_MAX(16) = 双缓冲同步
- **PIPE 绑定**: AIC 在 `PIPE_FIX` 上等待 AIV，AIV 在 `PIPE_V` 上等待 AIC —— 不同的硬件流水线阶段
- **AIC_SYNC_AIV_MODE=4**: 跨 core 同步模式 ID

### 6.3 Pipeline 流程 (`ProcessWithoutBatch`, 行 252-286)

```
for each round (ASW 分配的块):
    block_.UpdateBasicIndex(j)
    block_.UpdateBlockParams(j)      ← 确定 singleCoreM, singleCoreN
    block_.CalcGMOffset()            ← 计算各 buffer 的 GM 偏移

    ┌─── AIC (Cube Core) ───┐         ┌─── AIV (Vector Core) ───┐
    │ if j>0: WaitForVector()│         │ WaitForCube()            │
    │ MMCompute()            │         │ DequantCompute()         │
    │ NotifyVector()         │         │ NotifyCube()             │
    └────────────────────────┘         └──────────────────────────┘

末尾: AIC 加一次 WaitForVector() 吸收 AIV 多发的一次 NotifyCube
```

**冷启动优化** (行 265-267): 第一轮 (j=0) AIC 跳过 `WaitForVector()`，因为 AIV 在第一轮没有前置依赖。

### 6.4 AIC 端: MMCompute (行 330-339)

```cpp
mm.SetTensorA(aGlobal_[offsetA], aTrans);
mm.SetTensorB(bGlobal_[offsetB], bTrans);
if (isBias && !isBiasEpilogue_)  mm.SetBias(biasGlobal_[offsetBias]);
mm.Iterate();
mm.GetTensorC(l0cOutUb_, 0, true);  // ← 输出到 UB, 不是 GM!
```

`l0cOutUb_` 是 `LocalTensor<l0cDtype>` (UB 上的 int32 tensor)。这一步将 Cube 的 int32 累加结果通过 Fixpipe 搬到 UB，**不经过 GM**。

### 6.5 AIV 端: DequantCompute (行 342-384)

```
1. Split M: CV_RATIO=2
   halfSingleM = ceil(singleCoreM / 2)
   subBlock 0 处理 M[0:halfSingleM]
   subBlock 1 处理 M[halfSingleM:singleCoreM]
   如果 singleMInVec==0, 直接返回

2. CopyDataFromGm2Ub():
   - scale (GM→UB): CopyX2ScaleFromGm2Ub (per-channel, 从 scaleGlobal_ 读)
   - pertokenScale (GM→UB): CopyX1ScaleFromGm2Ub (per-token, 每行一个 scale)
   - bias (GM→UB): CopyBiasFromGm2Ub (仅在 isBiasEpilogue_ 模式, bias 在 UB 上累加)

3. UB 空间受限, 分 splitNumOfOut 次输出:
   for i in range(splitNumOfOut):        // splitNumOfOut = min(4, singleMInVec)
       dequantOutInUB = vecQueOut_.AllocTensor<cType>()  // 从 VECOUT queue 分配
       l0cOutUbAddr = l0cOutUb_ + offsetL0c               // 指向当前 M 段的 Cube 输出

       if isDoubleScale:         VFDoDequantWithX1Pertensor(...)
       elif isPertoken:          VFDoDequantWithX1Pertoken(..., offsetPtScale, mSize)
       else:                     VFDoDequantWithoutPertokenScale(...)

       vecQueOut_.EnQue(dequantOutInUB)   // 流水线入队
       dequantOutInUB = vecQueOut_.DeQue()
       CopyDequantResFromUb2Gm(mSize, ...)
       vecQueOut_.FreeTensor(dequantOutInUB)

4. FreeUbTensor(): 释放 scale / pertokenScale / bias 的 UB buffer
```

### 6.6 核心: VFDoDequant 寄存器级反量化 (行 583-673)

这是整个 950 实现中**最核心**的函数，将反量化完全放在 256-bit 向量寄存器中完成：

```cpp
template <bool isPertensor, BasicQuantMode x1QuantMode,
          bool isBiasEpilogue, class BiasDtype>
void VFDoDequant(dst, l0cOut, scale, perTokenScale, bias, mSize, nSize)
{
    eleNumPerVf = 256 / sizeof(l0cDtype);      // int32: 8 个元素/寄存器
    nLoopCnt = ceil(nSize / eleNumPerVf);      // N 方向循环次数

    __VEC_SCOPE__ {                             // 进入寄存器作用域
        for (mIdx in 0..mSize):
            for (vfBlockIdx in 0..nLoopCnt):
                // Step 1: Load Cube 输出 (int32) 从 UB → 寄存器
                DataCopy(l0cOutReg, l0cOut + offset)

                // Step 2: int32 → float32
                if (l0cDtype == int32_t):
                    Cast<float, int32_t, ctInt322Fp32>(castSrcOutReg, l0cOutReg, mask)

                // Step 3: × perChannelScale (per-tensor 用 Muls, 否则先 load scale 到 reg)
                if (isPertensor):
                    Muls(mulScaleOutReg, castSrcOutReg, scaleScalar_, mask)
                else:
                    DataCopy(scaleReg, scale + vfBlockIdx * eleNumPerVf)
                    if (scaleType != float):       // bf16 scale → float (含 interleave)
                        Cast<float, bf16, ctHalf2Fp32Zero>(castScaleReg, scaleReg, mask)
                        Cast<float, bf16, ctHalf2Fp32One>(castScaleOneReg, scaleReg, mask)
                        Interleave(castScaleReg, castScaleOneReg, ...)  // 恢复奇偶排列
                    Mul(mulScaleOutReg, castSrcOutReg, castScaleReg, mask)

                // Step 4: × perTokenScale
                if (x1QuantMode == PERTENSOR_MODE):
                    Muls(mulPtScaleOutReg, mulScaleOutReg, pertokenScaleScalar_, mask)
                elif (x1QuantMode == PERTOKEN_MODE):
                    DataCopy<DIST_BRC_B32>(perTokenScaleReg, perTokenScale + mIdx)
                    Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, mask)

                // Step 5: + bias (optional)
                if (isBiasEpilogue):
                    DataCopy(biasReg, bias + vfBlockIdx * eleNumPerVf)
                    if (bias is bf16/fp16):
                        Cast+bfloat16→float (含 interleave)
                    Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, mask)

                // Step 6: float32 → cType (fp16/bf16/float32)
                Cast<cType, float, ctFp322Half>(castResultOutReg, addBiasOutReg, mask)

                // Step 7: Store 寄存器 → UB
                if (cType == float):
                    DataCopy<DIST_NORM_B32>(dst + offset, castResultOutReg, mask)
                else:
                    DataCopy<DIST_PACK_B32>(dst + offset, castResultOutReg, mask)
    }
}
```

**逐行解读**：

- **Step 1**: `MicroAPI::DataCopy` — 从 UB 加载 int32 到向量寄存器，一次 256-bit (8 个 int32)
- **Step 2**: `MicroAPI::Cast` with `ctInt322Fp32` trait — 整型→浮点转换，`CAST_RINT` 舍入模式
- **Step 3**: Per-channel scale 乘法。bf16 scale 需要 `Interleave` 因为 AscendC 的 bf16 寄存器采用奇偶交错排列
- **Step 4**: Per-token scale 采用 `DIST_BRC_B32` 广播模式 —— 一个 scale 值广播到整个寄存器
- **Step 5**: Bias 累加。注意 bias **在 UB 上累加** (`isBiasEpilogue_=true`) 而不是在 L0C 上。这是为了支持 bf16/fp16 格式的 bias（L0C Fixpipe 的 bias 只支持有限格式）
- **Step 6**: 最终类型转换。`ctFp322Half` trait: NO_SAT + CAST_RINT
- **Step 7**: Store。fp16/bf16 用 `DIST_PACK_B32`（pack 模式，两个 16-bit 值打到一个 32B），float32 用 `DIST_NORM_B32`（正常模式）

### 6.7 为什么 UB 只够 4 次输出？

```cpp
// Init (行 190-193): VECOUT queue 的大小
pipe_->InitBuffer(vecQueOut_, BUFFER_NUM,  // BUFFER_NUM=2 (双缓冲)
    CeilDiv(mForSingleVec, FP32_OUTPUT_TIMES) * baseN * sizeof(cType));
    // FP32_OUTPUT_TIMES=4: float32 输出时 buffer 更小 (4x 膨胀)
```

UB 被多个 buffer 共享: `l0cOutUb_` (int32 Cube 输出) + `scaleUb_` + `ptScaleUb_` + `biasUb_` + `vecQueOut_`。当 M 较大时，单次无法容纳完整 M 的输出，因此循环 `splitNumOfOut = min(4, singleMInVec)` 次。

---

## 7. 内核 C: Per-Block 量化 (`qbmm_mix_perblock.h` + `qbmm_perblock_api_utils.h`)

### 7.1 与内核 B 的区别

| 维度 | 内核 B (在线动态) | 内核 C (Per-Block) |
|------|------------------|---------------------|
| Scale 粒度 | Per-channel (N) / Per-token (M) | **Per-block (M×K 和 N×K 都有)** |
| K 方向处理 | 单次 MatMul (K 一次 Iterate) | **K 方向循环** (每 PER_BLOCK_SIZE=128 一个子 K) |
| AIC 操作 | SetTensorA/B → Iterate → GetTensorC | Nd2Nz → LoadData2D → Mmad → Fixpipe (每 K-block) |
| Scale 组合 | scale × pertokenScale | **scaleX1 × scaleX2 × pertokenScale** (三步乘法) |
| 核心文件 | `qbmm_mix_online_dynamic.h` | `qbmm_perblock_api_utils.h` (784 行) |

### 7.2 Per-Block K 循环

```cpp
// qbmm_perblock_api_utils.h: ProcessAivSingleK (行 374-423)
for each K-base-block (kIdx):
    // 加载 per-block scale
    scaleX1 = scaleAGm[offsetScaleX1]  // per-token scale (M 方向)
    scaleX2 = scaleBGm[offsetScaleX2]  // per-channel scale (N 方向)
    scaleMul = scaleX1 × scaleX2       // 组合 scale

    // 等 AIC 完成当前 K-block 的 Fixpipe 输出
    WaitForCube()
    // 反量化 + 跨 K 累加
    if (first K-step):  result = l0cOut × scaleMul
    else:               result += l0cOut × scaleMul
    NotifyCube()
```

**为什么需要跨 K 累加？** Per-block 量化中，不同 K-block 的 scale 不同，不能先全局 MatMul 再一次反量化。必须每个 K-block 的 Cube 输出立即做反量化，然后跨 K 累加。

### 7.3 Per-Tile 优化

当 `groupSizeM == 1` (每个 M 行一个 scale)，per-token scale 被预加载到 UB 中（通过 Nddma 2D 多拷贝），减少每次 K-block 的 GM 访问次数。

---

## 8. Batch 处理 (`qbmm_api_utils.h`)

arch35 支持最多 **4 级 batch 维度** (C1/C2/C3/C4) 的嵌套循环：

```cpp
// 模式: 4 层嵌套 for 循环
for b1 in batchC1:
    for b2 in batchC2:
        for b3 in batchC3:
            for b4 in batchC4:
                ProcessWithoutBatch()  // 处理一个 batch 元素
```

广播语义: `multiA1C1 = batchA1 / batchC1` 等等 —— 当 A 的某级 batch 维度是 C 的整数倍时，同一个 A 数据被多个 C batch 共享（广播）。

两种模式的代码几乎完全一致（`ComputeBmmOptiLoop` 和 `ProcessWithBatch` 有相同的 4 层循环结构），仅 Batch 大小和 multi 比例参数不同。

---

## 9. 特殊路径: A 矩阵 L1 全载

`qbmm_cube_on_the_fly_al1_full_load.h` 和 `qbmm_mix_online_dynamic_al1_full_load.h`

**条件**: 当 `singleCoreM × K` 足够小，A 矩阵能完全装入 L1。

**优化**: A 矩阵在 kernel 初始化时通过 `CopyInA1` 一次性载入 L1 (`TPosition::TSCM`)，后续每轮不再从 GM 读 A。

```cpp
// qbmm_api_utils.h: CopyInA1 (行 28-79)
// 将 A 从 GM copy 到 L1
void CopyInA1(aGlobal_, aL1Buffer_, ...)
// 然后在 mm.SetTensorA 时用 TPosition::TSCM 而不是 GM
```

这对应 LLM Decode (M=1, K=4096) 的典型场景 —— A 矩阵只有一个 token，完全可以放进 L1。

---

## 10. Fixpipe 分拆策略 (`UpdatePerBlockMmParam`)

arch35 的 Fixpipe 支持两种分拆模式 (`qbmm_asw_block.h:384-406`)：

```
fixpipeSplitN = (singleCoreN > 128) || (singleCoreM == 1)

Split-N 模式:                    Split-M 模式:
  Fixpipe 输出按 N 方向分拆         Fixpipe 输出按 M 方向分拆
  给 2 个 AIV sub-block            给 2 个 AIV sub-block
  适用: N 很大 / M=1              适用: M 较大

fixpipeD (dequant 单次处理的 N 大小):
  Split-N: fixpipeN / CV_RATIO    (= fixpipeN / 2)
  Split-M: fixpipeN               (= N 全量)
```

**逻辑**: 
- `N > 128` 时选 Split-N —— 一个 AIV 处理不了全量 N
- `M == 1` 时选 Split-N —— 只有 1 行，分拆 M 没意义

---

## 11. 与 910B 的关键差异总结

| 特性 | 910B (arch22) | 950 (arch35) |
|------|---------------|--------------|
| **BF16 输出** | 需要 GM workspace (Cube→GM→UB→Dequant→GM) | 直接 UB→UB (Cube→UB→VFDoDequant→GM) |
| **Per-token** | 需要 GM workspace | 直接 UB→UB (VFDoDequant 寄存器级) |
| **Per-block 量化** | 不支持 | **支持** (K 方向循环 + 跨 K 累加) |
| **A 矩阵复用** | L2 cache (隐式) | **L1 显式预载** (`CopyInA1`) |
| **分块调度** | 简单 round-robin | **Adaptive Sliding Window** + 蛇形翻转 |
| **AIC/AIV 同步** | Event flag (粗粒度) | **CrossCoreSetFlag** (细粒度，绑定 PIPE 阶段) |
| **Dequant 实现** | `AscendDequant()` 高级 API | **MicroAPI 寄存器级** (更灵活，更高吞吐) |
| **Fixpipe 控制** | `SetQuantScalar/Vector` | 同上 + `SetTensorScaleA/B` (MX) + **DUAL_DST_SPLIT** |

---

## 12. 附录：关键常量速查表

| 常量 | 值 | 定义位置 | 含义 |
|------|-----|----------|------|
| `PER_BLOCK_SIZE` | 128 | `qbmm_asw_block.h:22` | Per-block 量化的 K 方向块大小 |
| `WINDOW_LEN` | 4 | `qbmm_asw_block.h:119` | ASW 滑动窗口最大宽度 |
| `DEQ_SCALE_MUL` | 0xFFFFE000 | `qbmm_cube_on_the_fly.h:29` | Fixpipe scale 高 19 位掩码 |
| `AIC_SYNC_AIV_FLAG` | 8 | `base.h:50` | AIC→AIV 同步 flag ID |
| `AIV_SYNC_AIC_FLAG` | 6 | `base.h:49` | AIV→AIC 同步 flag ID |
| `AIC_SYNC_AIV_MODE` | 4 | `base.h:51` | 跨 core 同步 mode |
| `CV_RATIO` | 2 | `base.h:52` | AIV sub-block 数 |
| `FLAG_ID_MAX` | 16 | `base.h:48` | 双缓冲 flag offset |
| `MXFP_GROUP_SIZE` | 32 | `base.h:53` | MX format group size |
| `CUBE_BLOCK` | 16 | `qbmm_asw_block.h:120` | NZ fractal 对齐 |
| `DATA_BLOCK` | 32 | `base.h:57` | 32B (256-bit) 数据对齐 |
| `FP32_OUTPUT_TIMES` | 4 | `base.h:58` | FP32 输出时 buffer 缩减因子 |
| `BUFFER_NUM` | 2 | `base.h:38` | 双缓冲数量 |
| Vector reg width | 256 bit | `qbmm_asw_block.h:84-91` | 向量寄存器宽度 |

---

## 13. 附录：完整文件索引 (arch35 only)

```
quant_batch_matmul_v3/
  op_kernel/arch35/
    qbmm_cube_on_the_fly.h              ← MatMulASWKernel (Pure-Cube, AIC-only)
    qbmm_cube_on_the_fly_al1_full_load.h ← + A 矩阵 L1 全载
    qbmm_mix_online_dynamic.h           ← Mc2QuantBmmPertokenRegbaseKernel (AIC+AIV 混合)
    qbmm_mix_online_dynamic_al1_full_load.h ← + A 矩阵 L1 全载
    qbmm_mix_perblock.h                 ← MatMulPerBlockASW (Per-block 量化)
    qbmm_perblock_api_utils.h           ← MatMulPerBlock 详细实现 (784行)
    qbmm_perblock_api_param_utils.h     ← Nd2Nz 参数 + L1 offset 计算
    qbmm_asw_block.h                    ← Mc2QuantBmmAswBlock (ASW 调度器)
    qbmm_api_utils.h                    ← CopyInA1, CopyInScaleA, ProcessWithBatch
    quant_batch_matmul_v3_tiling_data.h ← Tiling 数据结构 (arch35)
    mm_extension_interface/
      qbmm_copy_cube_out.h              ← 自定义 Cube 输出 Fixpipe
      qbmm_custom_mm_policy.h           ← QBmmCustomMatmulPolicy (DUAL_DST_SPLIT_M)
  op_host/op_tiling/arch35/
    quant_batch_matmul_v3_checker.h     ← 参数校验 (dtype/shape/量化模式)
    quant_batch_matmul_v3_checker.cpp
    adaptive_sliding_window_tiling.h    ← ASW Tiling 参数计算
    adaptive_sliding_window_tiling.cpp

weight_quant_batch_matmul_v2/
  op_host/op_tiling/arch35/
    weight_quant_batch_matmul_v2_reg_base_tiling.h     ← RegBase Tiling 策略
    weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h ← ASW Tiling
    weight_quant_batch_matmul_v2_adaptive_split_tiling.h
    weight_quant_batch_matmul_v2_basic_block_tiling.h  ← BasicBlock Tiling
    weight_quant_batch_matmul_v2_basic_block_table.h   ← Tiling 参数表
  op_kernel/arch35/
    weight_quant_batch_matmul_v2_reg_base.h            ← RegBase 内核
    weight_quant_batch_matmul_v2_reg_base_common.h
    weight_quant_batch_matmul_v2_adaptive_sliding_window.h
    weight_quant_batch_matmul_v2_asw_block.h
    weight_quant_batch_matmul_v2_vf.h                  ← Vector Function
    n_first/                                           ← A16W4 N方向优先子模块
    catlass/                                           ← Catlass 子模块 (block/iterator/pipeline/...)
```
