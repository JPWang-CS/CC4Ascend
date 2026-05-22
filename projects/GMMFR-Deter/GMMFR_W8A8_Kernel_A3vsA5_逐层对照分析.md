# GMMFR W8A8 int8 Kernel A3 vs A5 逐层对照分析

> **算子**: GroupedMatMul Finalize Routing (MoE 分组矩阵乘 + 反量化 + scatter-write)
> **配置**: W8A8 int8, activation INT8 ND × weight INT8 NZ, per-token x1Scale + per-channel x2Scale → Float32
> **芯片**: A3 (910B/910C, arch22) vs A5 (950, arch35/DAV_3510)

---

## 1. Kernel 完整逻辑流程对照 (Step-by-Step)

以下按 Kernel 执行的时间顺序，逐步骤列出 A3 和 A5 的对应逻辑。左右列是同一逻辑步骤在不同芯片上的实现。

### Step 1: 编译条件 & 入口声明

| A3 | A5 |
|----|----|
| `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` | `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310` |
| `#include "kernel_operator.h"` | `#include "kernel_basic_intf.h"` (AscendC 9.0+) |
| 直接 `__global__ __aicore__ void grouped_matmul_finalize_routing(...)` | APT 框架 `grouped_matmul_finalize_routing(...)` → `if constexpr` 分派 |
| 代码: `grouped_matmul_finalize_routing.cpp:15,110` | 代码: `grouped_matmul_finalize_routing_apt.cpp:16,60` |

### Step 2: TilingKey 解析 & 模板实例化

| A3 | A5 |
|----|----|
| **机制**: 十进制偏移字符串匹配 | **机制**: APT 模板参数枚举 |
| Host 编码: `tilingKey_ = 10000000000000000001UL + rowIdxFactor + scaleFactor + ...` | Host 编码: `GET_TPL_TILING_KEY(transA, transB, scaleType, rowIndexType)` |
| Kernel 分派: `if (TILING_KEY_IS(1000...01UL))` 逐个匹配 | Kernel 分派: `if constexpr (ATRANS==0 && BTRANS==0 && SCALETYPE==1 && ROWINDEXTYPE==0)` |
| 实例化: `GMMFR_A8W8_IMPL(int64_t, float, false, false)` 宏展开 | 实例化: `grouped_matmul_finalize_routing_pertoken_dequant<RowMajor, Nz, 1, 0>(...)` |
| 模板参数: `Param<combine, RowIndexDtype, TilingType, ScaleType, groupListType, sharedInputIsNone>` | 模板参数: `<layoutA, layoutB, scaleType, rowIndType>` |
| 代码: `grouped_matmul_finalize_routing.cpp:121-169` | 代码: `grouped_matmul_finalize_routing_apt.cpp:94-178` |

**W8A8 场景 TilingKey 对照表**:

| 场景 | A3 TilingKey | A5 模板参数 |
|------|-------------|------------|
| rowIdx=int64, scale=float, GL=cumsum | `10000000000000000001UL` | `(0,0,1,0)` |
| rowIdx=int32, scale=float, GL=cumsum | `10000000000000000011UL` | `(0,0,1,1)` |
| rowIdx=int64, scale=bf16, GL=cumsum | `10000000000000000101UL` | `(0,0,2,0)` |
| rowIdx=int32, scale=bf16, GL=cumsum | `10000000000000000111UL` | `(0,0,2,1)` |
| rowIdx=int64, scale=float, GL=direct | `10000000000000001001UL` | `(0,0,1,0)` + groupListType=1 |
| rowIdx=int32, scale=float, GL=direct | `10000000000000001011UL` | `(0,0,1,1)` + groupListType=1 |
| rowIdx=int64, scale=bf16, GL=direct | `10000000000000001101UL` | `(0,0,2,0)` + groupListType=1 |
| rowIdx=int32, scale=bf16, GL=direct | `10000000000000001111UL` | `(0,0,2,1)` + groupListType=1 |
| B=Zn 格式 | A3 不支持 | `(0,1,1,0)` ~ `(0,1,2,1)` 共4种 |

### Step 3: TilingData 获取

| A3 | A5 |
|----|----|
| `GET_TILING_DATA(tilingData, tilingGM)` 宏 | `REGISTER_TILING_DEFAULT(GMMFinalizeRoutingTilingData)` + `GET_TILING_DATA(tilingData, tilingGM)` |
| TilingData 结构: 扁平 struct + 宏 getter/setter | TilingData 结构: 嵌套 POD struct `{DataParams; TCubeTiling;}` |
| 成员访问: `tilingData.matmulTiling.baseM` | 成员访问: `tilingData.gmmFinalizeRoutingDataParams.groupNum` / `tilingData.matmulTiling.baseM` |
| 代码: `grouped_matmul_finalize_routing.cpp:116` | 代码: `grouped_matmul_finalize_routing_pertoken_dequant.h:32-33` |

### Step 4: TPipe 初始化

| A3 | A5 |
|----|----|
| **手动创建**: `TPipe pipe;` | **框架管理**: 由 `GetTPipePtr()` 获取 |
| AIC/AIV 各自独立创建 | AIC 在 `mmadOp_.Init()` 中获取, AIV 在 `prologueOp_/epilogueOp_` 中使用 |
| 代码: `grouped_matmul_finalize_routing.cpp:38` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:339` |

### Step 5: Init — 绑定 Global Tensor & 分配 UB

**逻辑目标**: 将 GM 地址绑定到 GlobalTensor, 在 UB 上分配工作缓冲区。

| A3: `QuantGroupMatmul::Init()` + `InitUbBuffer()` | A5: `prologueOp_.Init()` / `epilogueOp_.Init()` / `mmadOp_.Init()` |
|----|----|
| **GM 绑定**: 逐个 `SetGlobalBuffer()` 到类成员 | **GM 绑定**: 通过 `Params` 结构体传入各组件, 各自 `SetGlobalBuffer()` |
| `xGm.SetGlobalBuffer((__gm__ int8_t*)initParams.x)` | `aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr)` 在 `UpdateGlobalBuffer()` 中 |
| `weightGm.SetGlobalBuffer(...)` | `bGlobal_.SetGlobalBuffer(...)` |
| `scaleGm.SetGlobalBuffer(...)` | `x2ScaleGlobal_.SetGlobalBuffer(...)` (Epilogue 内) |
| `perTokenScaleGm.SetGlobalBuffer(...)` | `x1ScaleGlobal_.SetGlobalBuffer(...)` (Epilogue 内) |
| `logitsGm.SetGlobalBuffer(...)` | `logitGlobal_.SetGlobalBuffer(...)` (Epilogue 内) |
| `tokenRanksGm.SetGlobalBuffer(...)` | `rowIndexGlobal_.SetGlobalBuffer(...)` (Epilogue 内) |
| `yGm.SetGlobalBuffer(...)` | `yGlobal_.SetGlobalBuffer(...)` (Epilogue 内) |
| **UB 分配**: 手动 `GetWithOffset` 分片 | **UB 分配**: 位置式 `LocalTensor(VECIN/VECOUT, offset, size)` |
| `pipe->InitBuffer(scaleInQueue, 2, baseN*sizeof(float))` | `x2ScaleUbPing_ = LocalTensor<x2Scale>(VECIN, afterLogit, 256)` |
| `pipe->InitBuffer(vecInQueue, 2, ubCalSize*sizeof(int32_t))` | `l0cOutUb_ = LocalTensor<InType>(VECIN, 0, MAX_SINGLE_MNS)` (L0C直通) |
| `pipe->InitBuffer(vecOutQueue, 2, ubCalSize*sizeof(float))` | `outUbPing_ = LocalTensor<OutType>(VECOUT, afterBiasPong, 32*256)` |
| `dequantMiddleResult = tmpBuff.GetWithOffset<float>(ubCalSize, 0)` | 不需要单独分配 — 和 `l0cOutUbFloat_` 复用同一物理地址 |
| `pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(ubCalSize, ubCalSizeFloat)` | 不需要 — VF 寄存器直接算 |
| 代码: `grouped_matmul_finalize_routing.h:146-212` | 代码: `block_epilogue_dequant_finalize_routing.h:181-211` + `block_prologue_finalize_routing.h:130-144` |

### Step 6: 输出初始化 (Prologue)

**逻辑目标**: 将输出 `y[batch×N]` 的 shared_input 区域外初始化为 0, shared_input 区域搬运 residual 并 ×residualScale。

| A3: `QuantGroupMatmul::PreProcess()` + `PreProcessInit()` | A5: `BlockPrologueFinalizeRouting::operator()` |
|----|----|
| **调用入口**: `Process()` 开头 `if ASCEND_IS_AIV { PreProcess(); SyncAll(); }` | **调用入口**: `operator()` 开头 `if ASCEND_IS_AIV { prologueOp_.Init(); prologueOp_(); }` |
| **归零范围**: `[0, n×sharedInputOffset)` + `[tail×n, n×(batch-tail)]` | **归零范围**: 相同逻辑 |
| **归零方式**: `InitOutputWithZeros()` → `yGm[offset]` 逐块写0 | **归零方式**: `InitOutputWithZeros()` → `Duplicate(initWithZero_, 0)` 后 `DataCopyPad` 到 GM |
| **residual 搬运**: `DataCopyPad(residualLocal, residualGm[...])` → `Cast(bf16→float)` → `Muls(×residualScale)` → `DataCopyPad→yGm` | **residual 搬运**: `CopyInShareInput(bf16)` → `VFDoSharedCastAndMuls`: Cast + Muls 在 V 寄存器完成 → `CopyOutShareInput` |
| **流水**: 简单队列 `EnQue/DeQue` + `PipeBarrier<PIPE_V>` | **流水**: HardEvent 三级流水 `V_MTE2 → MTE2_V → V → V_MTE3 → MTE3_V` |
| 代码: `grouped_matmul_finalize_routing.h:229-287` | 代码: `block_prologue_finalize_routing.h:255-295` |

### Step 7: 同步初始化

| A3: `Process()` 中 | A5: `operator()` 中 |
|----|----|
| `SyncAll()` 在 PreProcess 之后 | `SyncAll<false>()` 在 Prologue + MatMul Init + Epilogue Init 之后 |
| 代码: `grouped_matmul_finalize_routing.h:294` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:344` |

### Step 8: Group 循环 — 读取当前 Group 的 M

**逻辑目标**: 遍历每个 Expert (group), 从 group_list GM 读取该组的 token 数量 M。

| A3 | A5 |
|----|----|
| `uint32_t m = groupTokensGm.GetValue(groupIdx)` | `int32_t splitValue = GetSplitValueFromGroupList(groupIdx, groupListType)` |
| groupListType=0 (cumsum): 在 Tiling 阶段预先存为绝对偏移, Kernel 直接读 | groupListType=0 (cumsum): `offset - preOffset_` 在 Kernel 中动态计算 |
| groupListType=1 (direct): `m = groupTokensGm.GetValue(groupIdx)` 直接值 | groupListType=1 (direct): `splitValue = groupListGm_.GetValue(groupIdx)` 直接值 |
| `if (m <= 0) continue` 跳过空组 | `if (Get<MNK_M>(problemShape_) == 0) return false` 跳过空组 |
| 代码: `grouped_matmul_finalize_routing.h:308-311` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:203-214, 266-276` |

### Step 9: 计算分块数 & 偏移

**逻辑目标**: 根据当前 group 的 M×N 和 baseM/baseN 计算分块维度, 更新各 GM tensor 的偏移。

| A3 | A5 |
|----|----|
| `mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM)` | 由 `BlockSchedulerOp::UpdateNextProblem(resProblemShape)` 内部计算 |
| `mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN)` (全局固定) | `CoordClass coord(m, n, k, baseM, baseN, baseK)` 构造坐标计算器 |
| 偏移: 手动 `offsetM += m` 累加 | 偏移: `UpdateOffset(groupIdx)` 自动计算 A/B/scale/bias/logit 的 GM 偏移 |
| `xOffset = (offsetM+mIdx*singleM) * k` | `aBaseOffset += m * k` |
| `weightOffset = groupIdx * n * k + tailN * k` | `bBaseOffset += CeilDiv(k,32)*CeilDiv(n,16)*512` (NZ格式) 或 `n*k` |
| `scaleOffset = groupIdx * n + nIdx * singleN` | `x2ScaleBaseOffset = groupIdx * n` (每 group 重置) |
| 代码: `grouped_matmul_finalize_routing.h:302-329` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:234-264, 285-291` |

### Step 10: 块调度循环 — 获取当前 Tile 坐标

**逻辑目标**: 确定当前 AIC core 应该处理哪个 (mTile, nTile) 块。

| A3: 手动 `MNBlockIdxCompute` | A5: 调度器 `bs.GetTileIdx(tileIdx)` |
|----|----|
| `curBlock = coreIdx >= preCount ? coreIdx : coreIdx + coreNum` | `while (bs.GetTileIdx(tileIdx))` 调度器自动分配 |
| `while (curBlock < curCount)` 手动循环 | 调度器内部管理 tile 遍历 |
| **小M策略** (blockDimM≤5): 行优先 | ASWT 自适应调度 (Adaptive Split With Tail) |
| `mIdx = (curBlock-count) / blockDimN` | `Get<IDX_M_TILEIDXS>(tileIdx)` |
| `nIdx = (curBlock-count) % blockDimN` | `Get<IDX_N_TILEIDXS>(tileIdx)` |
| **大M策略** (blockDimM>5): 8×8 对角块 | `Get<IDX_M_TAIL_SPLIT_TILEIDXS>(singleShape)` (尾块标记) |
| `curBlock += coreNum` 跳到下一个本 core 的 block | `Get<IDX_N_TAIL_SPLIT_TILEIDXS>(singleShape)` |
| 代码: `grouped_matmul_finalize_routing_utils.h:140-161` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:296-300` |

### Step 11: 计算当前 Tile 的实际大小 (含尾块)

**逻辑目标**: 最后一个 tile 在 M 或 N 方向可能不足 baseM/baseN, 需要截断。

| A3 | A5 |
|----|----|
| 手动 if 判断: | 调度器自动返回: |
| `curSingleN = (nIdx==blockDimN-1) ? n-tailN : singleN` | `BlockShape singleShape = bs.GetBlockShape(tileIdx)` |
| `curSingleM = (mIdx==blockDimM-1) ? m-mIdx*singleM : singleM` | `Get<MNK_M>(singleShape)` / `Get<MNK_N>(singleShape)` |
| 代码: `grouped_matmul_finalize_routing.h:340-349` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:297` |

### Step 12: Cube MatMul (AIC 端)

**逻辑目标**: 在 AIC core 上执行分块矩阵乘法 A[M_s×K] × B[N_s×K]^T → C[M_s×N_s]。

| A3: `MMCompute()` | A5: `mmadOp_()` |
|----|----|
| **设置形状**: `mm.SetOrgShape(m, n, k)` + `mm.SetSingleShape(curSingleM, curSingleN, k)` | **形状传入**: `mmSingleShape{Get<M>(singleShape), Get<N>(singleShape), k}` |
| **设置 Tensor A**: `mm.SetTensorA(xGm[xOffset])` | **A 地址**: `aGlobal_[Get<A_OFFSETS>(blockOffset_)]` |
| **设置 Tensor B**: `mm.SetTensorB(weightGm[weightOffset])` | **B 地址**: `bGlobal_[Get<B_OFFSETS>(blockOffset_)]` |
| **L2 Cache**: `weightSlice.SetL2CacheHint(CACHE_MODE_DISABLE)` (单块时) | 由 `TileCopyPolicy` 控制 |
| **迭代**: | **单次调用**: |
| `while (mm.Iterate()) {` | `mmadOp_(A, B, l0cOutUb_, mmSingleShape, transA, transB);` |
| `  mm.GetTensorC(mmOutGm[workspaceOffset], 0, true);` | |
| `  CrossCoreSetFlag<2,PIPE_FIX>(5);` // 每次Iterate后通知AIV | |
| `  workspaceOffset += baseM*baseN;` | |
| `}` | |
| **输出**: int32 → GM workspace | **输出**: int32 → `l0cOutUb_` (VECIN UB, L0C 直通) |
| **AIC→AIV 通知**: `CrossCoreSetFlag<2,PIPE_FIX>(5)` 每次 Iterate | **AIC→AIV 通知**: `NotifyVector()` = `CrossCoreSetFlag<4,PIPE_FIX>(4)` + `CrossCoreSetFlag<4,PIPE_FIX>(20)` |
| 代码: `grouped_matmul_finalize_routing.h:337-374` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:301-309` |

### Step 13: 等待 Cube 完成 (AIV 端同步)

| A3: `VectorCompute()` 中 | A5: `ProcessSingleGroup()` 中 |
|----|----|
| `CrossCoreWaitFlag(SYNC_AIC_TO_AIV)` = `CrossCoreWaitFlag(5)` | `WaitForCube()` = `CrossCoreWaitFlag<4,PIPE_V>(4)` |
| 在 `for(offsetN)` 循环内 | 在 `while(bs.GetTileIdx)` 循环内 |
| 代码: `grouped_matmul_finalize_routing.h:452` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:325` |

### Step 14: 搬运 Cube 输出 (A3 特有)

**A3 需要这一步; A5 不需要** — A5 的 L0C 输出已通过 `l0cOutUb_` 直接映射到 VECIN UB。

| A3: `DataCopyMMOut()` | A5 |
|----|----|
| `LocalTensor<int32_t> mmOutLocal = vecInQueue.AllocTensor<int32_t>()` | 不需要 — `l0cOutUb_` 已在 VECIN 位置 |
| `DataCopyPad2D(mmOutLocal, mmOutGm[offset], dimParams)` 2D搬运 | 直接通过 `(__ubuf__ DataTypeIn*)l0cOutUb_.GetPhyAddr()` 取址 |
| `vecInQueue.EnQue(mmOutLocal)` → 后面 DeQue | |
| 代码: `grouped_matmul_finalize_routing.h:527-535` | 代码: `block_epilogue_dequant_finalize_routing.h:310` (`l0cOutUbAddr = (__ubuf__ DataTypeIn*)l0cOutUb_.GetPhyAddr()`) |

### Step 15: 搬运 Scale (x2Scale / per-channel weight scale)

| A3: `DataCopyScale()` | A5: `CopyX2ScaleFromGm2Ub()` |
|----|----|
| `scaleLocal = scaleInQueue.AllocTensor<ScaleType>()` | `x2ScaleUb` = ping 或 pong buffer (已在 Init 分配) |
| `DataCopyPad(scaleLocal, scaleGm[scaleOffset], ...)` | `DataCopyPad(dst, x2ScaleGlobal_[offset], scale2UbParams, ...)` |
| `scaleInQueue.EnQue(scaleLocal)` |  |
| `scaleInUb = scaleInQueue.DeQue<ScaleType>()` | |
| `scaleInUb.SetSize(alignBaseN)` | |
| **bf16→float 转换**: 在搬入后: | **bf16→float 转换**: 在 VFDoDequant 的 V 寄存器中: |
| `Cast(mulsResultLocal[alignBaseN], scaleInUb, CAST_NONE, alignBaseN)` | `Cast<float,bf16,ctHalf2Fp32ZeroES>(castScaleReg, scaleReg, mask)` + `Cast<float,bf16,ctHalf2Fp32OneES>(castScaleOneReg, scaleReg, mask)` + `Interleave(...)` |
| 代码: `grouped_matmul_finalize_routing.h:537-554` | 代码: `block_epilogue_dequant_finalize_routing.h:483-491` |

### Step 16: 搬运 Bias (可选)

| A3: `DataCopyBias()` | A5: `CopyBiasFromGm2Ub()` |
|----|----|
| `biasLocal = biasInQueue.AllocTensor<bfloat16_t>()` | `biasUb` = ping 或 pong buffer |
| `DataCopyPad(biasLocal, biasGm[offset], ...)` | `DataCopyPad(dst, biasGlobal_[offset], ...)` |
| `biasInUb = biasInQueue.DeQue<bfloat16_t>()` | |
| `Cast(mulsResultLocal, biasInUb, CAST_NONE, alignBaseN)` | |
| 代码: `grouped_matmul_finalize_routing.h:556-572` | 代码: `block_epilogue_dequant_finalize_routing.h:493-502` |

### Step 17: 搬运 PerToken Scale (x1Scale) & Logit

| A3: `DataCopyPerTokenScale()` | A5: `CopyX1ScaleFromGm2Ub()` + `CopyInLogit()` |
|----|----|
| `perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>()` | `x1ScaleUb` + `logitUb` = ping 或 pong buffer |
| `DataCopyPad(perTokenScaleLocal[alignBaseM], perTokenScaleGm[...])` | `DataCopyPad(dst, x1ScaleGlobal_[offset], ...)` |
| `DataCopyPad(perTokenScaleLocal, logitsGm[...])` (combine=true 时复用前半段) | `DataCopyPad(logitUb, logitGlobal_[offset], ...)` 独立搬 logit |
| **×logit 融合**: `Mul(perTokenScale[alignBaseM], perTokenScale, perTokenScale[alignBaseM], curBaseM)` | **×logit 融合**: 在 `VFDoLogitMuls` 中 `DIST_BRC_B32` 广播后 × |
| 代码: `grouped_matmul_finalize_routing.h:593-619` | 代码: `block_epilogue_dequant_finalize_routing.h:252-259, 472-479` |

### Step 18: 反量化计算 (核心差异)

**逻辑目标**: `int32_C × x2Scale × x1Scale + bias → float`

| A3: `ComputeDequantAndActivate()` + `PerTokenScaleBrcb()` | A5: `VFDoDequant()` / `VFDoDequantWithX1X2Scale()` |
|----|----|
| **Step 1 — 广播 x1Scale**: | **Step 1 — int32→float**: |
| `BroadCast<float,2,1>(pertokenBrcbLocal, perTokenScaleInUb[...], {curBaseM, alignBaseN}, {curBaseM, 1}, sharedTmpLocal)` | `Cast<float,int32,ctInt322Fp32ES>(l0cOutRegFloat, l0cOutReg, maskN)` |
| **Step 2 — int32→float + ×x2Scale**: | **Step 2 — 加载 x2Scale, bf16→float**: |
| `AscendDequant(dequantMiddleResult, mmOutInUb, scaleBuf, sharedTmpLocal, {curBaseM, alignBaseN, curBaseN})` | `DataCopy(scaleReg, x2Scale+off)` → 如果是bf16: `Cast(ZeroES)`+`Cast(OneES)`+`Interleave` |
| **Step 3 — ×x1Scale**: | **Step 3 — ×x2Scale**: |
| `Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, computeSize)` | `Mul(mulScaleOutReg, l0cOutRegFloat, castScaleReg, maskN)` |
| **Step 4 — +bias**: | **Step 4 — 广播 + ×x1Scale**: |
| `Add(mulsResultLocal, dequantMiddleResult, biasCalcLocal, computeSize)` | `DataCopy<x1Scale,DIST_BRC_B32>(perTokenScaleReg, x1Scale+mIdx)` → `Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN)` |
| **Step 5 — Cast float→bf16** (combine=true 时): | **Step 5 — 加载 bias, bf16→float**: |
| `Cast(yLocal, mulsResultLocal, CAST_RINT, computeSize)` | `DataCopy(biasReg, bias+off)` → 如果是bf16: `Cast(ZeroES)`+`Cast(OneES)`+`Interleave` |
| **全部在 UB 上完成** (多次 UB 读写) | **Step 6 — +bias**: |
| | `Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN)` |
| | **Step 7 — 写回 UB**: |
| | `DataCopy<float,DIST_NORM_B32>(dst+dstUbOff, addBiasOutReg, maskN)` |
| | **全部在 V 寄存器中完成** (仅最后写回 UB 一次) |
| 代码: `grouped_matmul_finalize_routing.h:486-525` | 代码: `block_epilogue_dequant_finalize_routing.h:400-469` |

**每条指令的执行位置对比**:

| 操作 | A3 执行位置 | A5 执行位置 |
|------|-----------|-----------|
| int32→float | `AscendDequant` 内部 (UB) | `Cast<float,int32,...>` (V Reg) |
| 加载 x2Scale | `DataCopyScale` → queue (UB) | `DataCopy(scaleReg, ...)` (V Reg) |
| bf16→float (scale) | `Cast(mulsResult[alignN], ...)` (UB) | `Cast(ZeroES)+Cast(OneES)+Interleave` (V Reg) |
| ×x2Scale | `AscendDequant` 内部 | `Mul(mulScaleOutReg, ...)` (V Reg) |
| 广播 x1Scale | `BroadCast<float,2,1>(...)` (UB→UB) | `DataCopy<DIST_BRC_B32>(perTokenReg, ...)` (UB→V Reg, 硬件广播) |
| ×x1Scale | `Mul(mulsResult, ...)` (UB) | `Mul(mulPtScaleOutReg, ...)` (V Reg) |
| +bias | `Add(mulsResult, ...)` (UB) | `Add(addBiasOutReg, ...)` (V Reg) |
| Store | 无需 (结果在 UB 上) | `DataCopy<DIST_NORM_B32>(dst+off, ...)` (V Reg→UB) |

### Step 19: ×Logit (激活值乘)

| A3: `ComputeDequantAndActivate()` 中的 `Mul(yLocal, dequantMid, pertokenBrcb)` | A5: `VFDoLogitMuls()` |
|----|----|
| logit 已在 Step 17 与 perTokenScale 融合到 `pertokenBrcbLocal` | logit 独立存储, 在 VFDoLogitMuls 中逐行广播 |
| `Mul(yLocalInUb, dequantMiddleResult, pertokenBrcbLocal, computeSize)` | `DataCopy<OutType,DIST_BRC_B32>(vregLogit, logitUbAddr + i)` 硬件广播 |
| 一次 UB Mul 完成 | N 方向循环: `for(j=0;j<repeatTimesRe;j++) { DataCopy(vregRe, ...); Mul(vDstReg, vregLogit, vregRe, mask); }` |
| 代码: `grouped_matmul_finalize_routing.h:508-519` | 代码: `block_epilogue_dequant_finalize_routing.h:277-302` |

### Step 20: Cast 最终输出类型 (A3 特有)

**A3 需要这一步; A5 不需要** — A5 输出为 float32, 无需 trunc。

| A3: `ComputeDequantProcess()` | A5 |
|----|----|
| `LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>()` | 不需要 |
| `Cast(yLocalInUb, mulsResultLocal, CAST_RINT, computeSize)` float→bf16 | 输出 float32, 略过 |
| `vecOutQueue.EnQue(yLocalInUb)` | |
| 代码: `grouped_matmul_finalize_routing.h:472-484` | 代码: 无对应 |

### Step 21: Scatter Write (AtomicAdd 路由写回)

**逻辑目标**: 根据 `tokenRanks[row]` 路由表，将每行的结果 AtomicAdd 到 `y[outRow * N + nOffset]`。

| A3: `VectorAtomicProcess()` | A5: `VectorAtomicProcess()` |
|----|----|
| `SetAtomicAdd<float>()` | `SetAtomicAdd<float>()` (相同) |
| `for (uint32_t i = 0; i < curVecBaseM; i++) {` | `for (uint32_t i = 0; i < curVecBaseM; i++) {` (相同) |
| `  auto outRow = tokenRanksGm.GetValue(mGlobalOffset + offsetM + i)` | `  auto outRow = rowIndexGlobal_.GetValue(offsetM + i)` (相同) |
| `  DataCopyPad(yGm[outRow*n + yGmOffset0], yLocal[i*alignBaseN], paramsOut)` | `  DataCopyPad(yGlobal_[outRow*n_ + yOffset], yLocal[i*alignN_], paramsOut)` (相同) |
| `}` | `}` |
| `SetAtomicNone()` | `SetAtomicNone()` (相同) |
| **paramsOut**: `{1, curVecBaseN*sizeof(float), 1, 1, 0}` (srcStride=1,dstStride=1) | **paramsOut**: `{1, curBaseN*sizeof(OutType), 0, 0, 0}` (srcStride=0,dstStride=0) |
| 代码: `grouped_matmul_finalize_routing.h:376-402` | 代码: `block_epilogue_dequant_finalize_routing.h:262-274` |

### Step 22: 通知 AIC — Vector 完成

| A3 | A5 |
|----|----|
| `CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC)` = event 3 | `NotifyCube()` = `CrossCoreSetFlag<4, PIPE_V>(6)` |
| 在 VectorCompute 末尾 | 在 ProcessSingleGroup 末尾 |
| 代码: `grouped_matmul_finalize_routing.h:468` | 代码: `kernel_gmm_finalize_routing_pertoken_dequant.h:327` |

### Step 23: AIC 等待 AIV (workspace slot 复用)

**A3 特有; A5 不需要** — A5 没有 workspace 复用问题。

| A3: `MMCompute()` 中 | A5 |
|----|----|
| `if (cubeCount >= tiling->parallNum) {` | 不需要 |
| `  CrossCoreWaitFlag(SYNC_AIV_TO_AIC);` = WaitFlag(3) | |
| `}` | |
| 代码: `grouped_matmul_finalize_routing.h:355-357` | 代码: 无对应 |

### Step 24: ×Logit + Scatter 流水细节 (A5 Epilogue 内部流水)

A5 的 Epilogue 内部有精细的三级流水，A3 没有对应粒度:

```
A5 Epilogue operator() 内部流水:

  MTE2 ──────── V ──────── MTE3
  (搬入数据)    (计算)     (搬出结果)

  Ping:
  CopyInLogit(logitUbPing)     VFDoDequant(x2ScalePing, x1ScalePing, biasPing)
  CopyX2Scale(x2ScalePing) ──→ VFDoLogitMuls(logitUbPing, outUbPing) ──→ VectorAtomicProcess(outUbPing)
  SetFlag<MTE2_V>(0)           SetFlag<V_MTE3>(0)                       SetFlag<MTE3_V>(0)

  Pong: (同时进行)
                               VFDoDequant(x2ScalePong, ...)
  CopyInLogit(logitUbPong) ──→ VFDoLogitMuls(logitUbPong, outUbPong) ──→ VectorAtomicProcess(outUbPong)
  SetFlag<MTE2_V>(1)           SetFlag<V_MTE3>(1)                       SetFlag<MTE3_V>(1)
```

A3 的流水颗粒度较粗 — 仅在 `vecInQueue/vecOutQueue` 的双缓冲层面有隐式流水, 没有 MTE2/V/MTE3 的显式事件流水。

---

## 2. Shape 不同时的处理对比

### 2.1 Tiling 层: baseM/baseN 选择策略

| Shape 场景 | A3 (BaseTiling) | A5 (QuantTiling) |
|-----------|----------------|-----------------|
| **通用 W8A8** | `baseM = 128`, `baseN = 256`, `baseK = 128` 固定 | `CalBasicBlock()` 动态计算 (基于 ubSize/l1Size/l0CSize) |
| **n=7168/7680, k=2048** | 特殊分支: avg_m>128 且 avg_m≤256 时 swap → `baseM=256, baseN=128` | 无特殊分支, `CalBasicBlock` 自适应 |
| **avg_m > 128** | `baseM = (avg_m>128 && avg_m≤256) ? 256 : 128` | 由 `CalBasicBlock` 内部逻辑决定 |
| **vBaseM** | `UBCALSIZE / baseN = 4096 / baseN` (16×256 / baseN) | Epilogue 的 `MAX_SINGLE_MNS = 128*256` 硬编码上限 |
| **L0C double buffer** | `dbL0C=1` (单缓冲) | `dbL0C` 由 CalBasicBlock 动态计算 |
| **L0A/L0B double buffer** | `depthA1=8, depthB1=8` (K方向×4×2) | `dbL0A=2, dbL0B=2` (M/N方向双缓冲) |

代码对照:
- A3: `grouped_matmul_finalize_routing_base_tiling.cpp:350-399` (`W8A8TilingProcess`)
- A5: `grouped_matmul_finalize_routing_quant_tiling.cpp:468-510` (`DoLibApiTiling`)

### 2.2 VF (Vector) 处理: 单 tile M 的大小限制

| 维度 | A3 | A5 |
|------|----|----|
| **Vector 单次最大 M** | `vecBaseM = ubCalSize / (Ceil(baseN,8)*8)` 动态计算 | `MAX_OUTPUT_M_UBS = 32` (硬编码) |
| **M 超过 vecBaseM 时** | `for (offsetM < offsetMEnd; offsetM += vecBaseM)` M方向循环 | `loopNumY = CeilDiv(singleMInVec, 32)` 循环 |
| **N 方向处理** | `for (offsetN < curCubeSingleN; offsetN += baseN)` N 方向循环 | `nLoopCnt = (nSize+eleNumPerVf-1)/eleNumPerVf` VF 寄存器内循环 |
| **AIC0/AIC1 分半** | `GetOffset()` 按 `subBlockIdx` 分半: `singleCoreM = curCubeSingleM/2` | `halfSingleM = CeilDiv(singleM_, GetTaskRation())` 按 taskRation 分 |

代码对照:
- A3: `grouped_matmul_finalize_routing.h:432-467`
- A5: `block_epilogue_dequant_finalize_routing.h:509-514`

### 2.3 典型 Shape 数值模拟

#### Shape 1: 小 M 场景 (M=16, N=2560, K=2048, E=8)

| 参数 | A3 | A5 |
|------|----|----|
| baseM/baseN | 128/256 (固定) | CalBasicBlock 决定 (可能更小) |
| blockDimM | Ceil(16, 128) = 1 | 调度器决定 |
| blockDimN | Ceil(2560, 256) = 10 | 调度器决定 |
| 分块策略 | 小M → 行优先 | ASWT 自适应 |
| vecBaseM | 4096/(Ceil(256,8)*8) = 4096/256 = 16 | singleM=16, loopNumY=Ceil(8,32)=1(若AIC0/1均分) |
| Workspace slot | 4个AIC轮转 (parallNum=4) | 不需要 workspace |
| 关键行为 | 每个 block 只有 K方向的 while(Iterate) 循环 | mmadOp_ 一次调用, Epilogue 单次处理 |

#### Shape 2: 大 M 场景 (M=2048, N=7168, K=2048, E=8)

| 参数 | A3 | A5 |
|------|----|----|
| baseM/baseN | swap: `baseM=256, baseN=128` (avg_m=256, 触发特殊分支) | CalBasicBlock 决定 |
| blockDimM | Ceil(2048, 256) = 8 | 调度器决定 |
| blockDimN | Ceil(7168, 128) = 56 | 调度器决定 |
| 分块策略 | 大M → 8×8 对角块 | ASWT + tail split |
| vecBaseM | 4096/(Ceil(128,8)*8) = 4096/128 = 32 | singleM=256, loopNumY=Ceil(128,32)=4 (AIC0/1均分后) |
| Workspace slot | 4 slot × 20 core = 大 workspace | 不需要 |
| 关键差异 | baseM 比 baseN 还大 (256 vs 128) | A5 的 CalBasicBlock 可能选择不同的 baseM/N 比例 |

#### Shape 3: 非均匀 Group 场景 (M=[512, 8, 1024, 0, 256], N=4096, K=2048, E=5)

| 参数 | A3 | A5 |
|------|----|----|
| 空组跳过 (M=0) | `if (m <= 0) continue` | `if (Get<MNK_M>(problemShape_) == 0) return false` |
| groupList=cumsum | groupTokens 存绝对偏移 `[512, 520, 1544, 1544, 1800]` | groupListGm_ 存相同偏移, Kernel 内 `offset-preOffset_` 反算 |
| preCount 逻辑 | `preCount = curCount % coreNum` 剩余块分给后续 group | 无 preCount — 调度器独立处理每组 |
| offsetM 累加 | `mnConfig.offsetM += mnConfig.m` 跨 group 累积 | `aBaseOffset += m*k` 跨 group 累积 (在 UpdateOffset 中) |

---

## 3. 同步机制详细对照

### 3.1 AIC ↔ AIV 跨核同步

```
A3 (mode=2):                           A5 (mode=4):

事件映射:                              事件映射:
  AIC→AIV: event=5 (单事件)              AIC→AIV: event=4 + event=20 (双事件)
  AIV→AIC: event=3 (单事件)              AIV→AIC: event=6 + event=22 (双事件)

发送方:                                 发送方:
  CrossCoreSetFlag<2, PIPE_FIX>(5)      CrossCoreSetFlag<4, PIPE_FIX>(4)  +  CrossCoreSetFlag<4, PIPE_FIX>(20)
  CrossCoreSetFlag<2, PIPE_MTE2>(3)     CrossCoreSetFlag<4, PIPE_V>(6)

等待方:                                 等待方:
  CrossCoreWaitFlag<2, PIPE_FIX>(5)     CrossCoreWaitFlag<4, PIPE_FIX>(4)  +  CrossCoreWaitFlag<4, PIPE_FIX>(20)
  CrossCoreWaitFlag<2, PIPE_FIX>(3)     CrossCoreWaitFlag<4, PIPE_V>(6)    +  CrossCoreWaitFlag<4, PIPE_V>(22)
```

**为什么 A5 用双事件?** mode=4 时, event 4 和 event 20 (=4+16) 构成一对, 支持更细粒度的同步控制。A5 的 AIC→AIV 用 PIPE_FIX, AIV→AIC 用 PIPE_V, 管道分离避免死锁。

### 3.2 A5 HardEvent 流水事件 (A3 无对应)

```
A5 三级流水同步链:

  MTE2 (搬运引擎2)          V (向量引擎)           MTE3 (搬运引擎3)
  ═══════════════          ════════════           ═══════════════
  CopyInLogit ──SetFlag(MTE2_V, id)──→ WaitFlag(MTE2_V, id)
  CopyX2Scale                           VFDoDequant
  CopyX1Scale                           VFDoLogitMuls
  CopyBias                              ──SetFlag(V_MTE3, id)──→ WaitFlag(V_MTE3, id)
                                                                    VectorAtomicProcess
                                                                    ──SetFlag(MTE3_V, id)
                                        ←──WaitFlag(MTE3_V, id)──

  Ping (id=0) 和 Pong (id=1) 交替执行, 形成双缓冲流水。
```

A3 无此粒度的流水 — 仅依赖队列 `EnQue/DeQue` 的隐式同步和 `PipeBarrier<PIPE_V>` 的显式栅栏。

---

## 4. AIC/AIV 工作划分对照

| 操作 | A3 位置 | A5 位置 |
|------|--------|--------|
| TilingData 获取 | AIC+AIV 都执行 | AIC+AIV 都执行 |
| GM 绑定 | AIC+AIV 都执行 (Init) | AIC+AIV 各自在自己的 Init 中绑定需要的 |
| UB 分配 | AIC 直接 return, AIV 执行 InitUbBuffer | AIC 仅在 `mmadOp_.Init()` 中, AIV 在 `prologueOp_.Init()` + `epilogueOp_.Init()` 中 |
| 输出初始化 (Prologue) | AIV 执行 `PreProcess()` | AIV 执行 `prologueOp_()` |
| Cube MatMul | AIC 执行 `MMCompute()` | AIC 执行 `mmadOp_()` |
| 搬运 Cube 输出 | AIV `DataCopyMMOut` 从 workspace | 不需要 (L0C直通) |
| 搬运 Scale/Bias | AIV `DataCopyScale/DataCopyBias` | AIV MTE2 `CopyX2ScaleFromGm2Ub/CopyBiasFromGm2Ub` |
| 搬运 PerTokenScale/Logit | AIV `DataCopyPerTokenScale` | AIV MTE2 `CopyX1ScaleFromGm2Ub/CopyInLogit` |
| 反量化 | AIV V 引擎 `AscendDequant`+`Mul`+`Add` (UB级) | AIV V 引擎 `VFDoDequant` (寄存器级) |
| ×Logit | AIV V 引擎 `Mul` (UB级) | AIV V 引擎 `VFDoLogitMuls` (寄存器级) |
| Cast float→bf16 | AIV V 引擎 `Cast` (UB级) | 不需要 (输出float32) |
| Atomic Scatter | AIV MTE3 `DataCopyPad` | AIV MTE3 `DataCopyPad` |
| 确定性回写 | AIV `FRDeterministic()` + `queBind` | A5 W8A8 无此分支 |

---

## 5. 关键数据结构对照

### TilingData

```
A3: GroupMatmulFRTilingData (扁平struct, 宏生成getter/setter)
┌─────────────────────────────────┐
│ TCubeTiling matmulTiling        │  ← MatMul tiling 参数
│   .baseM, .baseN, .baseK        │
│   .dbL0C = 1                    │
│   .stepKa = 4, .stepKb = 4      │
│   .depthA1 = 8, .depthB1 = 8    │
├─────────────────────────────────┤
│ uint32_t coreNum                │
│ uint32_t groupNum               │
│ uint32_t batch, n, k            │
│ uint32_t vBaseM = 16            │
│ uint32_t ubCalSize = 4096       │
│ uint32_t parallNum = 4          │
│ uint32_t sharedInputOffset/Len  │
│ float residualScale             │
│ ... (共 ~20 个字段)              │
└─────────────────────────────────┘

A5: GMMFinalizeRoutingTilingData (嵌套 POD struct)
┌─────────────────────────────────────┐
│ GMMFinalizeRoutingDataParams        │
│   .groupNum, .batch                 │
│   .sharedInputOffset, .sharedInputLen│
│   .residualScale                    │
│   .aQuantMode, .bQuantMode          │
│   .biasDtype, .groupListType        │
│   .hasBias                          │
├─────────────────────────────────────┤
│ TCubeTiling matmulTiling            │
│   .M, .N, .Ka, .Kb                 │
│   .baseM, .baseN, .baseK           │
│   .singleCoreM, .singleCoreN        │
│   .dbL0A = 2, .dbL0B = 2           │
│   .stepKa, .stepKb                  │
│   .depthA1, .depthB1                │
│   .usedCoreNum                      │
└─────────────────────────────────────┘
```

### MNConfig (A3) vs BlockCoord (A5)

```
A3: MNConfig (运行时状态)              A5: BlockCoord (Shape+Coord 分离)
┌──────────────────────┐              ┌────────────────────────┐
│ m, baseM, baseN      │              │ ProblemShape {M,N,K,_} │  ← 问题规模
│ mIdx, nIdx           │              │ BlockShape {M,N,K,_}   │  ← tile 大小
│ blockDimM, blockDimN │              │ BlockCoord {m,n,...}   │  ← tile 坐标
│ singleM, singleN     │              │ BaseOffset {a,b,scale..}│ ← 基础偏移
│ offsetM              │              │ BlockOffset {a,b,scale..}│ ← tile 偏移
│ workSpaceOffset      │              └────────────────────────┘
│ curBlockM            │
└──────────────────────┘
```

---

## 6. 总结: 架构演进的本质

```
A3 (arch22)                            A5 (arch35/DAV_3510)
══════════                             ═════════════════════
单体类                                 组件化 (Builder + Prologue + Epilogue + Scheduler)
GM workspace 中转                       L0C→VECIN 直通 (零搬运)
UB 级计算 (反复 UB 读写)                VF 寄存器级计算 (一次 load, 多次 ALU)
队列隐式同步                            HardEvent 显式三级流水 (MTE2→V→MTE3)
手动偏移 UB 分片                        位置式 LocalTensor 声明
手动 MN 分块策略                        ASWT 调度器自动分配
固定 baseM/baseN                       CalBasicBlock 动态计算
bf16 trunc (Cast CAST_RINT)            输出 float32 (无精度损失)
手动 while(Iterate) K 循环              mmadOp_ 单次调用
仅 NZ 权重格式                          NZ + Zn 双格式
```

**最核心的三条演进**:
1. **零搬运**: L0C→VECIN 直接映射省去 GM workspace 中转
2. **寄存器计算**: 反量化全流程在 V 寄存器完成, 从 UB 多次读写 → 寄存器级流水
3. **显式流水**: HardEvent 让 MTE2/V/MTE3 三级流水显式可控, 替代队列隐式同步

---

> **源文件索引**
> - A3 Kernel 入口: `op_kernel/grouped_matmul_finalize_routing.cpp:110-171`
> - A3 Kernel 类: `op_kernel/grouped_matmul_finalize_routing.h:76-688`
> - A3 工具/数据结构: `op_kernel/grouped_matmul_finalize_routing_utils.h:19-203`
> - A3 Tiling: `op_host/grouped_matmul_finalize_routing_base_tiling.cpp:350-399, 491-521`
> - A5 Kernel 入口: `op_kernel/grouped_matmul_finalize_routing_apt.cpp:60-181`
> - A5 调度封装: `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h:27-91`
> - A5 Kernel 类: `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h:67-364`
> - A5 Epilogue: `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h:67-558`
> - A5 Prologue: `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h:48-295`
> - A5 Tiling: `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp:445-529`
