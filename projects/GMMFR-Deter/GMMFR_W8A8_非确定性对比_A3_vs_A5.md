# GMMFR W8A8 非确定性实现对比: A3 vs A5

> 对比范围: A3 (910B/910C) 和 A5 (950/DAV3510) 平台上 **W8A8 INT8 非确定性模式** 的 GMMFR 实现
>
> 源码位置:
> - A3: `op_kernel/grouped_matmul_finalize_routing.h` (主 kernel，`deterministicFlag=0` 路径)
> - A5: `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` + Cgmct 框架
>
> 两者都是非确定性实现，但架构理念、数据通路、调度策略完全不同。

---

## 目录
1. [核心差异速览](#1-核心差异速览)
2. [架构理念对比](#2-架构理念对比)
3. [Tiling 策略对比](#3-tiling-策略对比)
4. [MM 输出数据通路对比 (核心差异)](#4-mm-输出数据通路对比-核心差异)
5. [Tile 调度策略对比](#5-tile-调度策略对比)
6. [反量化 + Scatter-Add 实现对比](#6-反量化--scatter-add-实现对比)
7. [UB Buffer 管理对比](#7-ub-buffer-管理对比)
8. [AIC↔AIV 同步机制对比](#8-aicaiv-同步机制对比)
9. [硬件适配差异](#9-硬件适配差异)
10. [具体 Shape 示例: 4 experts × 64 tokens × k=2048 × n=7168](#10-具体-shape-示例-4-experts--64-tokens--k2048--n7168)
11. [总结对比表](#11-总结对比表)
12. [代码路径索引](#12-代码路径索引)

---

## 1. 核心差异速览

| 维度 | A3 非确定性 | A5 非确定性 |
|------|-----------|-----------|
| **框架** | 手写 AscendC 单体 kernel | Cgmct 模板组合框架 |
| **MM 输出路径** | Cube → **L2 Workspace** → UB → Dequant | Cube → **L0C → UB** → Dequant |
| **中间存储** | L2 workspace (按 core 分区) | 无 (L0C 直通) |
| **调度器** | `MNBlockIdxCompute` (手动对角/行优先) | `GroupedMatmulAswtWithTailSplitScheduler` |
| **UB 管理** | 手动 `TQue`/`TBuf` 分配 | Cgmct 框架自动管理 |
| **Tile 形状** | 固定 baseM=128, baseN=256 | GroupedQmmTiling 自动计算 |
| **AIC↔AIV 同步** | `CrossCoreSetFlag`/`WaitFlag` + `SyncAll` | 结构化 `WaitForVector/NotifyVector` 协议 |
| **Atomic Scatter** | 逐行循环 `tokenRanksGm.GetValue` + `DataCopyPad` | `VectorAtomicProcess` 逐行 `rowIndexGlobal_` + `DataCopyPad` |
| **硬件** | 910B: UB=192KB, L0C=128KB | 950: UB=256KB, L0C=256KB |

---

## 2. 架构理念对比

### 2.1 A3: 单体手写 Kernel

A3 的全部逻辑集中在 `QuantGroupMatmul<P>` 这一个类模板中 (~690 行)：

```cpp
// grouped_matmul_finalize_routing.h:77
template <class P>
class QuantGroupMatmul {
    // 成员: mm(MatmulImpl), pipe(TPipe)
    // 手动管理的 TQue: vecInQueue, vecOutQueue, scaleInQueue, biasInQueue, ...
    // 手动切分的 TBuf: dequantMiddleResult, pertokenBrcbLocal, mulsResultLocal, ...

    void Process();           // ★ 主循环: group 迭代 → tile 循环
    void MMCompute();         // Cube Matmul (AIC)
    void VectorSync();        // 确定性窗口管理 (非确定性下直接 return)
    void VectorCompute();     // 反量化 + 激活 (AIV)
    void VectorAtomicProcess(); // scatter-add 输出
    void FRDeterministic();   // 确定性刷出 (非确定性下不调用)
};
```

**设计特点**:
- 一个类囊括所有阶段，开发者掌控一切
- UB buffer 手动分配，通过 `TQue`/`TBuf`/`pipe->InitBuffer` 管理
- 每个阶段的控制流（循环、条件、偏移计算）全部显式编写
- 确定性/非确定性通过 `if (tiling->deterministicFlag == 1)` 分支切换

### 2.2 A5: Cgmct Block 组合框架

A5 的 W8A8 kernel 是一个 ~60 行的 **类型组装** 函数：

```cpp
// grouped_matmul_finalize_routing_pertoken_dequant.h:28-94
template <typename layoutA, typename layoutB, int scaleType, int rowIndType>
__aicore__ inline void grouped_matmul_finalize_routing_pertoken_dequant(...) {
    // 1. 组装 Block 类型
    using BlockMmadBuilder = Block::BlockMmadBuilder<...>;          // Cube MMAD
    using BlockPrologue    = BlockPrologueFinalizeRouting<...>;     // 初始化
    using BlockEpilogue    = BlockEpilogueDequantFinalizeRouting<...>; // 反量化
    using BlockScheduler   = GroupedMatmulAswtWithTailSplitScheduler;  // 调度

    // 2. 组装 Kernel 类型
    using GmmKernel = KernelGmmFinalizeRoutingPertokenDequant<
        ProblemShape, BlockMmadBuilder, BlockPrologue, BlockEpilogue, BlockScheduler>;

    // 3. 填充参数并运行
    GmmKernel gmm;
    gmm(params);
}
```

**设计特点**:
- 各阶段 (Prologue/MMAD/Epilogue) 是独立、可替换的 Block
- UB 管理、同步、数据搬移封装在 Block 内部
- 调度器 (`BlockScheduler`) 是独立模块，可替换
- 框架负责 tile 循环控制流

### 2.3 架构对比图

```
A3 (手写单体):
  ┌──────────────────────────────────────────────┐
  │          QuantGroupMatmul<P>                 │
  │  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
  │  │ InitUbBuf │  │ Process  │  │FRDeterministic│
  │  │(手动分配) │  │(主循环)  │  │(非det不调用) │
  │  └──────────┘  └─────┬────┘  └────────────┘ │
  │                      │                       │
  │  ┌──────┐ ┌──────┐ ┌─┴──────┐ ┌──────────┐  │
  │  │MMComp│ │VecSyn│ │VecComp │ │VecAtomic │  │
  │  │(AIC) │ │(nop) │ │(AIV)   │ │Process   │  │
  │  └──────┘ └──────┘ └────────┘ └──────────┘  │
  └──────────────────────────────────────────────┘

A5 (Cgmct 组合):
  ┌──────────────────────────────────────────────────────┐
  │   grouped_matmul_finalize_routing_pertoken_dequant   │
  │   (类型组装, ~60 lines)                              │
  └──────────────────────┬───────────────────────────────┘
                         │ 组合
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  ┌──────────────┐ ┌────────────┐ ┌──────────────────────┐
  │BlockPrologue │ │BlockMmad   │ │BlockEpilogueDequant  │
  │(独立Block)   │ │Builder     │ │FinalizeRouting(独立) │
  └──────────────┘ └────────────┘ └──────────────────────┘
          │              │              │
          └──────────────┼──────────────┘
                         │ 调度
                         ▼
          ┌──────────────────────────────────┐
          │ GroupedMatmulAswtWithTailSplit   │
          │ Scheduler (独立模块)             │
          └──────────────────────────────────┘
```

---

## 3. Tiling 策略对比

### 3.1 Tile 形状

**A3** (`base_tiling.cpp:350-365`):
```cpp
// W8A8TilingProcess 中硬编码
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
// n=7168/7680, k=2048 时的特殊优化:
if (avg_m > 128 && avg_m <= 256) {
    baseM = 256; baseN = 128;  // M/N 互换
}
```
baseM=128, baseN=256 固定（除特定 shape 互换），由 L0C=128KB 约束决定: 128×256×4=128KB。

**A5** (`GroupedQmmTiling::CalBasicBlock`):
```cpp
// 自动计算，适配硬件
CalBasicBlock()  // 考虑 UB/L1/L0C 容量、dtype、format
CalL1Tiling()    // 计算 L1 分块深度
```
baseM/baseN 未硬编码，由框架根据硬件 buffer 容量自动推导。A5 的 L0C=256KB 允许更大的 tile（如 256×256×4=256KB）。

### 3.2 Tiling 数据

| 字段 | A3 (20 字段) | A5 (2 字段) |
|------|------------|-----------|
| matmulTiling | TCubeTiling (所有 Cube 参数) | TCubeTiling |
| 确定性标记 | `deterministicFlag`, `deterWorkspaceSize` | 无 |
| UB 参数 | `ubCalSize=4096`, `ubRestBytes=101376`, `vBaseM` | 框架内部管理 |
| GMM 元数据 | 打平在 tiling 结构体中 | `GMMFinalizeRoutingDataParams` 子结构体 |
| 量化参数 | `withOffset`, `hasPertokenScale`, `hasBias` | `aQuantMode`, `bQuantMode`, `biasDtype` |

### 3.3 TilingKey

**A3**: 基于 `10000000000000000001UL` 的手动位编码，通过修改十进制位区分 rowIndex/scale/groupListType/sharedInput 组合。

**A5**: 4 参数模板 key `GET_TPL_TILING_KEY(ATRANS, BTRANS, SCALETYPE, ROWINDEXTYPE)`，共 24 种组合，由框架自动生成。

---

## 4. MM 输出数据通路对比 (核心差异)

这是两者最根本的差异。

### 4.1 A3: Cube → L2 Workspace → UB → Dequant

```
AIC Cube Core                         L2 Workspace
┌──────────────┐                   ┌────────────────┐
│ int8 × int8  │                   │  int32 buffer  │
│   Matmul     │ ──SetWorkspace──→ │  (按 core 分区) │
│ (MMCompute)  │                   │                │
└──────────────┘                   └───────┬────────┘
                                           │
                          DataCopyMMOut    │  (AIV 读回 UB)
                                           ▼
                                    ┌──────────────┐
                                    │   UB        │
                                    │ vecInQueue  │  ← int32 MM 输出
                                    │ (32KB)      │
                                    └──────┬───────┘
                                           │
                              Dequant +    │  (AIV Vector 计算)
                              Logit +      ▼
                              Scatter  ┌──────────┐
                                       │  y (GM)  │  ← float 最终输出
                                       └──────────┘
```

关键代码路径：

```cpp
// A3 MMCompute (line 369): 写 workspace
mm.SetWorkspace(workspaceGM[worskspaceOffset]);

// A3 VectorCompute: 从 workspace 读到 UB
// line 457: DataCopyMMOut 从 workspace 到 vecInQueue
// line 458-459: 构造 VectorAtomicParams

// A3 VectorAtomicProcess (non-det path, line 388-396):
SetAtomicAdd<float>();
for (i = 0; i < curVecBaseM; i++) {
    outRow = tokenRanksGm.GetValue(mGlobalOffset + offsetM + i);
    DataCopyPad(yGm[outRow * n + yGmOffset0], yLocal[i * alignBaseN], paramsOut);
}
SetAtomicNone();
```

**A3 的数据流**:
1. AIC `MMCompute`: Cube Matmul → **L2 Workspace** (int32)
2. AIV `VectorCompute`: L2 Workspace → UB `vecInQueue` (DataCopyMMOut)
3. AIV Dequant: int32 → float, ×scales, +bias
4. AIV `VectorAtomicProcess`: UB → GM `y` (atomic scatter)

**额外的 L2 往返**: MM 输出先写到 L2 workspace，再由 AIV 读回。这在 L2 带宽上产生了额外开销。

### 4.2 A5: Cube → L0C → UB → Dequant

```
AIC Cube Core                    L0C (256KB)                 UB (256KB)
┌──────────────┐              ┌──────────────┐           ┌──────────────┐
│ int8 × int8  │              │  int32 tile  │           │              │
│   MMAD       │ ──L0C write→ │  M×N output  │ ──copy→   │ l0cOutUb     │
│ (BlockMmad)  │              │              │ (L0C→UB) │ (int32)      │
└──────────────┘              └──────────────┘           │      │       │
                                                         │      ▼       │
                                                         │ Dequant +    │
                                                         │ Logit        │
                                                         │      │       │
                                                         │      ▼       │
                                      ┌──────────┐      │ outUb (float) │
                                      │  y (GM)  │ ←─── │ (scatter-add) │
                                      └──────────┘      └──────────────┘
```

关键代码路径：

```cpp
// KernelGmmFinalizeRoutingPertokenDequant::ProcessSingleGroup (kernel header):
// AIC: mmadOp_(aGlobal_, bGlobal_, l0cOutUb_, mmSingleShape, transA, transB);
//       ↑ 直接写 L0C → UB 映射

// BlockEpilogueDequantFinalizeRouting::VFDoDequantWithX1X2Scale:
// l0cOutUb (L0C→UB mapped, int32) → 读取进行 dequant

// BlockEpilogueDequantFinalizeRouting::VectorAtomicProcess:
// SetAtomicAdd<float>() → 逐行 DataCopyPad → SetAtomicNone()
```

**A5 的数据流**:
1. AIC `mmadOp_`: Cube MMAD → **L0C** (int32) → 映射为 UB `l0cOutUb_`
2. AIV `VFDoDequant`: UB 内直接 dequant (int32→float)
3. AIV `VFDoLogitMuls`: UB 内 ×logit
4. AIV `VectorAtomicProcess`: UB → GM `y` (atomic scatter)

**零 L2 往返**: MM 输出直接在 L0C→UB 中完成，无需经 L2 中转。

### 4.3 数据通路差异的影响

| 指标 | A3 (L2 Workspace) | A5 (L0C 直通) |
|------|-------------------|---------------|
| MM 输出→Dequant 延迟 | L2 write + L2 read (高延迟) | L0C→UB copy (低延迟) |
| L2 带宽占用 | 每个 tile 写+读各一次 | 零 |
| L2 容量占用 | `核心数 × baseM × baseN × 4B` | 0 |
| UB 布局 | vecInQueue (1 份) + tmpBuff (反量化临时) | l0cOutUb (L0C 映射, 1 份) + ping-pong 缓冲区 |
| 最大 tile 形状 | 受 L2 workspace 分区限制 | 受 L0C 容量限制 (256KB) |

---

## 5. Tile 调度策略对比

### 5.1 A3: MNBlockIdxCompute

```cpp
// grouped_matmul_finalize_routing_utils.h:140-161
__aicore__ inline void MNBlockIdxCompute(MNConfig& mnConfig, curBlock, count, thresholdMDimN, deterministicFlag)
{
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1 || deterministicFlag == 1) {
        // 行优先 (Row-Major): 小块或确定性模式
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        // ★ 对角策略 (Diagonal): 大块非确定性模式
        // 将 8×8 的子块按对角线分配给 core，提升 L2 cache 局部性
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN, thresholdMDimN) ?
            mnConfig.blockDimM % thresholdBlockNum : thresholdBlockNum;
        uint32_t curThresholdN = ...;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdMDimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock / LCM(curThresholdM, curThresholdN)) % curThresholdN + ...;
    }
}
```

**策略**:
- `blockDimM <= 5` 或确定性模式: **行优先**，按 `(mIdx, nIdx)` 线性分配
- `blockDimM > 5` 非确定性模式: **对角策略**，将 tile 按 8×8 block 对角线分配
- 对角策略的核心目的是提升 **L2 Cache 命中率**——相邻 core 处理的 tile 在 N 方向相邻，共享 B 矩阵的 L2 cache 行

**循环结构** (Process, line 307-330):
```cpp
for (groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
    // 确定当前 group 的 tile 范围 [preCount, curCount)
    curBlock = coreIdx >= preCount ? coreIdx : coreIdx + coreNum;
    while (curBlock < curCount) {
        MNBlockIdxCompute(mnConfig, curBlock, preCount, ...);
        MMCompute(groupIdx, mnConfig);
        VectorSync(mnConfig, syncConfig);   // 非确定性: 直接 return
        VectorCompute(groupIdx, mnConfig, syncConfig);
        curBlock += coreNum;  // ★ stride = coreNum (跨 core 调度)
    }
    preCount = curCount % coreNum;  // 剩余 tile 在下一 group 补上
}
```

**特点**: 所有 core 协同处理同一 group，stride = coreNum。每个 core 在一次迭代中处理一个 tile，然后所有 core 一起推进到下一个 tile。

### 5.2 A5: ASWT (Alternating Serpentine Walk with Tail Split)

```cpp
// block_scheduler_gmm_aswt_with_tail_split.h
using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
```

**策略**:
1. **Serpentine Walk**: 将 M×N 的 tile 网格按蛇形遍历
   ```
   Row 0: N=0,1,2,3  (正序)
   Row 1: N=3,2,1,0  (逆序) ← 提升 B 矩阵 L2 cache 局部性
   Row 2: N=0,1,2,3  (正序)
   ...
   ```
2. **Tail Split**: 当尾块 tile 太大且有闲置 core 时，将尾块进一步切分给多 core 并行处理
3. **Group-Aware Partitioning**: 不同 group 的 tile 分配给不同的 core 子集

**循环结构** (KernelGmmFinalizeRoutingPertokenDequant, operator()):
```cpp
for (groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
    UpdateGroupParams(params, groupIdx);
    ProcessSingleGroup(params, bs, groupIdx);
}

// ProcessSingleGroup 内部:
while (bs.GetTileIdx(tileIdx)) {
    AIC: WaitForVector() → mmadOp_(...) → NotifyVector()
    AIV: WaitForCube() → epilogueDequantOp_(...) → NotifyCube()
}
```

**特点**: 调度器控制 tile 分配，core 之间可以处理不同 group 的不同 tile，不一定同步推进。

### 5.3 调度对比图

```
A3 调度 (协同推进, stride=coreNum):
  Core 0: [G0 T0]──[G0 T4]──[G1 T0]──[G1 T4]── ...
  Core 1: [G0 T1]──[G0 T5]──[G1 T1]──[G1 T5]── ...
  Core 2: [G0 T2]──[G0 T6]──[G1 T2]──[G1 T6]── ...
  Core 3: [G0 T3]──[G0 T7]──[G1 T3]──[G1 T7]── ...
  (所有 core 处理同一个 group 的不同 tile)

A5 调度 (ASWT, 独立推进):
  Core 0: [G0 serpentine tile 0]──[G0 serpentine tile 1]── ...
  Core 1: [G0 serpentine tile 2]──[G1 tail split tile 0]── ...
  Core 2: [G0 tail split tile 0]──[G1 serpentine tile 0]── ...
  Core 3: [G0 tail split tile 1]──[G1 serpentine tile 1]── ...
  (不同 core 可能同时处理不同 group, 通过 scheduler 动态分配)
```

---

## 6. 反量化 + Scatter-Add 实现对比

### 6.1 A3 反量化

在 `VectorCompute` 中完成（line 421-469），分为多个子步骤：

```cpp
// 1. 复制 scale 到 UB (DataCopyScale)
// 2. 复制 bias 到 UB (DataCopyBias)
// 3. 复制 per-token scale 到 UB (DataCopyPerTokenScale)
// 4. 等待 AIC Matmul 完成 (CrossCoreWaitFlag SYNC_AIC_TO_AIV)
// 5. 遍历 M 方向:
for (offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
    // a. 从 workspace 读 MM 输出到 UB (DataCopyMMOut)
    // b. 反量化 + 激活 (ComputeDequantAndActivate)
    // c. 输出 scatter (VectorAtomicProcess)
}
```

`ComputeDequantAndActivate` 内部流程:
```
vecInQueue(int32) → Cast int32→float → * scale → + bias → * pertoken_scale
→ mulsResultLocal → * logit → dequantMiddleResult
→ vecOutQueue(float)  (输出)
```

### 6.2 A5 反量化

在 `BlockEpilogueDequantFinalizeRouting::operator()` 中完成，使用 **ping-pong 双缓冲** 重叠数据搬运和计算：

```
迭代 1 (ping):
  MTE2: CopyInLogit[0..31] → logitUbPing     ← 与迭代 0 的计算重叠
  MTE2: CopyX2Scale → x2ScaleUbPing
  WaitFlag<MTE2_V> → VFDoDequant → SetFlag<V_MTE2>
  
迭代 2 (pong):
  MTE2: CopyInLogit[32..63] → logitUbPong    ← 与迭代 1 的计算重叠
  计算 logitUbPing 上的数据 (VFDoLogitMuls → VectorAtomicProcess)

输出 (ping-pong):
  V_MTE3: VFDoLogitMuls(outUbPing, logitUb) → SetFlag<V_MTE3>
  MTE3: WaitFlag<V_MTE3> → VectorAtomicProcess(outUbPing) → SetFlag<MTE3_V>
  (同时 V 在 outUbPong 上开始下一轮计算)
```

### 6.3 Scatter-Add 对比

两者的 scatter-add 机制在**语义上等价**，但实现细节不同：

**A3** (`VectorAtomicProcess`, line 388-396):
```cpp
SetAtomicAdd<float>();
DataCopyExtParams paramsOut{1, curVecBaseN * sizeof(float), 1, 1, 0};
for (uint32_t i = 0; i < curVecBaseM; i++) {
    auto outRow = static_cast<uint64_t>(
        tokenRanksGm.GetValue(mGlobalOffset + offsetM + i));    // ★ 逐行查 rowIndex
    DataCopyPad(yGm[outRow * tiling->n + yGmOffset0],
                yLocal[i * alignBaseN], paramsOut);             // ★ DataCopyPad (带 padding)
}
SetAtomicNone();
```

**A5** (`VectorAtomicProcess`, epilogue line 262-274):
```cpp
SetAtomicAdd<float>();
for (uint32_t i = 0; i < curVecBaseM; i++) {
    uint64_t destRow = rowIndexGlobal_.GetValue(offsetM_ + i);  // ★ 逐行查 rowIndex
    uint64_t destAddr = destRow * n_ + yOffset;
    DataCopyPad(yGlobal_[destAddr], yLocal[i * alignN_], paramsOut); // ★ DataCopyPad
}
SetAtomicNone();
```

**相同点**:
- 都使用 `SetAtomicAdd<float>()` 开启原子加模式
- 都逐行查询 `rowIndex`/`tokenRanks` 确定目标行
- 都使用 `DataCopyPad` 搬运整行数据
- 都因多 core 并发 atomic-add 导致浮点累加顺序不确定 → **非确定性**

**不同点**:
- A3 的 rowIndex 来自 `tokenRanksGm` 全局张量，A5 的来自 `rowIndexGlobal_`（通过 `UpdateGlobalAddr` 设置）
- A3 在 `VectorCompute` 的大循环中调用，A5 在 Epilogue 的 ping-pong 流水线中调用

---

## 7. UB Buffer 管理对比

### 7.1 A3: 手动 TQue/TBuf 分配

```cpp
// InitUbBuffer (line 179-212)
pipe->InitBuffer(scaleInQueue,       2, baseN * sizeof(float));
pipe->InitBuffer(biasInQueue,        2, baseN * sizeof(bfloat16_t));
pipe->InitBuffer(perTokenScaleInQueue, 2, ceil(baseM/2 * sizeof(float) * bufferNum, 32) * 32);
pipe->InitBuffer(vecInQueue,         2, 4096 * sizeof(int32_t));    // 32KB
pipe->InitBuffer(vecOutQueue,        2, 4096 * sizeof(float));      // 32KB
pipe->InitBuffer(tmpBuff,            ubRestBytes);                  // 99KB
// 然后手动切分 tmpBuff 为 5 个子 buffer:
dequantMiddleResult = tmpBuff.GetWithOffset<float>(ubCalSize, 0);
pertokenBrcbLocal  = tmpBuff.GetWithOffset<float>(ubCalSize, ubCalSizeFloat);
mulsResultLocal    = tmpBuff.GetWithOffset<float>(ubCalSize, 2 * ubCalSizeFloat);
biasCalcLocal      = tmpBuff.GetWithOffset<float>(ubCalSize, 3 * ubCalSizeFloat);
sharedTmpLocal     = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 4 * ubCalSizeFloat);
```

**UB 布局 (192KB, 910B)**:

| 区域 | 大小 | 用途 |
|------|------|------|
| scaleInQueue ×2 | 2KB | scale ping-pong |
| biasInQueue ×2 | 1KB | bias ping-pong |
| perTokenScaleInQueue ×2 | ~1KB | per-token scale |
| vecInQueue ×2 | 32KB | MM 输出 int32 (从 L2 workspace) |
| vecOutQueue ×2 | 32KB | 反量化 float 输出 |
| tmpBuff | 99KB | dequant/bias/muls/pertoken 临时缓冲 |
| **合计** | ~190KB | (192KB UB, 余 2KB) |

### 7.2 A5: Cgmct 框架自动管理

```cpp
// BlockEpilogueDequantFinalizeRouting::Init (epilogue line 180-211)
// 框架自动计算偏移并创建 LocalTensor
l0cOutUb_       (VECIN, offset 0,     MAX_SINGLE_MNS * sizeof(DataTypeIn))
l0cOutUbFloat_  (VECIN, offset 0,     同名 aliasing)
logitUbPing_    (VECIN, +l0cOut,      256B)
logitUbPong_    (VECIN, +logitPing,   256B)
x2ScaleUbPing_  (VECIN, +logitPong,   256B)
x2ScaleUbPong_  (VECIN, +x2ScalePing, 256B)
x1ScaleUbPing_  (VECIN, +x2ScalePong, 256B)
x1ScaleUbPong_  (VECIN, +x1ScalePing, 256B)
biasUbPing_     (VECIN, +x1ScalePong, 256B)
biasUbPong_     (VECIN, +biasPing,    256B)
outUbPing_      (VECIN, +biasPong,    HALF_DB_MAX_SINGLE_MNS * sizeof(DataTypeOut))
outUbPong_      (VECIN, +outPing,     同上)
```

**UB 布局 (256KB, 950)**:

| 区域 | 大小 | 用途 |
|------|------|------|
| l0cOutUb (L0C 映射) | ~128KB | MMAD int32 输出 (L0C→UB) |
| logit ping-pong | 512B | logit 双缓冲 |
| x2Scale ping-pong | 512B | per-channel scale 双缓冲 |
| x1Scale ping-pong | 512B | per-token scale 双缓冲 |
| bias ping-pong | 512B | bias 双缓冲 |
| outUb ping-pong | ~64KB | 反量化 float 输出双缓冲 |
| **合计** | ~196KB | (256KB UB, 余 ~60KB) |

### 7.3 关键区别

| 维度 | A3 | A5 |
|------|----|----|
| 管理方式 | 手动 InitBuffer + GetWithOffset | 框架自动顺序分配 |
| L0C→UB | 无 (经 L2 workspace 中转) | `l0cOutUb_` 直接映射 |
| 双缓冲粒度 | TQue 级别 (vecInQueue/vecOutQueue) | Ping-pong 级别 (logit/scale/bias/output) |
| 反量化中间缓冲 | tmpBuff 手动切分 5 区 | l0cOutUbFloat 原地 aliasing |
| 可维护性 | 偏移计算需手动保证不越界 | 框架保证不越界 |

---

## 8. AIC↔AIV 同步机制对比

### 8.1 A3 同步

```cpp
// 常量定义 (grouped_matmul_finalize_routing_utils.h:28-29)
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;

// MMCompute (AIC):
if (cubeCount >= parallNum) {
    CrossCoreWaitFlag(SYNC_AIV_TO_AIC);  // 等待 AIV 完成上一 tile
}
// ... Matmul ...
CrossCoreSetFlag(SYNC_AIC_TO_AIV);       // 通知 AIV 当前 tile 就绪

// VectorCompute (AIV):
CrossCoreWaitFlag(SYNC_AIC_TO_AIV);      // 等待 AIC MM 完成
// ... Dequant + Scatter ...
// (不主动通知 AIC, 由下一轮 MMCompute 等待 SYNC_AIV_TO_AIC)
```

A3 的同步是**隐式的流水线**: AIC 在下一次 `MMCompute` 入口处等待 AIV，AIV 在 `VectorCompute` 入口处等待 AIC。

额外还有 `SyncAll()` (全局 barrier):
- `PreProcess()` 后: 确保 sharedInput 初始化完成
- `FRDeterministic()` 内: 确定性刷出的多 core 协调（非确定性下不调用）

### 8.2 A5 同步

```cpp
// kernel_gmm_finalize_routing_pertoken_dequant.h:41-61
AIC_SYNC_AIV_FLAGS = 4;  // AIV→AIC
AIV_SYNC_AIC_FLAGS = 6;  // AIC→AIV

// ProcessSingleGroup 内 (每 tile):
// AIC:
WaitForVector();   // 等待 AIV (flag 6+16 on PIPE_FIX)
mmadOp_(...);      // MMAD 计算
NotifyVector();    // 通知 AIV (flags 4,20 on PIPE_FIX)

// AIV:
WaitForCube();     // 等待 AIC (flag 4 on PIPE_V)
epilogueOp_(...);  // 反量化 + scatter
NotifyCube();      // 通知 AIC (flag 6 on PIPE_V)
```

A5 的同步是**显式的握手协议**，每 tile 一次完整的 AIC→AIV→AIC 双向通知，通过 `NotifyVector/WaitForVector/NotifyCube/WaitForCube` 四个命名方法明确表达。

### 8.3 同步对比

| 维度 | A3 | A5 |
|------|----|----|
| 风格 | 隐式流水线 | 显式握手协议 |
| flag ID | 3, 5 | 4, 6 (+16 偏移防止冲突) |
| 封装 | 直接调用 CrossCoreSetFlag/WaitFlag | NotifyVector/WaitForCube 等命名方法 |
| PIPE | AIC=PIPE_FIX, AIV=PIPE_V | AIC=PIPE_FIX, AIV=PIPE_V |
| SyncAll | PreProcess 后 + FRDeterministic | Prologue 后一次 |
| isVecSetSyncCom | 标记首个 tile 是否已发通知 | 内置在 WaitForVector 逻辑中 |

---

## 9. 硬件适配差异

### 9.1 硬件规格

| 硬件 | A3 (910B) | A5 (950) |
|------|----------|---------|
| UB | 192 KB | **256 KB** (+33%) |
| L1 | 512 KB | 512 KB |
| L0A | 64 KB | 64 KB |
| L0B | 64 KB | 64 KB |
| L0C | 128 KB | **256 KB** (+100%) |
| L2 | 芯片级共享 | 芯片级共享 |
| Cube 数量 | 32 | 更多 (架构不同) |
| 指令集 | AscendC 220 | AscendC 310 |

### 9.2 L0C 容量差异带来的影响

A5 的 L0C 翻倍 (128KB→256KB) 是最大的硬件变化，直接影响：

1. **Tile 尺寸**: A3 固定 baseM=128, baseN=256 → A5 可以 baseM=256, baseN=256 (更大的 tile 减少同步次数)
2. **数据通路**: A5 可以直接 L0C→UB (256KB 够存一个完整 tile 的 int32 输出)，A3 必须经过 L2 workspace 中转
3. **同步频率**: A5 tile 更大 → 每 group 更少的 tile → 更少的 AIC↔AIV 同步次数

### 9.3 UB 容量差异带来的影响

A5 的 UB 增大 33% (192KB→256KB) 允许：

1. **更大的反量化缓冲区**: A5 的 `outUb` ping-pong buffer 有 ~64KB，A3 的 vecOutQueue 只有 32KB
2. **更多的 ping-pong 通道**: A5 对 logit/scale/bias/output 分别做双缓冲，A3 只在 TQue 层面做双缓冲
3. **反量化中间结果原地存放**: A5 的 `l0cOutUbFloat_` 与 `l0cOutUb_` 共用同一块 UB (aliasing)，节省内存

### 9.4 硬件差异导致的架构选择

```
A3 (910B, L0C=128KB):
  L0C 太小无法存放完整 tile → 必须通过 L2 workspace 中转
  → L2 workspace 分区管理 → 需要手动 workSpaceOffset 计算
  → 确定性窗口自然附着在 L2 workspace 上 (复用同一块 buffer)

A5 (950, L0C=256KB):
  L0C 足够大 → L0C→UB 直通，不需要 L2 中转
  → 零 L2 带宽占用 → 更高吞吐
  → 不需要确定性窗口 (新芯片, 新架构选择)
```

---

## 10. 具体 Shape 示例: 4 experts × 64 tokens × k=2048 × n=7168

### 10.1 A3 非确定性执行流程

**Tiling 参数** (来自 `base_tiling.cpp`):

```
baseM=128, baseN=256, baseK=128
mSize=256 (64 tokens × 4 experts), nSize=7168, kSize=2048
coreNum=假设 8

groupTokens: [64, 64, 64, 64]  (每个 expert 64 tokens)
```

**Group 0 (expert 0, m=64)**:
```
blockDimM = ceil(64, 128) = 1
blockDimN = ceil(7168, 256) = 28
curCount = 1 × 28 = 28
thresholdMDimN = 8 × 28 = 224

因为 blockDimM(1) <= thresholdDimM(5) → 行优先策略
  Core 0: tile(0, 0), tile(0, 8),  tile(0, 16), tile(0, 24)
  Core 1: tile(0, 1), tile(0, 9),  tile(0, 17), tile(0, 25)
  Core 2: tile(0, 2), tile(0, 10), tile(0, 18), tile(0, 26)
  Core 3: tile(0, 3), tile(0, 11), tile(0, 19), tile(0, 27)
  Core 4: tile(0, 4), tile(0, 12), tile(0, 20), ...
  Core 5: tile(0, 5), tile(0, 13), tile(0, 21), (idle)
  Core 6: tile(0, 6), tile(0, 14), tile(0, 22), (idle)
  Core 7: tile(0, 7), tile(0, 15), tile(0, 23), (idle)

每个 core 处理 3-4 个 tile
每次 tile: MMCompute → VectorSync(nop) → VectorCompute
  VectorCompute: 读 workspace → dequant → logit → scatter-add
preCount = 28 % 8 = 4
```

**Group 1 (expert 1, m=64)**:
```
blockDimM = 1, blockDimN = 28
curCount = 28 + 28 = 56
Core 0: 从 tile(idx=28+0=28) 开始 → tile(0,0) of group 1
  ...
```

**L2 Workspace 占用** (非确定性):
```
workSpaceOffset = singleN × singleM × (coreIdx + cubeCount % parallNum × coreNum)
每个 core 的 workspace 分区: baseM × baseN × 4B = 128 × 256 × 4 = 128KB
总 workspace = 128KB × 8 cores × 4 (parallNum) = 4MB
(远小于确定性模式的 64MB/96MB)
```

### 10.2 A5 非确定性执行流程

**Tiling 参数** (由 `GroupedQmmTiling` 自动计算):

```
L0C=256KB 允许更大的 baseM/baseN
假设 baseM=128, baseN=256 (或更大如 256×256)

mSize=256, nSize=7168, kSize=2048
groupTokens: [64, 64, 64, 64]
```

**Prologue 阶段**:
```
假设 sharedInput: batch=256, sharedInputOffset=0, sharedInputLen=0
→ InitOutputWithZeros(0, 7168 × 256)  // 全零初始化
  8 个 AIV core 协作，每个处理 7168×256/8 = 229,376 个 float
```

**Group 0 (expert 0, m=64, n=7168, k=2048)**:
```
ASWT Scheduler:
  mCnt = ceil(64, 128) = 1
  nCnt = ceil(7168, 256) = 28
  totalCnt = 28
  serpentine: 1 row → 28 tile, N=0→27 (正序)

  blockNum=8, startBlockIdx=0, endBlockIdx=27%8=3
  round = ceil(28/8) = 4

  Core 0: round 0 tile 0, round 1 tile 8, round 2 tile 16, round 3 tile 24
  Core 1: round 0 tile 1, round 1 tile 9, round 2 tile 17, round 3 tile 25
  ...
  Core 4: round 0 tile 4, round 1 tile 12, round 2 tile 20, round 3 tile 28→idle(round--)
  ...

每 tile:
  AIC: WaitForVector → mmadOp_(a, b, l0cOutUb_, shape) → NotifyVector
       MMAD 输出直接写入 L0C, L0C→UB 直接可取
  AIV: WaitForCube → epilogueDequantOp_:
       1. CopyInLogit (singleM=128 floats, 512B)
       2. CopyX2Scale (singleN=256 floats, 1024B)
       3. CopyX1Scale (singleM=128 floats, 512B)
       4. CopyBias (singleN=256 bf16, 512B)
       5. VFDoDequant: int32[128][256] → float[128][256]
          (l0cOutUb * x2Scale_per_channel * x1Scale_per_token + bias)
       6. VFDoLogitMuls: float[128][256] *= logit[128] (每行广播乘)
       7. VectorAtomicProcess:
          SetAtomicAdd<float>
          for i in 0..128:
            destRow = rowIndex[i]
            DataCopyPad(y[destRow*n..], outUb[i*alignN..])
          SetAtomicNone
```

### 10.3 这个 Shape 下的关键差异

| 阶段 | A3 | A5 |
|------|----|----|
| **MM 输出写** | Cube→L2 workspace (~128KB/core/tile) | Cube→L0C (128×256×4=128KB) |
| **MM 输出读** | L2 workspace→UB vecInQueue (AIV 读回) | L0C→UB l0cOutUb (零拷贝映射) |
| **Dequant 缓冲** | vecInQueue→tmpBuff→vecOutQueue (3 次 UB 内拷贝) | l0cOutUb→原地 dequant→outUb (1 次写入) |
| **Scatter** | 逐行 tokenRanksGm.GetValue + DataCopyPad | 逐行 rowIndexGlobal_.GetValue + DataCopyPad |
| **Ping-pong 重叠** | TQue 级别 (vecIn/vecOut enqueue/dequeue) | 4 路 ping-pong (logit/scale/bias/output) |
| **每 tile 同步** | 1 次 CrossCoreSet/Wait | 2 次 (AIC→AIV→AIC) 完整握手 |
| **L2 流量** | 每 tile: 128KB write + 128KB read | **0** |

对于这个 shape (28 tiles × 4 groups = 112 tiles)，A3 有 112 × 256KB = **28MB L2 读写往返**，A5 完全没有。

---

## 11. 总结对比表

### 11.1 维度对比总表

| 维度 | A3 非确定性 | A5 非确定性 |
|------|-----------|-----------|
| **框架风格** | 手写单体类 (~690 行) | Cgmct Block 组合 (~60 行组装) |
| **可扩展性** | 修改需改动核心循环 | 替换 Block/调度器即可 |
| **MM 输出路径** | Cube→L2→UB (L2 往返) | Cube→L0C→UB (直通) |
| **L2 带宽** | 每 tile 写+读 | 零 |
| **调度算法** | 对角/行优先 + stride=coreNum | ASWT 蛇形 + Tail Split |
| **Group 处理** | 所有 core 协同处理同一 group | Core 可独立处理不同 group |
| **Tile 形状** | 固定 128×256 (特例互换) | 自动根据硬件计算 |
| **UB 管理** | TQue+TBuf 手动偏移 | 框架自动顺序分配 |
| **Ping-pong** | 2 路 (TQue enqueue/dequeue) | 4 路 (logit/scale/bias/output) |
| **AIC↔AIV 同步** | 隐式流水线 (flag 3,5) | 显式握手 (flag 4,6) |
| **UB 预算** | ~190KB/192KB (99%) | ~196KB/256KB (77%) |
| **L0C 利用** | 间接 (经 L2) | 直接 (128×256×4=128KB per tile) |
| **Atomic Scatter** | 逐行 tokenRanksGm + DataCopyPad | 逐行 rowIndexGlobal_ + DataCopyPad |
| **非确定性来源** | 多 core 同时 atomic-add 到同一 y 行 | 同左 |
| **TilingKey** | 手动位编码 UL | 模板 GET_TPL_TILING_KEY |
| **Tiling 数据** | 20 字段 (含确定性字段) | 2 字段 (纯计算参数) |

### 11.2 性能影响

| 操作 | A3 开销 | A5 开销 |
|------|---------|---------|
| MM→Dequant 延迟 | L2 write + L2 read (~数百 cycle) | L0C→UB copy (~数十 cycle) |
| 每 tile L2 带宽 | 2×tile_size | 0 |
| 同步次数/tile | 1 次 AIC↔AIV | 2 次 (完整握手) |
| UB 内数据搬移 | vecIn→tmpBuff→vecOut (3 次) | l0cOut→outUb (1 次) |

### 11.3 设计哲学

- **A3**: 以**兼容性和可控性**为优先——手写代码保证与 A2 的兼容，L2 workspace 模式自然支持确定性窗口扩展，每个细节可控。
- **A5**: 以**性能和模块化**为优先——Cgmct 框架解耦各阶段，ASWT 调度最大化 L2 cache 利用率，L0C 直通消除中间访存，零 L2 带宽占用适合更大 batch。

---

## 12. 代码路径索引

### A3 非确定性路径

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_kernel/grouped_matmul_finalize_routing.h` | 290-335 | `Process()`: 主循环, `deterministicFlag=0` 跳过后处理 |
| `op_kernel/grouped_matmul_finalize_routing.h` | 338-374 | `MMCompute()`: Cube → L2 workspace |
| `op_kernel/grouped_matmul_finalize_routing.h` | 622-650 | `VectorSync()`: **line 627-628 直接 return** |
| `op_kernel/grouped_matmul_finalize_routing.h` | 421-469 | `VectorCompute()`: L2→UB→反量化→scatter |
| `op_kernel/grouped_matmul_finalize_routing.h` | 377-402 | `VectorAtomicProcess()`: line 380-396 non-det 路径 |
| `op_kernel/grouped_matmul_finalize_routing.h` | 179-212 | `InitUbBuffer()`: UB 手动分配 |
| `op_kernel/grouped_matmul_finalize_routing_utils.h` | 33-56 | MNConfig, SyncConfig |
| `op_kernel/grouped_matmul_finalize_routing_utils.h` | 140-161 | `MNBlockIdxCompute()`: 对角/行优先调度 |
| `op_host/grouped_matmul_finalize_routing_base_tiling.cpp` | 350-399 | `W8A8TilingProcess()`: baseM=128, baseN=256 |
| `op_host/grouped_matmul_finalize_routing_base_tiling.cpp` | 413-425 | `DeterministicTilingProcess()`: flag=0 时直接 return |

### A5 非确定性路径

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 28-91 | W8A8 kernel 组装入口 |
| `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` | 332-363 | `operator()`: 主循环 |
| `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` | 285-330 | `ProcessSingleGroup()`: tile 循环 + AIC↔AIV sync |
| `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` | 234-264 | `UpdateOffset()`: 各 buffer 偏移推进 |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | 504-558 | `operator()`: 反量化 + logit + scatter |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | 304-330 | `VFDoDequantWithX1X2Scale()`: 反量化调度 |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | 399-469 | `VFDoDequant<isBias>`: 完整 dequant 实现 |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | 262-274 | `VectorAtomicProcess()`: atomic scatter |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | 276-302 | `VFDoLogitMuls()`: logit 广播乘 |
| `gmm/common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h` | 178-212 | `GetTileIdx()`: ASWT 调度算法 |
| `gmm/common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h` | 214-241 | `GetBlockShape()`: tail split 形状计算 |
| `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h` | 256-295 | `operator()`: zero init + sharedInput |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 445-459 | `DoOpTiling()`: GMM 参数填充 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 468-509 | `DoLibApiTiling()`: Matmul Tiling 填充 |
