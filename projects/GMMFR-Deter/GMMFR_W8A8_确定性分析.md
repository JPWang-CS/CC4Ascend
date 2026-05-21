# GMMFR (GroupedMatmulFinalizeRouting) W8A8 INT8 确定性实现分析

> 分析目标: A3 (Ascend 910B/910C) 平台上的 W8A8 (INT8 activation × INT8 weight) 模式
> 源码位置: `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/`
>
> **分析范围**: 主 kernel 路径 (`op_kernel/grouped_matmul_finalize_routing.h`)，即通过 `GroupedMatmulFinalizeRoutingBaseTiling` (registerList={0}) 的 A2/A3 兼容路径。不涉及 arch35/Cgmct 框架。

---

## 目录
1. [算子总体架构](#1-算子总体架构)
2. [W8A8 INT8 数据类型与 Tiling 策略](#2-w8a8-int8-数据类型与-tiling-策略)
3. [确定性窗口机制详解](#3-确定性窗口机制详解)
4. [Shape 相关的分块策略分析](#4-shape-相关的分块策略分析)
5. [完整数据流与确定性保证](#5-完整数据流与确定性保证)
6. [溢出防护机制](#6-溢出防护机制)
7. [不同 Shape 下的窗口行为示例](#7-不同-shape-下的窗口行为示例)
8. [硬件 Buffer 与 L2/Workspace 分配分析](#8-硬件-buffer-与-l2workspace-分配分析)

---

## 1. 算子总体架构

### 1.1 功能定义

GMMFR 是 **GroupedMatmul + MoeFinalizeRouting** 的融合算子。它完成：
- 多个专家 (expert) 的权重矩阵与输入 token 的矩阵乘法 (GroupedMatmul)
- 对 Matmul 输出进行 per-channel 反量化 + per-token 反量化
- 依据 `rowIndex` 对各个专家的输出做 **scatter-add** (即 FinalizeRouting / Combine)

### 1.2 执行模型

算子采用 **AIC + AIV 混合编程模型** (`KERNEL_TYPE_MIX_AIC_1_2`):

```
┌─────────────────────────────────────────────────┐
│                  AIC (Cube 核)                    │
│  ┌───────────────────────────────────────────┐   │
│  │  MMCompute(): 对每个 (group, M-block,     │   │
│  │  N-block) 执行 INT8 Matmul               │   │
│  │  -> 输出 int32 到 workspace              │   │
│  └───────────────────────────────────────────┘   │
│                       │                          │
│                       │ CrossCoreSetFlag         │
│                       ▼                          │
│                  AIV (Vector 核)                  │
│  ┌───────────────────────────────────────────┐   │
│  │  1. PreProcess(): sharedInput 初始化     │   │
│  │  2. VectorCompute(): 反量化 + 激活       │   │
│  │  3. VectorAtomicProcess(): scatter-add   │   │
│  │     输出到 y tensor                       │   │
│  │  4. FRDeterministic(): 确定性刷出        │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 1.3 Tiling 分发路径

A3 芯片 (DAV_3510) 的 W8A8 在 tiling 层通过 `GroupedMatmulFinalizeRoutingBaseTiling` 计算 tiling 参数，然后分发到对应 kernel。本章聚焦于主 kernel (`grouped_matmul_finalize_routing.h`) 中的核心实现逻辑——确定性窗口机制的分析以该文件为准，其代码最直接地展示了窗口调度、同步和刷出的完整流程。

### 1.4 Tiling Key 分发机制

W8A8 使用 TilingKey `10000000000000000001UL` 作为基础 key，通过修改十进制位来编码不同配置：

| Key 模式 | rowIndex 类型 | scale 类型 | groupListType | sharedInput |
|----------|-------------|-----------|---------------|-------------|
| `...001UL` | int64 | float | count | 非空 |
| `...011UL` | int32 | float | count | 非空 |
| `...101UL` | int64 | bf16 | count | 非空 |
| `...111UL` | int32 | bf16 | count | 非空 |
| `...1001UL` | int64 | float | cumsum | 非空 |
| `...10001UL` | int64 | float | count | 空 |
| `...11001UL` | int64 | float | cumsum | 空 |

这些 key 在编译期通过模板参数确定具体类型组合，避免了运行时分支。

```cpp
// op_kernel/grouped_matmul_finalize_routing.cpp:121-122
if (TILING_KEY_IS(10000000000000000001UL)) {
    GMMFR_A8W8_IMPL(int64_t, float, false, false);  // 大部分 W8A8 场景
}
```

---

## 2. W8A8 INT8 数据类型与 Tiling 策略

### 2.1 数据类型定义

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:28-33
using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;     // x (activation)
using bT = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;     // weight (NZ fractal)
using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;    // matmul 输出
```

关键设计:
- **Activation (x)**: `ND` 格式 `int8`，shape 为 `(m, k)`
- **Weight**: `NZ` 格式 (昇腾亲和数据排布) `int8`，shape 为 `(e, k, n)`，每个 expert 有独立的 k×n 矩阵
- **Matmul 中间结果**: `int32`，存于 workspace，累积精度不丢失
- **最终输出**: `float32` (dtype=0)

### 2.2 W8A8 Tiling 参数计算

```cpp
// op_host/grouped_matmul_finalize_routing_base_tiling.cpp:350-399
ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::W8A8TilingProcess()
{
    size_t userWorkspaceSize = CV_PARALL_NUM * 256 * 128 * sizeof(int32_t) * blockDim_;
    size_t systemWorkspaceSize = RPC_WORKSIZE * MB_SIZE;
    ubRestBytes_ = A8W8_UBRESTBYTES;           // 101376 bytes

    uint32_t baseM = BEST_BASE_M;               // 128
    uint32_t baseN = BEST_BASE_N;               // 256

    // 特殊 shape 优化: n=7168/7680, k=2048 时根据 avg_m 调优
    if ((n_ == 7168 || n_ == 7680) && k_ == 2048) {
        uint32_t avg_m = (tuningConfig_ != 0) ? tuningConfig_
                       : ((groupNum_ != 0) ? (m_ / groupNum_) : 1);
        // avg_m 在 (128, 256] 时交换 baseM/baseN
        baseM = (avg_m > AVG_M_THREHOLD && avg_m <= AVG_M_BIG_THREHOLD) ?
                BEST_BASE_N : BEST_BASE_M;     // baseM=256 or 128
        baseN = (avg_m > AVG_M_THREHOLD && avg_m <= AVG_M_BIG_THREHOLD) ?
                BEST_BASE_M : BEST_BASE_N;     // baseN=128 or 256
    }

    vBaseM_ = UBCALSIZE / baseN;                // 4096 / 256 = 16 (或 4096 / 128 = 32)
    mm_.SetAType(GM, ND, DT_INT8, false);
    mm_.SetBType(GM, NZ, DT_INT8, false);
    mm_.SetCType(GM, ND, DT_INT32);
    mm_.SetFixSplit(baseM, baseN, BEST_BASE_K); // baseK=128
}
```

**W8A8 Tiling 关键常量:**

| 常量 | 值 | 含义 |
|------|-----|------|
| `BEST_BASE_M` | 128 | Cube 单次 M 维分块 |
| `BEST_BASE_N` | 256 | Cube 单次 N 维分块 |
| `BEST_BASE_K` | 128 | Cube 单次 K 维分块 |
| `UB_CAL_SIZE` | 4096 (16×256) | Vector 计算 buffer 大小 (元素数) |
| `A8W8_UBRESTBYTES` | 101376 | 除 TQue 外的临时 buffer |
| `CV_PARALL_NUM` | 4 | Cube-Vector 并行流水深度 |
| `DETER_UB_SIZE` | 12288 (12KB) | 确定性输出时 QueBind buffer 大小 |

### 2.3 Workspace 布局

```
workspace 总体布局:
┌──────────────────────────────────────────────────────┐
│  Cube Matmul 输出 (int32)                             │
│  大小: CV_PARALL_NUM × baseM × baseN × sizeof(int32) │
│       × blockDim                                      │
│  = 4 × 128 × 256 × 4 × blockDim                       │
│  = 512KB × blockDim                                   │
├──────────────────────────────────────────────────────┤
│  系统保留                                             │
│  大小: RPC_WORKSIZE × MB (20MB)                       │
├──────────────────────────────────────────────────────┤
│  [确定性模式] 确定性 Workspace                         │
│  大小: 96MB (L2 > 96M) 或 64MB (L2 <= 96M)           │
│  存放 float 中间结果，用于延迟 combine                │
└──────────────────────────────────────────────────────┘
```

---

## 3. 确定性窗口机制详解

### 3.1 为什么需要确定性

在非确定性模式下，GMMFR 使用 **atomic add** 直接将 Matmul+反量化的部分结果 scatter-add 到输出：

```cpp
// 非确定性路径: grouped_matmul_finalize_routing.h:387-396
SetAtomicAdd<float>();
for (uint32_t i = 0; i < vecAParams.curVecBaseM; i++) {
    auto outRow = tokenRanksGm.GetValue(...);
    DataCopyPad(yGm[outRow * n + offset], yLocal[i * alignBaseN], paramsOut);
}
SetAtomicNone();
```

问题: 不同 Core 或同一 Core 的不同 group 处理的不同 token 可能 route 到**同一个输出行**。Atomic add 的完成顺序取决于硬件调度，导致浮点累加顺序不确定 → 结果不可复现。

### 3.2 确定性方案: 滑动窗口 + 延迟 Combine

确定性模式下，算子不在每个 group 的计算完成后立刻 scatter-add 到最终输出，而是:
1. 将部分结果写入确定性 workspace 的一个"窗口"区域
2. 当窗口填满时，按 rowIndex 顺序地 scatter-add 刷出到最终输出
3. 窗口向前滑动，覆盖下一批 M 行

### 3.3 SyncConfig: 窗口状态结构

```cpp
// op_kernel/grouped_matmul_finalize_routing_utils.h:49-56
struct SyncConfig {
    uint64_t curM = 0;        // 窗口中已有有效数据的 M 行数
    uint64_t curGroup = 0;    // 当前滑动到的 group 索引
    uint64_t curGroupM = 0;   // curGroup 中已处理到的行位置 (全局偏移)
    uint64_t lowBoundM = 0;   // 窗口下界 (需要刷出到此位置)
    uint64_t windowSize = 0;  // 窗口最大容纳的行数
    uint64_t baseN = 0;       // 确定性刷出时的 N 维分块大小
};
```

### 3.4 窗口初始化

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:302-306
syncConfig.windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(DTYPE_OUT));
syncConfig.lowBoundM = syncConfig.windowSize;

// baseN: 确定性输出时沿 N 维的分块粒
uint64_t nTimes = Ceil(tiling->n, DETER_UB_SIZE / sizeof(DTYPE_OUT));
syncConfig.baseN = Ceil(Ceil(tiling->n, nTimes), 128) * 128;
```

**windowSize 计算:**

```
windowSize = deterWorkspaceSize / (n * sizeof(float))

例如 n=7168, deterWorkspaceSize=96MB:
  windowSize = 96 × 1024 × 1024 / (7168 × 4)
             = 100663296 / 28672
             ≈ 3510 行
```

**deterWorkspaceSize 的确定 (Host 侧):**

```cpp
// op_host/grouped_matmul_finalize_routing_base_tiling.cpp:413-425
void DeterministicTilingProcess()
{
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;
        return;
    }
    deterministicFlag_ = 1;
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
    deterWorkspaceSize_ = (l2_size > DETER_WORK_SPACE_SIZE) ?
                          DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
    // 96MB (L2 > 96MB) 或 64MB (L2 <= 96MB)
    workspaceSize_ += deterWorkspaceSize_;
}
```

### 3.5 确定性 Workspace 的地址计算 (Kernel 侧)

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:162-166
if (tiling->deterministicFlag == 1) {
    mmQuantOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(
        initParams.workspace +
        tiling->parallNum * tiling->matmulTiling.baseM *
        tiling->matmulTiling.baseN * sizeof(int32_t) *
        tiling->coreNum));
}
```

确定性 workspace 紧接在 Cube Matmul workspace 之后:

```
workspace:
[0 .. cubeWorkspaceSize-1]:            Cube Matmul int32 中间结果
[cubeWorkspaceSize .. +deterWorkspaceSize-1]:  确定性窗口 (float)
```

### 3.6 VectorSync: 窗口滑动检查

这是确定性机制的核心调度逻辑。在每个 (group, M-block, N-block) 的 MMCompute + VectorCompute 完成后调用:

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:622-649
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorSync(
    MNConfig& mnConfig, SyncConfig& syncConfig)
{
    if (tiling->deterministicFlag == 0) return;  // 非确定性模式直接跳过

    // 当累计的 MM 输出行数超过窗口下界时，触发刷出
    while (mnConfig.curBlockM > syncConfig.lowBoundM) {

        // Step 1: 推进 curGroup 和 curM 到 lowBoundM 位置
        while (syncConfig.curGroup < tiling->groupNum) {
            uint32_t mi = groupTokensGm.GetValue(syncConfig.curGroup);
            if (syncConfig.curGroupM + mi <= syncConfig.lowBoundM) {
                // 当前 group 的所有 token 都在窗口内
                syncConfig.curGroupM += mi;
                syncConfig.curM = syncConfig.curGroupM;
                syncConfig.curGroup++;
            } else {
                // 当前 group 跨窗口边界，只推进到 lowBoundM
                syncConfig.curM += (syncConfig.lowBoundM - syncConfig.curM)
                                   / mnConfig.singleM * mnConfig.singleM;
                break;
            }
        }

        // Step 2: 执行确定性刷出
        FRDeterministic(syncConfig);

        // Step 3: 窗口向前滑动
        syncConfig.lowBoundM = syncConfig.curM + syncConfig.windowSize;
    }
}
```

**关键逻辑图解:**

```
时间线 →

窗口 1:  [lowBoundM - windowSize  ........  lowBoundM]
         已刷出区域                        待刷出边界
                                                  ↓
         group0(tokens 0-6) group1(tokens 7-70)   group2...
                                                  ↑
                                            lowBoundM 到达这里时触发 FRDeterministic

窗口 2:                                  [lowBoundM - windowSize ........ lowBoundM]
                                         上次的 lowBoundM 成为新的窗口下界
```

### 3.7 FRDeterministic: 确定性刷出

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:652-687
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::FRDeterministic(
    SyncConfig& syncConfig)
{
    SyncAll();  // 等待所有 AIC/AIV 完成当前工作

    // 本次需要刷出的 M 范围
    uint64_t totalM = syncConfig.curM -
                      (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = tiling->coreNum * GetTaskRation();
    uint64_t n = tiling->n;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        // 查找该 M 行的目标输出行
        auto outRow = tokenRanksGm.GetValue(
            (syncConfig.lowBoundM - syncConfig.windowSize) + mOffset);

        // 多核分工: 每个 Vector 核只处理自己负责的输出行
        if (outRow % coreNumVec != GetBlockIdx()) continue;

        // 沿 N 维分块搬运并 scatter-add
        uint64_t curVecBaseN = syncConfig.baseN;
        for (uint64_t nOffset = 0; nOffset < n;
             nOffset += syncConfig.baseN) {
            if (nOffset + syncConfig.baseN >= n) {
                curVecBaseN = n - nOffset;
            }

            // 从确定性 workspace 读取该行的部分结果
            LocalTensor<DTYPE_OUT> bindLocal =
                queBind.AllocTensor<DTYPE_OUT>();
            DataCopyPad2D(bindLocal,
                mmQuantOutGm[mOffset * n + nOffset], copyDimParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // Atomic add 到最终输出
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyPad(yGm[outRow * n + nOffset], bindLocal, paramsOut);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();
}
```

**为什么确定性模式下 Atomic Add 仍然是安全的:**

在确定性模式下，尽管仍然使用 atomic add，但关键区别在于:
1. **刷出时机确定**: FRDeterministic 只在 `VectorSync` 中调用，且调用前执行了 `SyncAll()`
2. **输出行不重叠**: 每个 Vector 核处理 `outRow % coreNumVec == GetBlockIdx()` 的行，**同一窗口内不同核处理不同的输出行** — 不同的核不会对同一输出行做 atomic add
3. **窗口间隔离**: 同一输出行在窗口 1 被刷出后，窗口 2 中不会再次出现 (因为每个 token 只属于一个 group)

**确定性保证的核心**: 浮点累加的顺序在每次运行时完全一致，因为:
- 每个窗口内的刷出顺序由 rowIndex 的遍历顺序决定 (mOffset 递增)
- 同一输出行的不同部分结果不会跨窗口并发写入
- SyncAll() 确保所有核完成后再进入下一窗口

### 3.8 确定性 Workspace 写入路径

在 VectorAtomicProcess 中，确定性模式下不是 scatter-add 到 y，而是写入 mmQuantOutGm:

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:377-402
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorAtomicProcess(
    const VectorAtomicParams& vecAParams, const SyncConfig& syncConfig)
{
    LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
    if constexpr (P::combine) {
        if (tiling->deterministicFlag == 1) {
            // ★ 确定性路径: 写入确定性 workspace, 而非直接 scatter-add
            DataCopy2DDimParams dimParams{
                vecAParams.curVecBaseM, vecAParams.curVecBaseN,
                vecAParams.alignBaseN};
            DataCopyPad2D(
                mmQuantOutGm[vecAParams.yGmOffset1 -
                    (syncConfig.lowBoundM - syncConfig.windowSize) * n],
                yLocal, dimParams, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
            return;
        }
        // 非确定性路径: 直接 atomic scatter-add
        SetAtomicAdd<float>();
        for (...) {
            DataCopyPad(yGm[outRow * n + offset], ...);
        }
        SetAtomicNone();
    }
}
```

写入地址 = `mmQuantOutGm[全局M偏移 - 窗口起始M * n]`
这确保了写入落在窗口区域内的正确偏移位置。

### 3.9 最终刷出

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:331-334
// Process() 函数末尾
if (tiling->deterministicFlag == 1) {
    syncConfig.curM = mnConfig.offsetM;   // 总 M 行数
    FRDeterministic(syncConfig);           // 刷出最终剩余窗口
}
```

所有 group 处理完毕后，剩余未刷出的窗口数据通过最后一次 `FRDeterministic` 写入最终输出。

### 3.10 确定性模式的额外代价

1. **内存代价**: 额外 64MB~96MB workspace
2. **延迟代价**: 输出不能流式写入，必须等待窗口填满才刷出
3. **带宽代价**: 多了一次确定性 workspace 的写入 + 读取
4. **UB 代价**: AIV 核多分配 12KB (DETER_UB_SIZE) 给 queBind

---

## 4. Shape 相关的分块策略分析

### 4.1 MNBlockIdxCompute: 分块索引计算

```cpp
// op_kernel/grouped_matmul_finalize_routing_utils.h:140-161
__aicore__ inline void MNBlockIdxCompute(
    MNConfig& mnConfig, const uint32_t curBlock,
    const uint32_t count, const uint32_t thresholdMDimN,
    const uint32_t deterministicFlag)
{
    // 小 M 维度或确定性模式: 直接按行优先顺序分配
    if (mnConfig.blockDimM <= thresholdDimM ||   // blockDimM <= 5
        thresholdDimM == 1 ||
        deterministicFlag == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        // 大 M 维度: 对角线策略 — 避免 Cube 访存冲突
        // 将 block 按 8×8 子矩阵对角线分配到 Cube 核
        ...
    }
}
```

**两种分配策略:**

#### 策略 1: 行优先 (Small M / 确定性模式)
```
M=3, N=4, blockDimM=3, blockDimN=4
每个 Core 的处理顺序:
  Core 0: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,0) -> ...
  Core 1: (0,1) -> (0,2) -> (0,3) -> (1,0) -> (1,1) -> ...
  ...
```

适用于:
- `blockDimM <= 5` (M 维分块数少)
- `deterministicFlag == 1` (**确定性模式强制使用此策略**)

#### 策略 2: 对角线 (Large M)
```
将 block 组织为 8×8 的子矩阵，以对角线方式分配，减少 Cube L0C 写冲突
```

适用于大 M 场景 (如 7168 tokens)，非确定性模式。

### 4.2 M 维度 (token 数) 的影响

```
M 维处理逻辑 (Process 函数):
┌──────────────────────────────────────────────────┐
│  groupIdx 从 0 到 groupNum-1:                     │
│    m = groupTokens[groupIdx]                      │
│    跳过 m <= 0 的 group                           │
│    mnConfig.m = m                                 │
│    blockDimM = Ceil(m, singleM)  // singleM=128   │
│    blockDimN = Ceil(n, singleN)  // singleN=256   │
│    totalBlocks = blockDimM × blockDimN            │
│                                                   │
│    当 m 较小时:                                    │
│      blockDimM 小 → 行优先策略                     │
│      blockDimM <= 5 → 直接线性分配                 │
│      Cube 利用率可能不足                           │
│                                                   │
│    当 m 很大时 (> 640):                            │
│      blockDimM > 5 → 对角线策略 (非确定性)          │
│      对角线策略 → 确定性模式强制回到行优先          │
└──────────────────────────────────────────────────┘
```

### 4.3 N=7168 的特殊 Shape 优化

这是 MoE 模型最典型的 shape (如 DeepSeek-V2/V3):

```cpp
// op_host/grouped_matmul_finalize_routing_base_tiling.cpp:358-365
if ((n_ == 7168 || n_ == 7680) && k_ == 2048) {
    uint32_t avg_m = tuningConfig_ ? tuningConfig_ : (m_ / groupNum_);

    // avg_m ∈ (128, 256] 时:
    //   baseM = 256, baseN = 128  (增大 M 分块, 减小 N 分块)
    // avg_m <= 128 或 > 256 时:
    //   baseM = 128, baseN = 256  (默认)
}
```

**调优原理:**
- 每个 token 被 route 到的 expert 数量有限，avg_m 反映每 expert 平均处理 token 数
- avg_m 在 (128, 256] 时，增大 baseM 到 256 可以减少 Cube launch 次数
- 同时减小 baseN 到 128 保证 UB 容量不溢出
- 这种交换本质是在 Cube 计算效率和 Vector 计算粒度之间做 trade-off

**vBaseM 也因此变化:**

| 配置 | vBaseM = UBCALSIZE/baseN | 含义 |
|------|--------------------------|------|
| baseN=256 | 4096/256 = 16 | 每次 Vector 计算处理 16 行 |
| baseN=128 | 4096/128 = 32 | 每次 Vector 计算处理 32 行 |

### 4.4 不同 Shape 对确定性窗口的影响

```
windowSize = deterWorkspaceSize / (n * sizeof(float))

n=7168:
  windowSize(96MB) = 100663296 / 28672 ≈ 3510 行
  windowSize(64MB) = 67108864 / 28672 ≈ 2340 行

n=7680:
  windowSize(96MB) = 100663296 / 30720 ≈ 3277 行
  windowSize(64MB) = 67108864 / 30720 ≈ 2185 行

n=4096:
  windowSize(96MB) ≈ 6144 行

n=2560:
  windowSize(96MB) ≈ 9830 行
```

**结论**: N 越小，窗口能容纳的行数越多，刷出频率越低，确定性开销越小。

### 4.5 Vector 维度的分块

VectorCompute 沿 M 和 N 二维分块处理反量化:

```cpp
// op_kernel/grouped_matmul_finalize_routing.h:422-469
// N 维外层循环
for (uint32_t offsetN = 0; offsetN < curCubeSingleN;
     offsetN += mnConfig.baseN) {
    curVecBaseN = mnConfig.baseN;  // 256 or 128

    // M 维内层循环 (按 vBaseM 分块)
    uint32_t vecBaseM = tiling->ubCalSize /
                        (Ceil(mnConfig.baseN, 8) * 8);  // 16 or 32
    vecBaseM = min(vecBaseM, curCubeSingleM);

    // M 维又分为 subBlock (AIV 内部双核)
    GetOffset(coreOffsetM, curCubeSingleM);
    // coreOffsetM.singleCoreM = curCubeSingleM / 2
    // subBlockIdx=0 处理前半, subBlockIdx=1 处理后半

    for (uint32_t offsetM = ...; offsetM < ...; offsetM += vecBaseM) {
        DataCopyMMOut(...);                // 搬 Cube 输出
        ComputeDequantAndActivate(...);    // 反量化 + bias + pertoken
        VectorAtomicProcess(...);          // 写入确定性 workspace / scatter
    }
}
```

---

## 5. 完整数据流与确定性保证

### 5.1 W8A8 确定性模式完整流程

```
输入: x(int8, m×k), weight(int8 NZ, e×k×n), scale(float, e×n),
      pertokenScale(float, m), groupList(int64, e),
      rowIndex(int64, m), logit(float, m), sharedInput(bf16, bsdp×n)

┌─────────────────────────────────────────────────────────────────┐
│ 1. PreProcess (AIV)                                             │
│    - 初始化输出 y 为 0                                           │
│    - 处理 sharedInput (乘系数 + 写 y 头部)                       │
│    SyncAll()                                                     │
├─────────────────────────────────────────────────────────────────┤
│ 2. 主循环 (AIC + AIV 交替)                                      │
│    For each group:                                               │
│      For each (mBlock, nBlock):                                 │
│        ┌─ AIC: MMCompute() ─────────────────────────────┐       │
│        │  INT8 Matmul: x(mBlock×k) × weight(k×nBlock)   │       │
│        │  → int32 结果存入 workspace                     │       │
│        │  CrossCoreSetFlag → AIV                         │       │
│        └────────────────────────────────────────────────┘       │
│        ┌─ AIV: VectorSync() ────────────────────────────┐       │
│        │  if curBlockM > lowBoundM:                      │       │
│        │    推进 curGroup/curM 到 lowBoundM               │       │
│        │    FRDeterministic() — 刷出窗口                  │       │
│        │    lowBoundM = curM + windowSize                │       │
│        └────────────────────────────────────────────────┘       │
│        ┌─ AIV: VectorCompute() ─────────────────────────┐       │
│        │  1. DataCopyScale: 搬 per-channel scale 到 UB  │       │
│        │  2. DataCopyBias: 搬 bias 到 UB (可选)          │       │
│        │  3. DataCopyPerTokenScale: 搬 pertoken scale    │       │
│        │  4. CrossCoreWaitFlag ← AIC                     │       │
│        │  For M-sub-block, N-block:                      │       │
│        │    a. DataCopyMMOut: 搬 int32 Matmul 结果到 UB  │       │
│        │    b. AscendDequant: per-channel 反量化          │       │
│        │       int32 × scale → float                     │       │
│        │    c. Mul(pertokenBrcbLocal, dequant):          │       │
│        │       per-token 反量化 (× logit)                 │       │
│        │    d. + bias (可选)                              │       │
│        │    e. Cast(float → float) + EnQue               │       │
│        │    f. VectorAtomicProcess:                      │       │
│        │       ★ 确定性路径:                              │       │
│        │         DataCopyPad2D → mmQuantOutGm            │       │
│        │         (写入确定性 workspace 窗口内偏移)         │       │
│        │       ★ 非确定性路径:                            │       │
│        │         AtomicAdd → yGm[rowIndex[i]*n + offset] │       │
│        │  CrossCoreSetFlag → AIC                         │       │
│        └────────────────────────────────────────────────┘       │
│                                                                  │
│ 3. 最终刷出 (AIV)                                                │
│    syncConfig.curM = mnConfig.offsetM (总 M 行)                  │
│    FRDeterministic(syncConfig)                                   │
│    → 最后一个窗口: scatter-add 到 y                               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 确定性正确性总结

| 非确定性模式 | 确定性模式 |
|------------|---------|
| 每个 VectorCompute 后立即 atomic scatter-add 到 y | 先写入确定性 workspace 窗口 |
| 多核可能同时写同一 y 行 | FRDeterministic 时按 outRow % coreNum 分工，核间不冲突 |
| 浮点累加顺序不可控 | 窗口内顺序确定 (mOffset 递增) |
| 无额外内存开销 | 64MB~96MB 确定性 workspace |
| 无额外同步 | 每次刷出前 SyncAll() |

### 5.3 关键代码路径索引

| 功能 | 文件 | 行号 |
|------|------|------|
| W8A8 Tiling | `op_host/..._base_tiling.cpp` | 350-399 |
| 确定性 Tiling | `op_host/..._base_tiling.cpp` | 413-425 |
| 窗口初始化 | `op_kernel/grouped_matmul_finalize_routing.h` | 303-306 |
| 确定性 workspace 地址 | `op_kernel/grouped_matmul_finalize_routing.h` | 162-166 |
| VectorSync (窗口滑动) | `op_kernel/grouped_matmul_finalize_routing.h` | 622-649 |
| FRDeterministic (刷出) | `op_kernel/grouped_matmul_finalize_routing.h` | 652-687 |
| 确定性写入路径 | `op_kernel/grouped_matmul_finalize_routing.h` | 381-386 |
| Final 刷出 | `op_kernel/grouped_matmul_finalize_routing.h` | 331-334 |
| SyncConfig 定义 | `op_kernel/..._routing_utils.h` | 49-56 |
| MNBlockIdxCompute | `op_kernel/..._routing_utils.h` | 140-161 |
| Process (主循环) | `op_kernel/grouped_matmul_finalize_routing.h` | 289-335 |
| TilingKey 分发 | `op_kernel/grouped_matmul_finalize_routing.cpp` | 110-170 |
| Infershape | `op_host/..._infershape.cpp` | 269-322 |
| 确定性 workspace 分配 | `op_host/..._base_tiling.cpp` | 27-28 |
| SetBufferSpace + GetTiling | `op_host/..._base_tiling.cpp` | 193-196, 376-378 |
| L1 tiling stepKa/depthA1 | `op_host/..._base_tiling.cpp` | 430-434 |
| curBlockM 计算 | `op_kernel/..._routing.h` | 338-349 |
| vecBaseM 计算 | `op_kernel/..._routing.h` | 432-433 |
| InitUbBuffer (UB 预算) | `op_kernel/..._routing.h` | 178-212 |
| VectorAtomicProcess 确定写入 | `op_kernel/..._routing.h` | 376-402 |
| DETER_UB_SIZE | `op_kernel/..._routing_utils.h` | 30 |
| MAX_K_A8W4_MSD | `op_host/..._base_tiling.cpp` | 49 |
| DETER_WORK_SPACE_SIZE | `op_host/..._base_tiling.cpp` | 27-28 |
| A8W8_UBRESTBYTES | `op_host/..._base_tiling.cpp` | 51 |
| UBCALSIZE | `op_host/..._base_tiling.cpp` | 48 |

---

## 6. 溢出防护机制

### 6.1 概述

GMMFR W8A8 确定性模式在三个层级实施溢出防护，覆盖从最外层 L2/Workspace 到最内层 UB：

| 层级 | 防护对象 | 防护方式 | 触发时机 | 失败后果 |
|------|---------|---------|---------|---------|
| Level 1 | 确定性 Workspace (L2) | 滑动窗口 + 提前刷出 | 运行时 VectorSync 检查 | 不会溢出，但多刷出降低性能 |
| Level 2 | UB (Unified Buffer) | 编译期预算 + vecBaseM 约束 | InitUbBuffer + VectorCompute | 编译期 UB 不足则无法启动 |
| Level 3 | L1 / L0C | Tiling 时 mm_.GetTiling() 检查 | Tiling 阶段 | 返回错误，算子无法执行 |

### 6.2 Level 1: 确定性 Workspace 溢出防护 (运行时窗口滑动)

#### 6.2.1 Workspace 大小的确定

确定性 workspace 的大小在 Host 端 tiling 阶段确定为 **64MB 或 96MB**，依据芯片 L2 总容量：

```cpp
// base_tiling.cpp:27-28
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024;      // 96MB
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024; // 64MB

// base_tiling.cpp:421-423
uint64_t l2_size;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
deterWorkspaceSize_ = l2_size > DETER_WORK_SPACE_SIZE ?
                      DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
```

逻辑：如果芯片 L2 > 96MB → 分配 96MB；否则 → 分配 64MB。此值通过 tiling data 字段 `deterWorkspaceSize` 传给 kernel。

#### 6.2.2 窗口容量约束

kernel 初始化时将 workspace 容量转为窗口行数上限：

```cpp
// kernel.h:303
syncConfig.windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(DTYPE_OUT));
```

对于 W8A8 [P::combine = true]，`DTYPE_OUT` 为 `float`（4 bytes）：

```
windowSize = deterWorkspaceSize / (n × 4)
```

**这意味着确定性 workspace 最多同时容纳 `windowSize` 行的部分结果。** 超出此容量时必须刷出。

#### 6.2.3 溢出检测：curBlockM vs lowBoundM

每次 AIC 完成一个 Matmul 块后，AIV 调用 `VectorSync` 检查是否需要刷出：

```cpp
// kernel.h:622-649
while (mnConfig.curBlockM > syncConfig.lowBoundM) {
    // 1. 推进 curGroup/curM 到 lowBoundM
    // 2. FRDeterministic(syncConfig) 刷出窗口
    // 3. lowBoundM = curM + windowSize   滑动窗口
}
```

**关键变量 `curBlockM`** — 当前已处理的**绝对行位置**（行尾）：

```cpp
// kernel.h:349
mnConfig.curBlockM = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM + curSingleM;
```

其中：
- `offsetM`: 前面所有 group 的累计 M 偏移
- `mIdx * singleM`: 当前 group 内的 M 块偏移
- `curSingleM`: 当前块的实际行数（处理尾部截断）

当 `curBlockM > lowBoundM` 时，意味着 **已经写入的行数据超出了当前窗口下界**，必须立即刷出，否则后续写入会覆盖未刷出的数据。

#### 6.2.4 写入地址的循环复用

确定性路径的写入不是直接 scatter 到 y，而是写入 `mmQuantOutGm` 的窗口内偏移：

```cpp
// kernel.h:381-386 (确定性写入路径)
DataCopyPad2D(
    mmQuantOutGm[vecAParams.yGmOffset1 -
        (syncConfig.lowBoundM - syncConfig.windowSize) * tiling->n],
    yLocal, dimParams, tiling->n);
```

写入的物理地址公式：
```
物理偏移 = 全局M偏移 - (lowBoundM - windowSize) × n
         = 全局M偏移 - 窗口基址
```

`(lowBoundM - windowSize)` 是当前窗口在全局 M 空间中的**起始行号**。减去它后将全局坐标映射到 workspace 内部的物理地址。每轮刷出后 `lowBoundM += windowSize`，窗口基址前移，**同一块物理 workspace 被循环复用**。

**溢出防护原理图：**

```
确定性 Workspace (固定大小):
┌────────────────────────────────────────────────┐
│  ←─────────── windowSize 行 ──────────→        │
│ [行0..行(窗口-1)]                               │
│  ↑                                              │
│  物理地址: mmQuantOutGm[0 .. windowSize*n-1]     │
│                                                 │
│ 窗口 1: 全局 M [0, windowSize) → 物理 [0, ..]    │
│   → FRDeterministic 刷出到 y                     │
│ 窗口 2: 全局 M [windowSize, 2×windowSize)        │
│   → 复用同一块物理地址 [0, ..]                    │
│ 窗口 3: ...                                      │
└────────────────────────────────────────────────┘

溢出防护：curBlockM > lowBoundM → 立即刷出 → lowBoundM += windowSize
                                   ↑
                              新写入的地址永远不会超出物理 workspace 边界
```

#### 6.2.5 溢出防护的触发频率

```
触发条件: curBlockM > lowBoundM

在 Process() 主循环中:
  For each group:
    For each (mBlock, nBlock):
      MMCompute()   → curBlockM 推进
      VectorSync()  → 检查 curBlockM > lowBoundM ?
                       是 → FRDeterministic + 窗口滑动
      VectorCompute() → 写入 mmQuantOutGm
```

由于每次 MMCompute + VectorCompute 后都检查，**最多延迟一个 Matmul tile (singleM=128 行) 就会发现溢出**。在最坏情况下，实际写入的行数最多超出窗口下界 `(singleM - 1)` 行，但这部分数据的写入地址经过了窗口基址修正，始终落在物理 workspace 内。

### 6.3 Level 2: UB Buffer 溢出防护 (编译期 UB 预算)

#### 6.3.1 UB 总预算

A3 (910B) 的 UB 为 **192KB**，910C 为 **256KB**。`InitUbBuffer()` 在 kernel 启动时一次性分配所有 TQue 和 TBuf：

| Buffer | 分配代码 (kernel.h) | 计算公式 | 大小 (bytes) |
|--------|---------------------|---------|-------------|
| `scaleInQueue` | line 184 | `BUFFER_NUM × baseN × sizeof(float)` = 2×256×4 | 2,048 |
| `biasInQueue` | line 185 | `BUFFER_NUM × baseN × sizeof(bf16)` = 2×256×2 | 1,024 |
| `perTokenScaleInQueue` | lines 192-196 | `2 × Ceil(128/2 × 4 × 2, 32) × 32` (hasPertokenScale) | 1,024 |
| `vecInQueue` | line 198 | `BUFFER_NUM × ubCalSize × sizeof(int32_t)` = 2×4096×4 | 32,768 |
| `vecOutQueue` | line 199 | `BUFFER_NUM × ubCalSize × sizeof(float)` = 2×4096×4 | 32,768 |
| `queBind` (确定性) | line 187 | `BUFFER_NUM × DETER_UB_SIZE` = 2×12288 | 24,576 |
| `tmpBuff` | line 200 | `ubRestBytes = A8W8_UBRESTBYTES` = 101376 | 101,376 |
| **合计** | | | **~194,560 (~190KB)** |

190KB < 192KB → 刚好适配 910B 的 UB，仅 2KB 余量。

**注**：`perTokenScaleInQueue` 的大小在不同配置下有变化。若 `hasPertokenScale=0` 则大小为 512 bytes；若 `hasPertokenScale=1` 且 `P::combine=true` 则为 1024 bytes（per-token scale 和 logit 打包在同一 buffer）。

#### 6.3.2 vecBaseM — Vector 分块的行数约束

```cpp
// kernel.h:432-433
uint32_t vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, uint32_t(8)) * 8);
vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
```

对于 W8A8 (`baseN=256`)：
```
vecBaseM = 4096 / (Ceil(256, 8) × 8)
         = 4096 / (32 × 8)
         = 4096 / 256
         = 16 行
```

**保证**：每次 `VectorCompute` 处理的 M 行数 ≤ `vecBaseM=16`，因此：
- `vecInQueue` 最多容纳 16 × 256 = 4096 个 int32 = **16KB**（总线 32KB，安全）
- `vecOutQueue` 最多容纳 16 × 256 = 4096 个 float = **16KB**（总线 32KB，安全）

#### 6.3.3 tmpBuff 内部分区

```cpp
// kernel.h:201-211
uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);  // = 16384

dequantMiddleResult = tmpBuff[0       .. 16383]    // 16KB: 反量化输出
pertokenBrcbLocal   = tmpBuff[16384   .. 32767]    // 16KB: per-token scale广播结果
mulsResultLocal     = tmpBuff[32768   .. 49151]    // 16KB: 乘法中间结果
biasCalcLocal       = tmpBuff[49152   .. 65535]    // 16KB: bias计算
sharedTmpLocal      = tmpBuff[65536   .. 98303]    // 32KB: AscendDequant API临时空间
// unused           = tmpBuff[98304   .. 101375]   //  3KB: 保留
```

```
tmpBuff (101376 bytes) 布局:
┌──────────────┬──────────────┬──────────────┬──────────────┬────────────────────┬─────────┐
│ dequantMid   │ pertokenBrcb │ mulsResult   │ biasCalc     │ sharedTmpLocal     │ unused  │
│  16384 B     │  16384 B     │  16384 B     │  16384 B     │  32768 B           │ 3072 B  │
└──────────────┴──────────────┴──────────────┴──────────────┴────────────────────┴─────────┘
  使用 98304 bytes                                               保留 3072 bytes
```

**保证**：每个分区大小 = `ubCalSize × sizeof(float) = 16384` bytes = 4096 个 float。ComputeDequantAndActivate 中的 `computeSize = curVecBaseM × alignBaseN ≤ vecBaseM × baseN = 16 × 256 = 4096`，确保所有中间计算结果不超出各自分区。

#### 6.3.4 K 维度约束

```cpp
// base_tiling.cpp:49
constexpr uint32_t MAX_K_A8W4_MSD = 18432;
// 注释: "k is limited by pre process, a line of X should be able to put in UB"
```

一行 X 为 `k × sizeof(int8_t) = k bytes`。`MAX_K = 18432` 确保一行 X（~18KB）可放入 UB。对于 W8A8，虽然没有 pre-process（只有 A8W4 有），但此约束确保即使最坏情况下单行数据不超 UB。

### 6.4 Level 3: L1 / L0C Tiling 溢出防护 (Tiling 阶段检查)

#### 6.4.1 SetBufferSpace 传递硬件约束

```cpp
// base_tiling.cpp:193-196
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
```

这告诉 Matmul tiling 求解器硬件的真实容量，求解器在内部计算 tiling 参数时会遵守这些限制。

#### 6.4.2 GetTiling 失败检查

```cpp
// base_tiling.cpp:376-378
if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
    OP_LOGE(..., "Get Tiling Failed!, m, n, k: %lu, %lu, %lu", m_, n_, k_);
    return ge::GRAPH_FAILED;
}
```

如果 `SetFixSplit(baseM, baseN, baseK)` 提出的分块方案超出 L1/L0C 容量，`GetTiling()` 返回 -1，算子**在 tiling 阶段就报错退出**。绝对不会在运行时发生 L1/L0C 溢出。

#### 6.4.3 L0C 约束分析

```
L0C (910B) = 128KB
L0C (910C) = 256KB

W8A8 L0C 需求 = baseM × baseN × sizeof(int32_t)
             = 128 × 256 × 4
             = 131,072 bytes = 128KB
```

恰好等于 910B 的 L0C 容量，这是最紧的约束。910B 无法开启 L0C double buffer（`set_dbL0C(1)` 在 line 429）。910C 有 256KB L0C，可以开启 `dbL0C=2` 但当前 W8A8 代码未使用（保持 dbL0C=1）。

如果调整 baseM=256, baseN=256：
```
L0C 需求 = 256 × 256 × 4 = 262,144 bytes = 256KB
```
这会超出 910B 的 128KB L0C → `GetTiling()` 返回 -1 → **算子失败**。

#### 6.4.4 L1 约束分析

```
L1 = 512KB (both 910B and 910C)

A 矩阵 (左矩阵, int8):
  stepKa × depthA1 × baseK × baseM × sizeof(int8)
  = 4 × 8 × 128 × 128 × 1
  = 524,288 bytes = 512KB  ← 恰好等于 L1 总容量

B 矩阵 (右矩阵, int8 NZ):
  stepKb × depthB1 × baseK × baseN × sizeof(int8)
  = 4 × 8 × 128 × 256 × 1
  = 1,048,576 bytes = 1024KB  ← 超过 L1
```

B 矩阵超 L1 但不会导致失败，原因是：
- B 矩阵数据来自 GM，通过 L2 Cache 透明缓冲
- Line 362-363: 当 `blockDimM == 1` 时设置 `SetL2CacheHint(CACHE_MODE_DISABLE)`
- L1 的 `depthB1=8` 表示 L1 中可以同时驻留 8 份 stepKb 的 B 数据，物理上通过 DMA 控制器管理 L1 ↔ L2 ↔ GM 的数据搬移
- Tiling 求解器内部会检查实际 L1 占用是否超过物理 L1 容量

#### 6.4.5 各 Buffer 容量约束总表

| Buffer | 910B 容量 | 910C 容量 | W8A8 使用量 | 最紧约束项 |
|--------|-----------|-----------|------------|-----------|
| L0C | 128KB | 256KB | 128KB | `baseM × baseN × 4` |
| L1 | 512KB | 512KB | 512KB (A矩阵) | `stepKa × depthA1 × baseK × baseM` |
| L0A | 64KB | 64KB | ~16KB | `baseK × baseM × 1` (单step) |
| L0B | 64KB | 64KB | ~32KB | `baseK × baseN × 1` (单step) |
| UB | 192KB | 256KB | ~190KB | All TQue + TBuf |
| L2 | varies | varies | 64MB/96MB | 确定性 workspace |

---

## 7. 不同 Shape 下的窗口行为示例

### 7.1 窗口大小速查表

```cpp
// kernel.h:303
syncConfig.windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(DTYPE_OUT));
// W8A8: DTYPE_OUT = float = 4 bytes
```

| n (N dim) | deter=64MB | deter=96MB | 典型场景 |
|-----------|-----------|-----------|---------|
| 2560 | 6553 | 9830 | 小 N 模型 |
| 4096 | 4096 | 6144 | 中等 N |
| 7168 | 2340 | **3510** | **DeepSeek-V2/V3 常见** |
| 7680 | 2184 | 3277 | Qwen2-MoE 常见 |
| 13824 | 1213 | 1820 | 大 N 模型 |

**规律**: N 越小 → windowSize 越大 → 刷出频率越低 → 确定性开销越小。

### 7.2 示例 1: 无需中途刷出 (Small Total M)

**场景**: 4 experts，每个 64 tokens，n=7168，96MB workspace

```
参数:
  e = 4, groupList = [64, 64, 64, 64]
  n = 7168, k = 2048
  deterWorkspaceSize = 96MB
  windowSize = 96 × 1024 × 1024 / (7168 × 4) = 3510 行
  totalM = 64 + 64 + 64 + 64 = 256 行
  singleM = 128

执行时间线:
┌──────────────────────────────────────────────────────────┐
│ Group 0 (64 tokens):                                      │
│   m=64, blockDimM = Ceil(64, 128) = 1                    │
│   curBlockM: 0 → 64                                      │
│   VectorSync: 64 ≤ 1500(lowBoundM)?  → 不触发            │
│                                                          │
│ Group 1 (64 tokens):                                      │
│   curBlockM: 64 → 128                                    │
│   VectorSync: 128 ≤ 1500?  → 不触发                      │
│                                                          │
│ Group 2 (64 tokens):                                      │
│   curBlockM: 128 → 192                                   │
│   VectorSync: 192 ≤ 1500?  → 不触发                      │
│                                                          │
│ Group 3 (64 tokens):                                      │
│   curBlockM: 192 → 256                                   │
│   VectorSync: 256 ≤ 1500?  → 不触发                      │
│                                                          │
│ 循环结束: totalM = 256                                    │
│ 最终 FRDeterministic: 刷出 [0, 256)                       │
└──────────────────────────────────────────────────────────┘

结论: 整个执行过程只有 1 次刷出（最终刷出），确定性开销最小。
```

### 7.3 示例 2: 中途触发一次刷出 (Large Total M)

**场景**: 4 experts，1024 tokens each，n=7168，96MB workspace

```
参数:
  groupList = [1024, 1024, 1024, 1024]
  totalM = 4096 行
  windowSize = 3510
  singleM = 128

时间线追踪:
───────────────────────────────────────────────────────────

Group 0 (1024 tokens):
  blockDimM = Ceil(1024, 128) = 8
  处理 8 个 mBlock:
    mBlock 0: curBlockM = 0   + 0*128 + 128 = 128
    mBlock 1: curBlockM = 0   + 1*128 + 128 = 256
    ...
    mBlock 7: curBlockM = 0   + 7*128 + 128 = 1024
  每次 VectorSync: curBlockM(128..1024) ≤ lowBoundM(3510) → 不触发

Group 1 (1024 tokens):
  offsetM = 1024 (group 0 累计)
  mBlock 0: curBlockM = 1024 + 0*128 + 128 = 1152  ≤ 3510 → OK
  ...
  mBlock 7: curBlockM = 1024 + 7*128 + 128 = 2048  ≤ 3510 → OK

Group 2 (1024 tokens):
  offsetM = 2048
  mBlock 0: curBlockM = 2048 + 128 = 2176  ≤ 3510 → OK
  ...
  mBlock 7: curBlockM = 2048 + 1024 = 3072  ≤ 3510 → OK

Group 3 (1024 tokens):
  offsetM = 3072
  mBlock 0: curBlockM = 3072 + 128 = 3200  ≤ 3510 → OK
  mBlock 1: curBlockM = 3072 + 256 = 3328  ≤ 3510 → OK
  mBlock 2: curBlockM = 3072 + 384 = 3456  ≤ 3510 → OK
  mBlock 3: curBlockM = 3072 + 512 = 3584  > 3510 ⚡触发 VectorSync
              ↓
    ┌─────────────────────────────────────────────────────┐
    │ VectorSync 执行:                                    │
    │                                                     │
    │ 内层循环推进 curGroup:                              │
    │   curGroup=0, mi=1024, curGroupM=1024 ≤ 3510 → ✓   │
    │   curGroup=1, mi=1024, curGroupM=2048 ≤ 3510 → ✓   │
    │   curGroup=2, mi=1024, curGroupM=3072 ≤ 3510 → ✓   │
    │   curGroup=3, mi=1024, curGroupM=4096 > 3510 → ✗   │
    │     → curM 只能推进到:                              │
    │       curM + (3510-3072)/128 * 128                  │
    │       = 3072 + 3*128 = 3456                  │
    │     → break (group 3 跨窗口边界)                     │
    │                                                     │
    │ FRDeterministic: 刷出窗口 [0, 3456)                  │
    │ lowBoundM = 3456 + 3510 = 6966                      │
    └─────────────────────────────────────────────────────┘

  Group 3 继续:
    mBlock 4: curBlockM = 3072 + 640 = 3712  ≤ 6966 → OK
    ...
    mBlock 7: curBlockM = 3072 + 1024 = 4096  ≤ 6966 → OK

循环结束:
  syncConfig.curM = 4096
  final FRDeterministic: 刷出窗口 [3456, 4096)  共 640 行

───────────────────────────────────────────────────────────
总结: 2 次刷出 (中途 1 次 + 最终 1 次)
  第 1 次: 3456 行 → 覆盖 group 0,1,2 全量 + group 3 前 384 行
  第 2 次: 640 行  → 覆盖 group 3 剩余 640 行
```

**关键**: group 3 跨窗口边界时，VectorSync 不会把整个 group 3 都刷出，而是按 `singleM=128` 对齐只刷到 3456（=3072+3×128），剩余的 640 行留在下一个窗口。确保窗口内的数据始终对齐到 `singleM` 边界。

### 7.4 示例 3: 不均匀分布 + Group 边界追踪

**场景**: 8 experts，不均匀 token 分布，n=7168，96MB workspace

```
groupList = [100, 200, 50, 800, 150, 300, 100, 400]
totalM = 2100 行
windowSize = 3510

2100 < 3510 → 整个执行不需要中途刷出，只有最终一次。

但假设 windowSize 缩小到 1500（对应 deterWorkspace 较小或 n 较大的场景）:
───────────────────────────────────────────────────────────

初始: lowBoundM = 1500, curGroup=0, curGroupM=0, curM=0

Group 0 (100): curBlockM 推进 0→100, ≤1500, OK
Group 1 (200): curBlockM 推进 100→300, ≤1500, OK
Group 2 (50):  curBlockM 推进 300→350, ≤1500, OK
Group 3 (800): curBlockM 推进 350→1150, ≤1500, OK
Group 4 (150): curBlockM 推进 1150→1300, ≤1500, OK
Group 5 (300):
  mBlock 0: curBlockM = 1300 + 128 = 1428, ≤1500, OK
  mBlock 1: curBlockM = 1300 + 256 = 1556, >1500 ⚡触发!

  VectorSync 内层循环:
    Step 1: curGroup=0, mi=100, curGroupM=0+100=100 ≤1500
    Step 2: curGroup=1, mi=200, curGroupM=100+200=300 ≤1500
    Step 3: curGroup=2, mi=50,  curGroupM=300+50=350 ≤1500
    Step 4: curGroup=3, mi=800, curGroupM=350+800=1150 ≤1500
    Step 5: curGroup=4, mi=150, curGroupM=1150+150=1300 ≤1500
    Step 6: curGroup=5, mi=300, curGroupM=1300+300=1600 >1500 ✗
            → curM = 1300 + (1500-1300)/128 * 128
                   = 1300 + 1*128 = 1428
            → break

  FRDeterministic: 刷出 [0, 1428)
  lowBoundM = 1428 + 1500 = 2928

  此时 curGroup=5 仍持有 300-128=172 个未处理的 token
  (curGroupM 被重置逻辑处理，或在下一次窗口中对齐)

继续 Group 5 剩余...
  ...
```

**这个例子揭示了 VectorSync 的一个重要设计**：group 跨窗口边界时，只刷出到 `singleM` 对齐边界，不会丢弃部分 group 的已计算数据，剩余部分在下个窗口继续。

### 7.5 刷出频率分析

```
总刷出次数 = max(1, Ceil(totalM / windowSize))

其中:
  - 最后一次刷出覆盖剩余的 ≤ windowSize 行
  - 除最后一次外，每次刷出 windowSize 行（按 singleM 对齐）
```

| totalM | n=7168, 96MB (windowSize=3510) | n=7168, 64MB (windowSize=2340) | n=4096, 96MB (windowSize=6144) |
|--------|-------------------------------|-------------------------------|-------------------------------|
| 256 | 1 | 1 | 1 |
| 1024 | 1 | 1 | 1 |
| 2048 | 1 | 1 | 1 |
| 4096 | 2 | 2 | 1 |
| 8192 | 3 | 4 | 2 |
| 16384 | 5 | 8 | 3 |

**结论**: 
- N 越小 → windowSize 越大 → 刷出次数越少
- 在 n=7168 场景下（MoE 最常见），`totalM ≤ 2340` 时完全不需要中途刷出
- 对于大 batch 推理（如 totalM=16384），64MB workspace 需要 8 次刷出，96MB 仅需 5 次

### 7.6 n=7168/7680, k=2048 特殊 Shape 下的窗口行为

```cpp
// base_tiling.cpp:358-365
if ((n_ == 7168 || n_ == 7680) && k_ == 2048) {
    uint32_t avg_m = ...;
    baseM = (avg_m > 128 && avg_m <= 256) ? 256 : 128;   // baseM 可能变为 256
    baseN = (avg_m > 128 && avg_m <= 256) ? 128 : 256;   // baseN 可能变为 128
}
```

当 avg_m ∈ (128, 256] 时，baseM 和 baseN 互换：
- `singleM` = baseM = **256** (原为 128)
- `singleN` = baseN = **128** (原为 256)

对窗口的影响：
- **windowSize 不变**：只取决于 n=7168 和 workspace 大小，不受 baseM/baseN 影响
- **跨窗口对齐粒度变化**：VectorSync 中的对齐使用 `mnConfig.singleM` (line 643)，对齐粒度从 128 → 256
- **vBaseM 变化**：`4096 / 128 = 32` (原为 16)，Vector 计算一次处理更多行

```
对比:
  默认 (baseM=128, baseN=256):
    singleM = 128, 跨窗口对齐 = 128 行
    
  优化 (baseM=256, baseN=128, avg_m∈(128,256]):
    singleM = 256, 跨窗口对齐 = 256 行
    → 跨窗口时 group 边界切分粒度更粗
    → 更少的碎片对齐，但可能浪费最多 255 行的窗口空间
```

---

## 8. 硬件 Buffer 与 L2/Workspace 分配分析

### 8.1 A3 硬件 Buffer 规格

A3 系列芯片 (Ascend 910B / 910C) 的片上存储层次：

```
┌─────────────────────────────────────────────────┐
│                   GM (HBM/DDR)                   │
│                  数十 GB                         │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│                   L2 (片上 SRAM)                 │
│               varies (决定 workspace 大小)       │
│          ┌──────────────────────────┐           │
│          │  确定性 Workspace         │           │
│          │  64MB or 96MB             │           │
│          └──────────────────────────┘           │
└───────────────┬─────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    ▼                       ▼
┌───────────┐         ┌───────────┐
│  AI Core  │  ...    │  AI Core  │
│  ┌─────┐  │         │  ┌─────┐  │
│  │ L1  │ 512KB      │  │ L1  │  │
│  │ ┌─┐ │            │  │ ┌─┐ │  │
│  │ │L0││  UB:192KB  │  │ │L0││  │
│  │ │A ││  64KB      │  │ │A ││  │
│  │ │L0││            │  │ │L0││  │
│  │ │B ││  64KB      │  │ │B ││  │
│  │ │L0││            │  │ │L0││  │
│  │ │C ││  128KB     │  │ │C ││  │
│  │ └─┘ │            │  │ └─┘ │  │
│  └─────┘            │  └─────┘  │
└─────────────────────┴───────────┘
```

| Buffer | 910B (A3) | 910C (A3) | 作用 |
|--------|-----------|-----------|------|
| UB | 192KB | 256KB | Vector 计算中间结果、TQue 队列 |
| L1 | 512KB | 512KB | Cube 计算时缓存的矩阵分块数据 |
| L0A | 64KB | 64KB | 左矩阵 (A) 的 Cube 输入 |
| L0B | 64KB | 64KB | 右矩阵 (B) 的 Cube 输入 |
| L0C | 128KB | 256KB | Cube 累加器 (int32 部分和) |
| L2 | varies | varies | 确定性 workspace + 系统 RPC |

数据来源：`GetCoreMemSize()` 运行时查询 + `cgmct/utils/arch.h` 编译期常量。

### 8.2 Workspace 总量拆解

```
TotalWorkspace = userWorkspace + systemWorkspace + deterWorkspace
```

#### 8.2.1 userWorkspace: Cube Matmul 输出

```cpp
// base_tiling.cpp:352
userWorkspaceSize = CV_PARALL_NUM × 256 × 128 × sizeof(int32_t) × blockDim_
                  = 4 × 256 × 128 × 4 × blockDim_
                  = 524,288 × blockDim_  bytes
```

每核 512KB。`CV_PARALL_NUM=4` 意味着 4 组 Cube-Vector 流水并行，每组需要独立的 Matmul 输出缓冲区。

```
userWorkspace 布局 (每核):
┌──────────┬──────────┬──────────┬──────────┐
│  Slot 0  │  Slot 1  │  Slot 2  │  Slot 3  │
│  128KB   │  128KB   │  128KB   │  128KB   │
│ baseM×   │          │          │          │
│ baseN×4  │          │          │          │
└──────────┴──────────┴──────────┴──────────┘
```

#### 8.2.2 systemWorkspace: RPC 保留

```cpp
// base_tiling.cpp:31-32, 353
constexpr uint64_t RPC_WORKSIZE = 20;      // 20 MB
constexpr uint64_t MB_SIZE = 1024 * 1024;
systemWorkspaceSize = 20 × 1024 × 1024 = 20,971,520 bytes
```

用于 Runtime 内部 RPC 通信，与算子逻辑无关。

#### 8.2.3 deterWorkspace: 确定性窗口

```cpp
// base_tiling.cpp:27-28, 422-424
deterWorkspaceSize = L2 > 96MB ? 96MB : 64MB
```

#### 8.2.4 总 Workspace 示例

假设 blockDim_ = 2（2 个 AIC 核），L2 > 96MB：

```
userWorkspace  = 512KB × 2        =    1 MB
systemWorkspace                     =   20 MB
deterWorkspace                      =   96 MB
─────────────────────────────────────────────
Total                               = ~117 MB
```

### 8.3 各硬件 Buffer 如何约束 Tiling

```
L0C (128KB) ──约束──→ baseM × baseN ≤ 128KB / sizeof(int32) = 32768
                     当前: 128 × 256 = 32768 ✓ (恰好)

L1 (512KB) ──约束──→ stepKa × depthA1 × baseK × baseM ≤ 512KB
                     当前: 4 × 8 × 128 × 128 = 524288 = 512KB ✓ (恰好)

L0A (64KB) ──约束──→ 单 step: baseK × baseM × sizeof(int8) = 128 × 128 = 16KB ✓

L0B (64KB) ──约束──→ 单 step: baseK × baseN × sizeof(int8) = 128 × 256 = 32KB ✓

UB (192KB) ──约束──→ 所有 TQue + TBuf ≈ 190KB ✓ (2KB 余量)

L2        ──约束──→ windowSize = deterWorkspaceSize / (n × 4)
                     n 越大 → windowSize 越小 → 刷出越频繁
```

**最紧的三个约束**:
1. **L0C = 128KB** ←→ `baseM × baseN = 32768`：决定了 baseM/baseN 的乘积上限
2. **L1 = 512KB** ←→ `stepKa × depthA1 × baseK × baseM = 512KB`：限制了 K 维度的流水深度
3. **UB = 192KB** ←→ 总 UB 占用 ~190KB：仅 2KB 余量，无法再增加任何 buffer

### 8.4 DETER_UB_SIZE 的计算逻辑

```cpp
// utils.h:30
constexpr uint32_t DETER_UB_SIZE = 12 * 1024;  // 12KB

// kernel.h:305-306
uint64_t nTimes = Ceil(tiling->n, DETER_UB_SIZE / sizeof(DTYPE_OUT));
syncConfig.baseN = Ceil(Ceil(tiling->n, nTimes), 128) * 128;
```

`DETER_UB_SIZE = 12KB` 是 `queBind` 单 buffer 的大小。在 FRDeterministic 中，queBind 用于将确定性 workspace 中的数据通过 UB 中转到 y：

```
对于 n=7168, DTYPE_OUT=float:
  DETER_UB_SIZE / sizeof(float) = 12288 / 4 = 3072 个 float / buffer
  nTimes = Ceil(7168, 3072) = 3
  baseN = Ceil(Ceil(7168, 3), 128) × 128 = Ceil(2390, 128) × 128
        = 19 × 128 = 2432

FRDeterministic 沿 N 维分 3 次搬完（每次搬 2432 或更少的列）：

  nOffset=0:    curVecBaseN = 2432
  nOffset=2432: curVecBaseN = 2432  
  nOffset=4864: curVecBaseN = 7168 - 4864 = 2304
```

`baseN=2432` 确保每块 N ≤ `3072` 个 float = 12KB，不超出 queBind 的单 buffer 容量。

### 8.5 n=7168/7680 特殊 Shape 对各 Buffer 的影响

当 `avg_m ∈ (128, 256]` 时，baseM/B=256, baseN=128：

| Buffer | 默认 (128×256) | 优化 (256×128) | 变化 |
|--------|---------------|---------------|------|
| L0C | 128×256×4 = 128KB | 256×128×4 = 128KB | 不变（乘积相同） |
| L1 (A) | 4×8×128×128 = 512KB | 4×8×128×256 = 1024KB | **翻倍！超 L1** |
| L0A | 128×128 = 16KB | 128×256 = 32KB | 翻倍 |
| L0B | 128×256 = 32KB | 128×128 = 16KB | 减半 |
| UB (vBaseM) | 4096/256 = 16 | 4096/128 = 32 | Vector 处理更多行 |
| windowSize | 3510 | 3510 | 完全不变 |

**注意**：当 baseM=256 时 L1 的 A 矩阵需求变为 1024KB，超过 512KB L1。这通过 `SetL2CacheHint` 和 Matmul tiling 求解器的内部 L1 ↔ L2 管理来处理——B 矩阵更多地依赖 L2 Cache，而 A 矩阵获得更多 L1 驻留。

---

