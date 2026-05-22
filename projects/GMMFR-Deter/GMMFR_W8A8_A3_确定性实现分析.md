# GMMFR W8A8 int8 A3 确定性实现分析 —— 滑窗回写机制

> **芯片**: Ascend 910B/910C (A3, arch22, `__CCE_AICORE__ == 220`)
> **数据类型**: activation INT8 ND × weight INT8 NZ → output FLOAT32
> **确定性**: 开启 (`deterministicFlag = 1`)
> **核心机制**: 确定性工作空间 + 滑窗回写 (Sliding Window Write-Back)

---

## 1. 为什么非确定性？—— 问题根因

### 1.1 非确定性的来源

GMMFR 的输出是 scatter-write：每个 token 经过 MatMul + Dequant 后，通过 `tokenRanksGm` 路由到不同的输出行。非确定性路径的核心写回逻辑：

```cpp
// grouped_matmul_finalize_routing.h:388-396 — 非确定性路径
SetAtomicAdd<float>();  // ← 开启浮点原子加
DataCopyExtParams paramsOut{1, static_cast<uint32_t>(vecAParams.curVecBaseN * sizeof(float)), 1, 1, 0};
for (uint32_t i = 0; i < vecAParams.curVecBaseM; i++) {
    auto outRow = static_cast<uint64_t>(
        tokenRanksGm.GetValue(vecAParams.mGlobalOffset + vecAParams.offsetM + i));
    DataCopyPad(yGm[outRow * tiling->n + vecAParams.yGmOffset0],
                yLocal[i * vecAParams.alignBaseN], paramsOut);  // 直接 scatter 到 yGm
}
SetAtomicNone();
```

**非确定性根因有两个层面**：

| 层面 | 原因 | 影响 |
|------|------|------|
| **浮点累加顺序** | 多核通过 AtomicAdd 并发写同行不同列块，浮点 `(a+b)+c ≠ a+(b+c)` | 每次运行结果数值不同 |
| **核间调度顺序** | 各核独立推进 group 循环，无全局同步点 | 不同 token 的写回顺序不固定 |

关键场景：MoE 中多个 expert 的 token 可能路由到同一个输出行 → 多核同时对该行做 AtomicAdd → 累加顺序不确定。

### 1.2 非确定性路径的完整写回流

```
AIC Core 0..N-1  (MatMul)                 AIV Core 0..M-1  (Vector)
┌──────────────────────────┐              ┌──────────────────────────┐
│ mm.SetTensorA / SetTensorB│              │ DataCopyScale (N方向)    │
│ while (mm.Iterate()) {   │              │ DataCopyBias             │
│   mm.GetTensorC(ws)      │──CrossCore──▶│ DataCopyPerTokenScale    │
│   SetFlag(AIC→AIV, 5)    │  SetFlag     │ WaitFlag(AIC→AIV, 5)    │
│ }                        │              │ DataCopyMMOut (从ws读)  │
│ WaitFlag(AIV→AIC, 3)     │◀─CrossCore───│ AscendDequant            │
│                          │  SetFlag     │ Mul/Muls + Bias + Cast  │
└──────────────────────────┘              │ SetAtomicAdd<float>()    │  ← 非确定性!
                                          │ for each row:            │
                                          │   outRow=tokenRanks[i]   │
                                          │   DataCopyPad→yGm[outRow]│
                                          │ SetAtomicNone()           │
                                          │ SetFlag(AIV→AIC, 3)      │
                                          └──────────────────────────┘
```

---

## 2. 确定性方案总览

### 2.1 核心思想：延迟写回 + 滑窗批量同步

确定性方案引入了一个**确定性工作空间（Deterministic Workspace）**作为中间缓冲，将 scatter-write 拆成两个阶段：

```
阶段1 (Compute):  AIV 将结果写入确定性工作空间 (按 token 顺序线性排列)
阶段2 (Flush):   当窗口填满时，全局同步，按行号取模分配核，顺序回写到 yGm
```

```
非确定性:  Cube → ws → Dequant → AtomicAdd → yGm (即时 scatter, 无序)
确定性:    Cube → ws → Dequant → mmQuantOutGm (线性写) ──[窗口满]──▶ SyncAll → → yGm (有序 scatter)
                                                         滑窗触发
```

### 2.2 关键改动一览

| 改动点 | 非确定性 | 确定性 | 文件:行号 |
|--------|---------|--------|-----------|
| Tiling: workspace 分配 | userWS + systemWS | + deterWorkspace (64/96MB) | `base_tiling.cpp:413-425` |
| Kernel: Init 指针 | `mmQuantOutGm` 不初始化 | 指向 deterWS 区域 | `grouped_matmul_finalize_routing.h:162-166` |
| UB: queBind 缓冲 | 不分配 | 分配 DETER_UB_SIZE=12KB | `grouped_matmul_finalize_routing.h:186-188` |
| Process: 主循环 | 无 VectorSync | 每个 tile 后调用 VectorSync | `grouped_matmul_finalize_routing.h:324` |
| Process: 收尾 | 不处理 | 调用 FRDeterministic 最终 flush | `grouped_matmul_finalize_routing.h:331-334` |
| MNBlockIdxCompute | 可选 diagonal | 强制 row-priority | `grouped_matmul_finalize_routing_utils.h:143` |
| VectorAtomicProcess | AtomicAdd scatter | DataCopyPad2D → mmQuantOutGm | `grouped_matmul_finalize_routing.h:381-386` |
| Sync | CrossCoreSet/WaitFlag | 增加 SyncAll × 2 | `grouped_matmul_finalize_routing.h:654-686` |

---

## 3. Tiling 层：确定性工作空间分配

### 3.1 DeterministicTilingProcess

```cpp
// base_tiling.cpp:413-425
void GroupedMatmulFinalizeRoutingBaseTiling::DeterministicTilingProcess()
{
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;    // ← 非确定性: 直接返回, 不做任何分配
        return;
    }
    deterministicFlag_ = 1;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
    // 根据 L2 大小选择工作空间: 大 L2 → 96MB, 小 L2 → 64MB
    deterWorkspaceSize_ = l2_size > DETER_WORK_SPACE_SIZE ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
    workspaceSize_ += deterWorkspaceSize_;  // ← 叠加到总 workspace
}
```

**工作空间大小策略**：
```
L2 > 96MB  → deterWorkspaceSize = 96MB   (DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024)
L2 ≤ 96MB  → deterWorkspaceSize = 64MB   (DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024)
```

### 3.2 总 Workspace 布局

```
workspace 总布局 (非确定性):
┌─────────────────────────────────┬──────────────────┐
│  Cube MatMul 输出缓冲 (int32)    │  System RPC (20MB)│
│  parallNum × baseM × baseN      │                   │
│  × sizeof(int32) × coreNum      │                   │
└─────────────────────────────────┴──────────────────┘

workspace 总布局 (确定性):
┌─────────────────────────────────┬──────────────────┬──────────────────────────────┐
│  Cube MatMul 输出缓冲 (int32)    │  System RPC (20MB)│  确定性工作空间 (float)       │
│  parallNum × baseM × baseN      │                   │  64MB or 96MB                │
│  × sizeof(int32) × coreNum      │                   │  = mmQuantOutGm 指向区域     │
└─────────────────────────────────┴──────────────────┴──────────────────────────────┘
```

### 3.3 FillTilingData 传递参数

```cpp
// base_tiling.cpp:454-455
tilingData_.set_deterministicFlag(deterministicFlag_);     // 0 或 1
tilingData_.set_deterWorkspaceSize(deterWorkspaceSize_);   // 确定性工作空间字节数
```

---

## 4. Kernel 层：滑窗机制详解

### 4.1 窗口大小计算

```cpp
// grouped_matmul_finalize_routing.h:303 — Process() 函数内
syncConfig.windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(DTYPE_OUT));
syncConfig.lowBoundM = syncConfig.windowSize;  // 初始下界 = windowSize
```

**含义**：
- `deterWorkspaceSize`：确定性工作空间字节数 (64MB 或 96MB)
- `n`：输出矩阵的列数 (如 7168)
- `sizeof(DTYPE_OUT)`：`float` (combine=true) 时为 4，`bfloat16_t` 时为 2
- `windowSize`：窗口能容纳的**行数**

**数值示例 (n=7168)**：
```
deterWorkspaceSize = 96MB = 100,663,296 bytes
sizeof(float) = 4
windowSize = 96MB / (7168 × 4) = 100,663,296 / 28,672 ≈ 3511 行
```

### 4.2 滑窗数据结构 SyncConfig

```cpp
// grouped_matmul_finalize_routing_utils.h:49-56
struct SyncConfig {
    uint64_t curM = 0;          // 窗口中已处理的有效数据行数 (全局 M 坐标)
    uint64_t curGroup = 0;      // 窗口中已处理到的 group 索引
    uint64_t curGroupM = 0;     // curGroup 之前所有 group 的累积行数
    uint64_t lowBoundM = 0;     // 窗口下界 (窗口起始 M 坐标), 初始 = windowSize
    uint64_t windowSize = 0;    // 窗口最大行容量
    uint64_t baseN = 0;         // flush 时的 baseN (对齐到 128 的 N 分块)
};
```

**滑窗概念图**：
```
全局 M 维度 (所有 group 的所有 token 展开)
──────────────────────────────────────────────────────────────────────▶

Group 0         Group 1         Group 2              Group N-1
[M0 rows]       [M1 rows]       [M2 rows]             [...]
█████████████████████████████████████████████████████████████████████

滑窗视角:
                    lowBoundM - windowSize        lowBoundM
                         │                           │
    [已flush回写]        │     [窗口中: 已计算完成]    │   [尚未计算]
◀───────────────────────┼───────────────────────────┼────────────────▶
                         │◀────── windowSize ──────▶│
                         │   = 窗口中最多容纳的行数   │
                         │                           │
                       curM (已计算的全局行数, ≤ lowBoundM)

当 curBlockM (当前 tile 的上界) > lowBoundM 时 → 触发窗口滑动
```

### 4.3 Process 主循环：滑窗触发点

```cpp
// grouped_matmul_finalize_routing.h:290-335
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::Process()
{
    if ASCEND_IS_AIV {
        PreProcess();   // 初始化 output + 处理 sharedInput
        SyncAll();      // 初始全局同步
    }
    MNConfig mnConfig;
    SyncConfig syncConfig;
    mnConfig.baseM = tiling->matmulTiling.baseM;
    mnConfig.baseN = tiling->matmulTiling.baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);

    // ★ 确定性窗口大小
    syncConfig.windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(DTYPE_OUT));
    syncConfig.lowBoundM = syncConfig.windowSize;

    uint64_t nTimes = Ceil(tiling->n, DETER_UB_SIZE / sizeof(DTYPE_OUT));
    syncConfig.baseN = Ceil(Ceil(tiling->n, nTimes), 128) * 128;

    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        uint32_t m = static_cast<uint32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) continue;
        // ...设置 mnConfig ...

        while (curBlock < curCount) {
            MNBlockIdxCompute(mnConfig, curBlock, preCount, thresholdMDimN,
                              tiling->deterministicFlag);  // ★ 确定性时强制行优先
            MMCompute(groupIdx, mnConfig);
            VectorSync(mnConfig, syncConfig);      // ★ 滑窗检查 + flush
            VectorCompute(groupIdx, mnConfig, syncConfig);
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
    // ★ 最终收尾 flush
    if (tiling->deterministicFlag == 1) {
        syncConfig.curM = mnConfig.offsetM;
        FRDeterministic(syncConfig);
    }
}
```

### 4.4 MNBlockIdxCompute：确定性强制行优先调度

```cpp
// grouped_matmul_finalize_routing_utils.h:140-161
__aicore__ inline void MNBlockIdxCompute(MNConfig& mnConfig, const uint32_t curBlock,
                                         const uint32_t count, const uint32_t thresholdMDimN,
                                         const uint32_t deterministicFlag)
{
    // ★ 确定性路径或小 M → 强制行优先(row-priority)策略
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1 || deterministicFlag == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;   // 行索引 = block序号 / N块数
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;   // 列索引 = block序号 % N块数
    } else {
        // 非确定性大 M → 8×8 diagonal 策略 (优化 L2 缓存命中)
        // ...复杂的 diagonal 映射逻辑...
    }
}
```

**为什么确定性要强制行优先？** 滑窗需要按行号顺序知道哪些行已完成计算。Diagonal 策略会让不同核以非行序方式处理 tile，导致无法判断窗口是否连续完成。

```
行优先策略 (确定性):                   Diagonal 策略 (非确定性, 大 M):
  Core 0  Core 1  Core 2  Core 3         Core 0  Core 1  Core 2  Core 3
M0 [0,0]  [0,1]  [0,2]  [0,3]        M0 [0,0]                    [0,3]
M1 [1,0]  [1,1]  [1,2]  [1,3]        M1        [1,1]      [1,2]
M2 [2,0]  [2,1]  [2,2]  [2,3]        M2               [2,2]
  ...                                    ...
  按行顺序 → 窗口可连续推进              行不连续 → 无法确定窗口边界
```

---

## 5. 阶段1: 计算写入确定性工作空间

### 5.1 Init：mmQuantOutGm 指针计算

```cpp
// grouped_matmul_finalize_routing.h:162-166
if (tiling->deterministicFlag == 1) {
    mmQuantOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(
        initParams.workspace + tiling->parallNum * tiling->matmulTiling.baseM *
                                   tiling->matmulTiling.baseN *
                                   sizeof(int32_t) * tiling->coreNum));
    //                      ↑ 跳过 Cube MatMul workspace + System RPC 区域
}
```

`mmQuantOutGm` 指向确定性工作空间的起始地址，按 token 线性顺序存储：

```
mmQuantOutGm 布局 (确定性工作空间):
┌──────────┬──────────┬──────────┬──────────┬──────────┬─────┐
│ Token 0  │ Token 1  │ Token 2  │ Token 3  │ Token 4  │ ... │
│ [0..n-1] │ [0..n-1] │ [0..n-1] │ [0..n-1] │ [0..n-1] │     │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────┘
  ← n × sizeof(float) →  每行连续存储，按 token 在 group 中的顺序排列
```

### 5.2 VectorAtomicProcess：从 AtomicAdd Scatter 到线性写

```cpp
// grouped_matmul_finalize_routing.h:377-402
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorAtomicProcess(
    const VectorAtomicParams& vecAParams, const SyncConfig& syncConfig)
{
    LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
    if constexpr (P::combine) {
        if (tiling->deterministicFlag == 1) {
            // ★ 确定性路径: 写入确定性工作空间 (线性写, 无 AtomicAdd)
            DataCopy2DDimParams dimParams{vecAParams.curVecBaseM, vecAParams.curVecBaseN,
                                           vecAParams.alignBaseN};
            // 关键地址计算: 当前窗口内的相对偏移
            // lowBoundM - windowSize = 窗口起始行号
            DataCopyPad2D(mmQuantOutGm[vecAParams.yGmOffset1 -
                (syncConfig.lowBoundM - syncConfig.windowSize) * tiling->n],
                yLocal, dimParams, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
            return;
        }
        // ★ 非确定性路径: AtomicAdd scatter 到 yGm
        SetAtomicAdd<float>();
        DataCopyExtParams paramsOut{1, static_cast<uint32_t>(vecAParams.curVecBaseN * sizeof(float)), 1, 1, 0};
        for (uint32_t i = 0; i < vecAParams.curVecBaseM; i++) {
            auto outRow = static_cast<uint64_t>(
                tokenRanksGm.GetValue(vecAParams.mGlobalOffset + vecAParams.offsetM + i));
            DataCopyPad(yGm[outRow * tiling->n + vecAParams.yGmOffset0],
                        yLocal[i * vecAParams.alignBaseN], paramsOut);
        }
        SetAtomicNone();
    } else {
        // non-combine 路径...
    }
    vecOutQueue.FreeTensor(yLocal);
}
```

**对比分析**：

| 特性 | 非确定性 | 确定性 |
|------|---------|--------|
| 写目标 | `yGm[outRow × n + col]` | `mmQuantOutGm[(globalM - windowStart) × n + col]` |
| 寻址方式 | 按输出行号散列 (scatter) | 按 token 顺序线性 (linear) |
| AtomicAdd | 需要 (多核可能写同行) | 不需要 (每个 token 独占一行) |
| 搬运方式 | 逐行 DataCopyPad | 整块 DataCopyPad2D |
| 路由信息 | 写时查询 tokenRanksGm | 暂不查询, flush 时再查 |

### 5.3 UB 缓冲变化：queBind

```cpp
// grouped_matmul_finalize_routing.h:184-188 (InitUbBuffer)
if (tiling->deterministicFlag == 1) {
    pipe->InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);  // DETER_UB_SIZE = 12 * 1024 = 12KB
}
```

```cpp
// grouped_matmul_finalize_routing.h:121
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;  // VECIN/VECOUT 绑定队列
```

`queBind` 是一个 **VECIN + VECOUT 绑定队列**，用于 FRDeterministic flush 时的高效数据搬运。它将输入和输出队列绑定在一起，实现 Ping/Pong 双缓冲的搬入→搬出流水。

**UB 布局对比**：
```
非确定性 UB 布局:
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┐
│scaleInQ  │biasInQ   │pertokenQ │vecInQ    │vecOutQ   │ tmpBuff      │
│(2×buf)   │(2×buf)   │(2×buf)   │(2×buf)   │(2×buf)   │ dequant+...  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘

确定性 UB 布局 (+queBind):
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┐
│scaleInQ  │biasInQ   │pertokenQ │queBind   │vecInQ    │vecOutQ   │ tmpBuff      │
│(2×buf)   │(2×buf)   │(2×buf)   │(2×12KB)  │(2×buf)   │(2×buf)   │ dequant+...  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
                                 ▲
                         新增 12KB (DETER_UB_SIZE)
```

---

## 6. 阶段2: 滑窗触发与批量回写

### 6.1 VectorSync：滑窗检查逻辑

```cpp
// grouped_matmul_finalize_routing.h:622-650
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorSync(MNConfig& mnConfig, SyncConfig& syncConfig)
{
    if ASCEND_IS_AIC {
        return;  // AIC 不参与
    }
    if (tiling->deterministicFlag == 0) {
        return;  // 非确定性: 跳过
    }
    // ★ 核心判断: 当前 tile 的上界是否超出窗口下界?
    while (mnConfig.curBlockM > syncConfig.lowBoundM) {
        // 推进 curGroup 和 curM, 直到到达 lowBoundM
        while (syncConfig.curGroup < tiling->groupNum) {
            uint32_t mi = static_cast<uint32_t>(groupTokensGm.GetValue(syncConfig.curGroup));
            if constexpr (P::groupListType) {
                if (syncConfig.curGroup > 0) {
                    mi -= static_cast<uint32_t>(groupTokensGm.GetValue(syncConfig.curGroup - 1));
                }
            }
            if (syncConfig.curGroupM + mi <= syncConfig.lowBoundM) {
                // 当前 group 完全在窗口内
                syncConfig.curGroupM += mi;
                syncConfig.curM = syncConfig.curGroupM;
                syncConfig.curGroup++;
            } else {
                // 当前 group 跨越窗口边界, 对齐到 singleM
                syncConfig.curM += (syncConfig.lowBoundM - syncConfig.curM)
                                   / mnConfig.singleM * mnConfig.singleM;
                break;
            }
        }
        // ★ 执行窗口 flush
        FRDeterministic(syncConfig);
        // ★ 窗口下界前移
        syncConfig.lowBoundM = syncConfig.curM + syncConfig.windowSize;
    }
}
```

**滑窗推进示意图**：
```
初始状态:
  lowBoundM = windowSize (如 3511)
  curM = 0, curGroup = 0

每当 curBlockM (当前计算的 tile 上界) > lowBoundM 时:

  Step 1: 推进 curM 到 lowBoundM
          ┌──────────────────────────┐
          │  G0 (M=2048)  │ G1 (M=512)│...    curGroup=1, curGroupM=2048
          │ 全部在窗内     │部分在窗内  │       curM = lowBoundM = 3511
          └──────────────────────────┘
                   curM=3511 (对齐到 singleM 的倍数, 如 128 对齐 → 3456)

  Step 2: FRDeterministic(syncConfig) → flush 窗口 [0, curM) 到 yGm

  Step 3: lowBoundM = curM + windowSize
           即 lowBoundM = 3511 + 3511 = 7022

  Step 4: 继续计算, 直到 curBlockM 再次超出新的 lowBoundM
```

### 6.2 FRDeterministic：核心回写函数

```cpp
// grouped_matmul_finalize_routing.h:652-687
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::FRDeterministic(SyncConfig& syncConfig)
{
    if ASCEND_IS_AIC {
        return;  // AIC 不参与
    }
    SyncAll();  // ★ 全局同步: 等所有核完成当前窗口内的计算

    // 窗口中实际需要 flush 的行数
    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = tiling->coreNum * GetTaskRation();  // AIV 总子核数
    uint64_t n = tiling->n;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        // ★ 读取路由信息: 第 mOffset 个 token 路由到哪个输出行
        auto outRow = static_cast<uint64_t>(
            tokenRanksGm.GetValue((syncConfig.lowBoundM - syncConfig.windowSize) + mOffset));

        // ★ 按行号取模分配核: 每行只由一个核负责
        if (outRow % coreNumVec != GetBlockIdx()) {
            continue;  // 跳过不属于本核的行
        }

        // 分段搬运 N 方向
        uint64_t curVecBaseN = syncConfig.baseN;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            if (nOffset + syncConfig.baseN >= n) {
                curVecBaseN = n - nOffset;  // 尾块处理
            }
            DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(float)), 0, 0, 0};
            DataCopy2DDimParams copyDimParams{static_cast<uint32_t>(1),
                                              static_cast<uint32_t>(curVecBaseN),
                                              static_cast<uint32_t>(curVecBaseN)};

            // 使用 queBind 进行 Ping/Pong 搬运
            LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
            DataCopyPad2D(bindLocal,
                          mmQuantOutGm[mOffset * n + nOffset],  // 源: 确定性工作空间
                          copyDimParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // ★ 目标: yGm 正式输出 (AtomicAdd 确保同行多 token 正确累加)
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyPad(yGm[outRow * tiling->n + nOffset], bindLocal, paramsOut);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();  // ★ 全局同步: flush 完成, 所有核可以继续下一轮计算
}
```

### 6.3 FRDeterministic 数据流图

```
FRDeterministic 单次 flush 流程:

                    SyncAll() ←── 所有 AIV 核在此同步
                         │
┌────────────────────────┼────────────────────────┐
│  Core 0               │  Core 1               │  Core N-1
│  totalM = 窗口行数     │  totalM = 窗口行数     │  totalM = 窗口行数
│  for mOffset in [0..]: │  for mOffset in [0..]: │  for mOffset in [0..]:
│    outRow=tokenRanks[] │    outRow=tokenRanks[] │    outRow=tokenRanks[]
│    if outRow%N != 0:   │    if outRow%N != 1:   │    if outRow%N != N-1:
│      skip              │      skip              │      skip
│                        │                        │
│    处理本核负责的行      │    处理本核负责的行      │    处理本核负责的行
│    ┌──────────────┐    │    ┌──────────────┐    │    ┌──────────────┐
│    │queBind 搬运    │    │    │queBind 搬运    │    │    │queBind 搬运    │
│    │mmQuantOutGm   │    │    │mmQuantOutGm   │    │    │mmQuantOutGm   │
│    │   ↓           │    │    │   ↓           │    │    │   ↓           │
│    │AtomicAdd→yGm  │    │    │AtomicAdd→yGm  │    │    │AtomicAdd→yGm  │
│    └──────────────┘    │    └──────────────┘    │    └──────────────┘
└────────────────────────┼────────────────────────┘
                         │
                    SyncAll() ←── flush 完成
```

**为什么确定性路径仍然使用 AtomicAdd？** 因为在 MoE 场景中，同一窗口内可能有多个 token 路由到相同的输出行（不同 group 的不同 token 可能路由到同一行）。虽然每个输出行只由一个核负责，但该核可能需要将多个源 token 累加到同一目标行 → 需要 AtomicAdd 保证单核内的正确累加。但由于是单核顺序执行，累加顺序完全确定。

### 6.4 确定性关键保证

```
确定性成立的三要素:

1. SyncAll 全局屏障
   → 所有核在 flush 前同步，确保窗口内数据完整

2. outRow % coreNumVec 行分配
   → 每行只有一个核写入，消除跨核浮点累加顺序不确定性

3. 顺序窗口推进
   → 窗口按 group/token 顺序推进，全局顺序确定
   → 跨窗口的同行累加顺序唯一: 窗口1 → 窗口2 → ...
```

---

## 7. 完整同步时序对比

### 7.1 非确定性同步

```
时间 ──────────────────────────────────────────────────────▶

AIC Core 0:  │ MM tile 0  │──SetFlag(5)──│ Wait(3) │ MM tile 1 │...
AIC Core 1:  │ MM tile 1  │──────────────│SetFlag(5)│ Wait(3) │...
AIV Core 0:  │ Wait(5) │ Vec tile 0 │SetFlag(3)│ Wait(5) │ Vec tile 1 │...
AIV Core 1:  │ Wait(5) │ Vec tile 1 │──────────│SetFlag(3)│ Wait(5) │...

同步事件: mode=2 的 CrossCoreSetFlag/WaitFlag (事件号 3 和 5)
- 无全局 SyncAll
- 各核对独立推进，仅通过 tile 级别的 AIC↔AIV 配对同步
```

### 7.2 确定性同步

```
时间 ──────────────────────────────────────────────────────▶

AIC Core 0:  │ MM tile 0  │ SetFlag(5) │ Wait(3) │ MM tile 1 │...
AIC Core 1:  │ MM tile 1  │────────────│SetFlag(5)│ Wait(3) │...
AIV Core 0:  │SyncAll│ Wait(5) │ Vec tile 0 → mmQuantOutGm │ SetFlag(3) │
AIV Core 1:  │SyncAll│ Wait(5) │ Vec tile 1 → mmQuantOutGm │────────────│
             │       │                 ...                      │
   [滑窗触发: curBlockM > lowBoundM]                            │
             │       │         │                                  │
AIV ALL:     │       │    ┌────▼──────────────────────────┐      │
             │       │    │ SyncAll (等所有核完成窗口内计算)│      │
             │       │    │ FRDeterministic:               │      │
             │       │    │   逐行读 tokenRanksGm           │      │
             │       │    │   按 outRow%N 分配              │      │
             │       │    │   queBind: ws→UB→yGm (AtomicAdd)│      │
             │       │    │ SyncAll (flush 完成)            │      │
             │       │    └────┬──────────────────────────┘      │
             │       │         │   lowBoundM += windowSize        │
             │       │         │   继续计算...                     │
             │       │         ▼                                  │

额外同步: +2× SyncAll 每次窗口 flush (窗口首尾各一次)
         +CrossCoreSetFlag/WaitFlag 仍在每个 tile 内使用 (AIC↔AIV)
```

---

## 8. 滑窗数值模拟

### 8.1 场景设定

```
参数:
  n = 7168
  groupNum = 4
  M = [2048, 512, 1024, 2560]  (各 group 的 token 数)
  windowSize = 3511 (假设 96MB 工作空间)
  baseM = 128, baseN = 256
  coreNum = 8, taskRation = 2 → coreNumVec = 16
```

### 8.2 逐步骤模拟

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 0: 初始状态
  lowBoundM = 3511, curM = 0, curGroup = 0, curGroupM = 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1-16: Group 0 (M=2048, 2048/128=16 个 tile)
  curBlockM 从 128 → 1512 → ... → 2048 逐步增大
  始终 curBlockM ≤ 3511 (lowBoundM)
  → VectorSync 不触发 flush

  AIV 写入 mmQuantOutGm[0..2048*7168-1]  ← 2048 行

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 17-20: Group 1 (M=512, 4 个 tile)
  curBlockM 从 2176 → ... → 2560
  仍然 ≤ 3511, 不触发

  AIV 写入 mmQuantOutGm[2048*7168..2560*7168-1]  ← 512 行

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 21-28: Group 2 (M=1024, 8 个 tile)
  curBlockM 从 2688 → ... → 3328
  仍然 ≤ 3511, 不触发

  AIV 写入 mmQuantOutGm[2560*7168..3584*7168-1]  ← 1024 行

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 29: Group 2 最后几个 tile
  curBlockM = 3584 (Group 0+1+2 = 2048+512+1024 = 3584)
  curBlockM = 3584 > lowBoundM = 3511  ← ★ 触发!

  VectorSync:
    推进 curGroup: G0(2048) + G1(512) + G2 partial(951) = 3511
    → curM = 3456 (对齐到 singleM=128)
    → curGroup = 2, curGroupM = 2048+512 = 2560

  FRDeterministic:
    SyncAll()
    totalM = 3456 - (3511 - 3511) = 3456 行
    → 处理 mmQuantOutGm[0..3456*7168-1]
    → 逐行查 tokenRanksGm, 按 outRow%16 分配
    → AtomicAdd 写回 yGm
    SyncAll()

  lowBoundM = 3456 + 3511 = 6967

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 30-32: Group 2 剩余 (1024 - 951 = 73 行内)
  写入 mmQuantOutGm[(3456-2560)*7168..(3584-2560)*7168-1]
  ← 覆写到确定性工作空间开头 (窗口已释放)

  curBlockM ≤ 6967, 不触发

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 33-52: Group 3 (M=2560, 20 个 tile)
  curBlockM 从 3712 → ... → 6144
  始终 ≤ 6967, 不触发

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最终收尾:
  syncConfig.curM = totalM = 6144
  FRDeterministic:
    SyncAll()
    totalM = 6144 - (6967 - 3511) = 2688 行 (第二个窗口剩余)
    → 处理并写回
    SyncAll()
```

### 8.3 窗口空间复用

确定性工作空间的复用示意：

```
确定性工作空间 (96MB, ~3511 行):
┌──────────────────────────────────┬─────────────────────────────┐
│  窗口1: 行 0~3455                 │  窗口2: 行 3456~6143        │
│  (352 次 N=7168 float 搬运)       │  (264 次 N=7168 float)      │
│  ↓ flush 到 yGm                  │  ↓ flush 到 yGm             │
│  释放 → 复用空间                  │  释放                        │
└──────────────────────────────────┴─────────────────────────────┘
                                    ▲
                    窗口1 flush 完成后, 行3456开始的数据
                    覆写到工作空间开头, 实现循环复用
```

---

## 9. 总结

### 9.1 确定性机制五要素

```
┌──────────────────────────────────────────────────────────────────┐
│                     GMMFR W8A8 A3 确定性机制                        │
├────────────┬─────────────────────────────────────────────────────┤
│ ① 工作空间  │ 64/96MB L2 确定性工作空间, 存储中间结果 (float)       │
├────────────┼─────────────────────────────────────────────────────┤
│ ② 行优先   │ 强制 row-priority tile 调度, 保证窗口连续性            │
│   调度     │ MNBlockIdxCompute → (curBlock-count)/blockDimN       │
├────────────┼─────────────────────────────────────────────────────┤
│ ③ 线性写入  │ VectorAtomicProcess → mmQuantOutGm (线性地址)        │
│            │ 非确定性: yGm[outRow×n] (散列 AtomicAdd)               │
├────────────┼─────────────────────────────────────────────────────┤
│ ④ 滑窗触发  │ VectorSync: curBlockM > lowBoundM → 推进窗口         │
│            │ FRDeterministic: SyncAll → 逐行分配 → queBind 搬运     │
│            │ lowBoundM += windowSize → 窗口复用                     │
├────────────┼─────────────────────────────────────────────────────┤
│ ⑤ 全局同步  │ SyncAll 包围每次 flush (首尾各一次)                    │
│            │ outRow % coreNumVec 行分配 → 一核一行, 消除竞争        │
│            │ 单核内顺序 AtomicAdd → 累加顺序确定                      │
└────────────┴─────────────────────────────────────────────────────┘
```

### 9.2 代价与收益

| 维度 | 非确定性 | 确定性 | 增量 |
|------|---------|--------|------|
| **额外 Workspace** | 0 | 64~96 MB | +64~96 MB |
| **额外 UB** | 0 | 12 KB (queBind) | +12 KB |
| **额外同步** | 0 | 2×SyncAll / 窗口 | 与窗口数成正比 |
| **syncconfig 计算** | 无 | VectorSync 内 group 遍历 | O(groupNum) |
| **写回方式** | 即时 scatter | 延迟批量 flush | 增加延迟 |
| **调度策略** | Diagonal (大M) | Row-priority | 可能降低 L2 命中率 |
| **数值确定性** | ✗ | ✓ | — |

### 9.3 关键代码索引

| 功能 | 文件 | 行号 |
|------|------|------|
| Tiling: 确定性空间分配 | `base_tiling.cpp` | 413-425 |
| Tiling: 参数下发 | `base_tiling.cpp` | 454-455 |
| Kernel: mmQuantOutGm 初始化 | `grouped_matmul_finalize_routing.h` | 162-166 |
| Kernel: queBind 缓冲分配 | `grouped_matmul_finalize_routing.h` | 186-188 |
| Kernel: windowSize 计算 | `grouped_matmul_finalize_routing.h` | 303-306 |
| Kernel: Process 主循环 | `grouped_matmul_finalize_routing.h` | 290-335 |
| Kernel: VectorSync 滑窗检查 | `grouped_matmul_finalize_routing.h` | 622-650 |
| Kernel: FRDeterministic 回写 | `grouped_matmul_finalize_routing.h` | 652-687 |
| Kernel: VectorAtomicProcess 确定性分支 | `grouped_matmul_finalize_routing.h` | 381-386 |
| Kernel: MNBlockIdxCompute 行优先 | `grouped_matmul_finalize_routing_utils.h` | 143-144 |
| SyncConfig 结构体 | `grouped_matmul_finalize_routing_utils.h` | 49-56 |
| DETER_UB_SIZE 常量 | `grouped_matmul_finalize_routing_utils.h` | 30 |
| DETER_WORK_SPACE_SIZE 常量 | `base_tiling.cpp` | 27-28 |
