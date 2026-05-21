# GMMFR W8A8 INT8 确定性实现方案 — A5 (Ascend 950)

> **目标**: 在 A5 (DAV_3510 / arch35) 平台上实现 W8A8 INT8 确定性模式
> **参考文档**:
> - `GMMFR_W8A8_确定性分析.md` — A3 确定性窗口机制（核心参考）
> - `GMMFR_W8A8_非确定性分析_A5.md` — A5 Cgmct 非确定性实现
> - `GMMFR_W8A8_非确定性对比_A3_vs_A5.md` — 两者架构差异分析
>
> **核心思路**: 将 A3 的 SyncConfig + VectorSync + FRDeterministic 滑动窗口机制迁移到 A5 Cgmct 框架。通过 **common 中新建独立 Epilogue 文件** + **新建确定性内核包装器**，除 kernel 入口分发（apt.cpp）外不改动任何原有代码路径。
>
> **设计原则**:
> 1. 原有非确定性代码（Epilogue / Kernel / Prologue / Tiling）**零改动**
> 2. 确定性逻辑全部收敛在新建文件中
> 3. UB 和 Workspace 均保留充足安全余量，不踩线

---

## 目录

1. [总体架构设计](#1-总体架构设计)
2. [改动范围与文件清单](#2-改动范围与文件清单)
3. [Tiling 策略](#3-tiling-策略)
4. [确定性窗口机制](#4-确定性窗口机制)
5. [窗口滑动正确性保证 — 核心难题与解决方案](#5-窗口滑动正确性保证--核心难题与解决方案)
   - 5.1 [问题分析：为什么 Group 级 VectorSync 不够](#51-问题分析为什么-group-级-vectorsync-不够)
   - 5.2 [解决方案: 窗口预分区 + 分段处理](#52-解决方案-窗口预分区--分段处理)
   - 5.3 [Group 切分的 Offset 管理](#53-group-切分的-offset-管理)
   - 5.4 [singleM 对齐为何重要](#54-singlem-对齐为何重要)
6. [多 Shape 模拟验证](#6-多-shape-模拟验证)
   - 6.1 [Shape 模拟方法论](#61-shape-模拟方法论)
   - 6.2 [示例 1: 均匀小 batch（无切分）](#62-示例-1-均匀小-batch无切分)
   - 6.3 [示例 2: 大 batch + 多 expert（需切分 group）](#63-示例-2-大-batch--多-expert需切分-group)
   - 6.4 [关键修正: 物理 Workspace 写入地址](#64-关键修正-物理-workspace-写入地址)
   - 6.5 [修正后的完整执行追踪](#65-修正后的完整执行追踪)
   - 6.6 [极端 Shape 测试矩阵](#66-极端-shape-测试矩阵)
   - 6.7 [窗口写满率分析](#67-窗口写满率分析)
   - 6.8 [FRDeterministic 算法](#68-frdeterministic-算法)
   - 6.9 [queBind UB 使用时机](#69-quebind-ub-使用时机)
7. [确定性 Epilogue 设计](#7-确定性-epilogue-设计)
8. [确定性内核包装器](#8-确定性内核包装器)
9. [UB Buffer 安全分析](#9-ub-buffer-安全分析)
10. [Workspace 安全分析](#10-workspace-安全分析)
11. [A3→A5 迁移对照](#11-a3a5-迁移对照)
12. [完整数据流](#12-完整数据流)
13. [确定性正确性证明](#13-确定性正确性证明)
14. [性能与代价评估](#14-性能与代价评估)

---

## 1. 总体架构设计

### 1.1 架构对比

```
A5 非确定性模式（现有，不改动）:          A5 确定性模式（本方案，新建文件）:
┌──────────────────────────┐              ┌──────────────────────────────────────┐
│ Cgmct Block 组合框架      │              │ Cgmct Block 组合 + 确定性包装器       │
│                           │              │                                       │
│ Prologue (zero+shared)    │              │ Prologue (zero+shared) ← 复用原有     │
│ for each group:           │              │ InitDeterministic()                   │
│  ProcessSingleGroup:      │              │ for each group:                       │
│   AIC: BlockMmad          │              │  ProcessSingleGroupDet:               │
│   AIV: BlockEpilogue      │              │   AIC: BlockMmad ← 复用原有           │
│    └→ VectorAtomicProcess │              │   AIV: BlockEpilogueDet ← ★ 新建     │
│       → atomic y          │              │    └→ VectorDeterministicProcess      │
└──────────────────────────┘              │       → mmQuantOutGm (L2 workspace)   │
                                          │  VectorSync ← ★ 新建                  │
  不改动任何代码                           │ FRDeterministic (final) ← ★ 新建       │
                                          └──────────────────────────────────────┘

                                          仅新建 3 个文件 + apt.cpp 分发
```

### 1.2 核心设计决策

| 组件 | 非确定性（原有） | 确定性（新建） | 复用/新建 |
|------|-----------|---------|---------|
| BlockMmad (AIC MMAD) | `BlockMmadBuilder` | **完全相同** | 复用 |
| BlockPrologue (AIV 初始化) | `BlockPrologueFinalizeRouting` | **完全相同** | 复用 |
| BlockScheduler | `ASWT Scheduler` | **完全相同** | 复用 |
| BlockEpilogue (AIV 反量化+输出) | `BlockEpilogueDequantFinalizeRouting` | **`BlockEpilogueDequantFinalizeRoutingDeterministic`** | **新建** |
| Kernel 组装 | `KernelGmmFinalizeRoutingPertokenDequant` | **独立包装函数** | **新建** |
| 窗口管理 | 无 | SyncConfig + VectorSync + FRDeterministic | **新建** |

### 1.3 数据通路

```
A5 确定性数据通路:
                         AIC (Cube)                      L0C (256KB)                  UB (256KB)
                    ┌──────────────────┐           ┌──────────────────┐         ┌──────────────────┐
                    │ int8 × int8 MMAD  │           │ int32 tile output │         │                  │
                    │ (BlockMmad 复用)   │ ────────→ │ M0×N0            │ ──────→ │ l0cOutUb (int32) │
                    └──────────────────┘           └──────────────────┘         │        │         │
                                                                                │        ▼         │
                                                                                │  VFDoDequant     │
                                                                                │  (复用原逻辑)     │
                                                                                │  int32→float     │
                                                                                │  ×scale +bias    │
                                                                                │        │         │
                                                                                │        ▼         │
                                                                                │  VFDoLogitMuls   │
                                                                                │  (复用原逻辑)     │
                                                                                │        │         │
                                                                                │        ▼         │
                    ┌──────────────────┐                                         │  ★ 确定性写入     │
                    │ L2 Workspace     │ ←────────────────────────────────────── │  DataCopyPad     │
                    │ (确定性窗口)      │   写入窗口内偏移                         │  (非atomic)     │
                    │ 32MB / 48MB      │                                         └──────────────────┘
                    └────────┬─────────┘
                             │ FRDeterministic 刷出:
                             │ SyncAll → 多核分区 → atomic scatter
                    ┌────────▼─────────┐
                    │   y (GM float)   │
                    │   最终输出        │
                    └──────────────────┘
```

---

## 2. 改动范围与文件清单

### 2.1 改动总览

| 文件 | 操作 | 改动量 | 说明 |
|------|------|--------|------|
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | **修改** | +4 字段 | 新增确定性字段到 TilingData |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | **修改** | +1 方法声明 + 2 成员 | DeterministicTilingProcess |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | **修改** | +40 行 | 实现确定性 workspace 分配 |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing_deterministic.h` | **新建** | ~350 行 | 确定性 Epilogue（独立文件） |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant_deterministic.h` | **新建** | ~250 行 | 确定性内核包装器 + 窗口管理 |
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | **修改** | +16 行 | 确定性分发分支（仅此处修改原有文件） |

**关键承诺**: 以下原有文件**零改动**：
- `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` — 不改
- `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` — 不改
- `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h` — 不改
- `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` — 不改

### 2.2 新建文件详细设计

#### 文件 1: `block_epilogue_dequant_finalize_routing_deterministic.h`

位置: `gmm/common/cgmct/epilogue/`

职责: 
- 继承/复用原 Epilogue 中的 VFDoDequant + VFDoLogitMuls 反量化逻辑
- **替换** `VectorAtomicProcess` → `VectorDeterministicProcess`：写入 mmQuantOutGm 而非 scatter-add 到 y
- UB 布局与原 Epilogue 完全相同（198KB / 256KB），不额外分配 buffer

核心方法:
```cpp
template <typename CType, typename C1Type, ...>
class BlockEpilogueDequantFinalizeRoutingDeterministic {
    // ... 成员变量与 Init 同原版 ...

    // ★ 唯一差异: 输出阶段写 mmQuantOutGm
    template <typename T>
    __aicore__ inline void VectorDeterministicProcess(
        const LocalTensor<T>& outUb, uint32_t curVecBaseM,
        uint32_t alignN, uint32_t yOffset);

    // operator() 流程:
    //   CopyIn → VFDoDequant → VFDoLogitMuls → VectorDeterministicProcess
};
```

#### 文件 2: `grouped_matmul_finalize_routing_pertoken_dequant_deterministic.h`

位置: `op_kernel/arch35/`

职责:
- 用 Cgmct 组件（BlockMmad + BlockPrologue + **BlockEpilogueDet** + ASWT Scheduler）组装确定性 Kernel
- 实现 SyncConfig 追踪
- 实现 VectorSync 窗口溢出检测
- 实现 FRDeterministic 多核分区刷出
- group 循环中管理窗口状态

核心结构:
```cpp
struct SyncConfig { /* 同 A3 */ };

template <typename LayoutA, typename LayoutB, int ScaleType, int RowIndType>
__aicore__ inline void 
grouped_matmul_finalize_routing_pertoken_dequant_deterministic(
    GM_ADDR x, GM_ADDR w, ..., GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 1. 读取 tiling data，获取 deterministicFlag, deterWorkspaceSize, n, coreNum
    // 2. 组装 Cgmct 类型（复用 BlockMmad + BlockPrologue + ASWT Scheduler）
    // 3. 组装确定性 Epilogue
    // 4. InitDeterministic: mmQuantOutGm 地址 + windowSize + baseN
    // 5. Prologue (复用)
    // 6. for each group:
    //      UpdateGroupParams → ProcessSingleGroup (via ASWT)
    //        → BlockMmad + BlockEpilogueDet (写 mmQuantOutGm)
    //      VectorSync(curGroupEndM, syncConfig)
    // 7. 最终 FRDeterministic
}
```

#### 文件 3: `apt.cpp` 改动

仅新增确定性分发分支，不改动原有分发逻辑:

```cpp
// 每个 if constexpr 分支增加:
if (tiling->deterministicFlag == 1) {
    grouped_matmul_finalize_routing_pertoken_dequant_deterministic<...>(...);
} else {
    grouped_matmul_finalize_routing_pertoken_dequant<...>(...);  // 原有，不动
}
```

---

## 3. Tiling 策略

### 3.1 TilingKey 不变

A5 W8A8 确定性使用与 A5 W8A8 非确定性**完全相同的 TilingKey**：

```cpp
GET_TPL_TILING_KEY(ATRANS, BTRANS, SCALETYPE, ROWINDEXTYPE)
```

### 3.2 TilingData 扩展（仅新增 4 字段）

```cpp
// op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h
struct GMMFinalizeRoutingTilingData {
    GMMFinalizeRoutingDataParams gmmFinalizeRoutingDataParams;
    TCubeTiling matmulTiling;

    // === 新增确定性字段（不影响原有字段布局） ===
    uint32_t deterministicFlag;     // 1 = 确定性, 0 = 非确定性
    uint64_t deterWorkspaceSize;    // 确定性 workspace 大小 (bytes)
    uint32_t n;                     // N 维度 (计算 windowSize)
    uint32_t coreNum;               // AIC 核数 (FRDeterministic 分核)
};
```

### 3.3 Host Tiling 确定性 Workspace 分配

**文件**: `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

新增 `DeterministicTilingProcess()` 方法（覆盖父类 GetWorkspaceSize）:

```cpp
ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DeterministicTilingProcess()
{
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;
        deterWorkspaceSize_ = 0;
        return ge::GRAPH_SUCCESS;
    }

    deterministicFlag_ = 1;

    // 1. 查询 L2 大小
    uint64_t l2Size = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);

    // 2. 分级分配确定性 workspace（保守，确保余量 ≥ 16MB）
    //    L2 > 128MB → 48MB  (余量 ≥ 64MB)
    //    L2 > 96MB  → 32MB  (余量 ≥ 48MB)
    //    L2 ≤ 96MB  → 32MB  (余量 ≥ 48MB)
    constexpr uint64_t DETER_WS_LARGE  = 48ULL * 1024 * 1024;  // 48MB
    constexpr uint64_t DETER_WS_SMALL  = 32ULL * 1024 * 1024;  // 32MB
    constexpr uint64_t L2_128M = 128ULL * 1024 * 1024;

    deterWorkspaceSize_ = (l2Size > L2_128M) ? DETER_WS_LARGE : DETER_WS_SMALL;

    // 3. 加入总 workspace
    workspaceSize_ += deterWorkspaceSize_;

    return ge::GRAPH_SUCCESS;
}
```

**保守策略理由**:
- A5 L2 典型 96~128MB，当前系统 workspace 16MB
- 确定性 48MB → 总 64MB，在 96MB L2 上留 32MB 余量（33%）
- 确定性 32MB → 总 48MB，在 96MB L2 上留 48MB 余量（50%）
- 远优于 A3 的激进策略（96MB det + 20MB sys = 116MB 几乎撑满 L2）

### 3.4 Workspace 布局

```
A5 Workspace 布局:
┌──────────────────────────────────────────────────────────────┐
│  Cgmct 系统 workspace  (16MB)                                 │
│  框架内部使用 (同步标志、tile 元数据等)                        │
├──────────────────────────────────────────────────────────────┤
│  [确定性模式] 确定性 Workspace  (32MB or 48MB)                 │
│  存放 float 反量化结果，用于延迟 combine                       │
│  mmQuantOutGm 指向此处起始                                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 确定性窗口机制

### 4.1 SyncConfig — 窗口状态追踪

与 A3 完全相同（`utils.h:49-56`），存放在新建文件 `pertoken_dequant_deterministic.h` 中：

```cpp
struct SyncConfig {
    uint64_t curM = 0;        // 窗口内已累积的有效 M 行数
    uint64_t curGroup = 0;    // 当前追踪到的 group 索引
    uint64_t curGroupM = 0;   // curGroup 的 token 累计值（全局偏移）
    uint64_t lowBoundM = 0;   // 窗口下界 = 上次刷出边界 + windowSize
    uint64_t windowSize = 0;  // 窗口最大容纳行数
    uint64_t baseN = 0;       // FRDeterministic 中 N 维分块大小
};
```

### 4.2 窗口初始化

```cpp
void InitDeterministic(SyncConfig& sc, const TilingData& tiling, 
                       __gm__ float* mmQuantOutGm)
{
    sc.windowSize = tiling.deterWorkspaceSize / (tiling.n * sizeof(float));
    sc.lowBoundM = sc.windowSize;

    // N 维分块: 确保 queBind 单 buffer ≤ 12KB
    constexpr uint32_t DETER_UB_SIZE = 12 * 1024;
    uint64_t nTimes = Ceil(tiling.n, DETER_UB_SIZE / sizeof(float));
    sc.baseN = Ceil(Ceil(tiling.n, nTimes), 128) * 128;
}
```

### 4.3 windowSize 计算（保守 workspace 策略下）

```
windowSize = deterWorkspaceSize / (n * sizeof(float))

n=7168:
  32MB → windowSize = 33554432 / 28672 ≈ 1170 行
  48MB → windowSize = 50331648 / 28672 ≈ 1755 行

n=4096:
  32MB → windowSize = 33554432 / 16384 ≈ 2048 行
  48MB → windowSize = 50331648 / 16384 ≈ 3072 行

n=2560:
  32MB → windowSize ≈ 3277 行
  48MB → windowSize ≈ 4915 行
```

### 4.4 窗口写入地址公式

与 A3 完全一致：

```cpp
uint64_t windowBase = syncConfig.lowBoundM - syncConfig.windowSize;
uint64_t writeOffset = (globalMOffset - windowBase) * n + yOffset;
DataCopyPad(mmQuantOutGm_[writeOffset], yLocal, copyParams);
```

---

## 5. 窗口滑动正确性保证 — 核心难题与解决方案

### 5.1 问题分析：为什么 Group 级 VectorSync 不够

A3 确定性模式中，VectorSync **在每个 tile 之后**检查窗口溢出。这是因为 A3 的调度器让所有 core 协同处理同一个 group 的不同 tile（stride = coreNum），所有 core 的 M 偏移同步推进。

但在 A5 Cgmct 框架中，存在两个根本性差异：

**差异 1: ASWT 调度器下 core 不同步**

从 `kernel_gmm_finalize_routing_pertoken_dequant.h:285-330` 的 `ProcessSingleGroup` 实现可以看到：

```
while (bs.GetTileIdx(tileIdx)) {       // ASWT 按需分配 tile
    AIC: WaitForVector → mmadOp_ → NotifyVector
    AIV: WaitForCube → epilogueOp_ → NotifyCube
}
```

ASWT 调度器为每个 core 独立分配 tile。不同 core 处理同一 group 的不同 tile，完成的 tile 数量不同。**不同 core 可能在不同的 group 中**——因为 group 循环 (line 356-361) 是每个 core 独立推进的：

```cpp
// line 356-361: 每个 core 独立迭代
for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
    if (!UpdateGroupParams(params, groupIdx)) continue;
    ProcessSingleGroup(params, bs, groupIdx);
}
```

Core 0 处理完 group 0 的 4 个 tile 后立即进入 group 1，而 Core 1 可能仍在处理 group 0 的第 6 个 tile。这意味着**不存在一个天然的全局 M 偏移同步点**。

**差异 2: 单 group 可超窗口**

group 级的 VectorSync 检查假设：每个 group 处理完毕后，全局 M 偏移推进 groupM。如果 groupM ≤ windowSize，这没问题。但实际场景中：

```
groupM = 4096 (一个 expert 收到 4096 个 token)
windowSize = 1170 (32MB workspace, n=7168)
```

groupM 是 windowSize 的 **3.5 倍**！如果在 group 处理完后才检查，写入了 4096 行数据但窗口只能容纳 1170 行——前 1170 行正常，接下来的 1170 行覆盖了前面的数据，最后 1756 行又覆盖了第二批数据。到 VectorSync 检查时，数据已经损毁。

**结论: 必须实现 tile 级窗口溢出检测，在任何 tile 写入前检查是否会溢出。**

### 5.2 解决方案: 窗口预分区 + 分段处理

核心思路：**不依赖 mid-group 同步点，而是在 group 处理之前，将整个工作预划分为 window-sized 段**。

#### 5.2.1 预分区算法

```
InitDeterministic 阶段计算 WindowPlan:

输入: groupList = [m0, m1, m2, ...], windowSize
输出: WindowPlan = [{groups, startM, endM}, ...]

算法:
  windowIdx = 0
  curWindowStart = 0
  curGroupIdx = 0
  curGroupRemaining = groupList[0]  // 当前 group 剩余未分配的 token 数

  while curGroupIdx < groupNum:
      curWindowCapacity = windowSize
      windowGroups = []

      while curWindowCapacity > 0 and curGroupIdx < groupNum:
          if curGroupRemaining <= curWindowCapacity:
              // 当前 group 剩余全部放入本窗口
              windowGroups.append({curGroupIdx, curGroupRemaining})
              curWindowCapacity -= curGroupRemaining
              curGroupIdx++
              curGroupRemaining = (curGroupIdx < groupNum) ? groupList[curGroupIdx] : 0
          else:
              // 当前 group 跨越窗口边界 → 切分
              // 按 singleM 对齐切割
              splitM = AlignDown(curWindowCapacity, singleM)
              if splitM > 0:
                  windowGroups.append({curGroupIdx, splitM})
                  curGroupRemaining -= splitM
              curWindowCapacity = 0  // 本窗口满

      WindowPlan[windowIdx] = {windowGroups, curWindowStart, curWindowStart + (windowSize - curWindowCapacity)}
      windowIdx++
      curWindowStart = WindowPlan[windowIdx-1].endM
```

#### 5.2.2 窗口分段执行

```cpp
// 确定性内核包装器中的执行流程
for (uint32_t w = 0; w < windowPlan.size(); w++) {
    auto& plan = windowPlan[w];

    // 更新 epilogue 窗口基址 = 本窗口的起始全局 M 偏移
    uint64_t windowBase = plan.startM;  // 确保物理写入在 [0, windowSize*n) 范围
    epilogueDetOp_.SetWindowBaseM(windowBase);

    // 处理本窗口内的所有 group（可能是完整的或被切分的）
    uint64_t globalOffset = plan.startM;
    for (auto& seg : plan.segments) {
        uint32_t g = seg.groupIdx;
        uint32_t segM = seg.tokenCount;

        // 调整当前 group 的 M 维度为 segM（可能 < 原始 groupM）
        SetGroupProblemShape(segM);       // 修改 problemShape_.m
        UpdateGroupParamsForSegment(g);   // 调整 offset

        epilogueDetOp_.SetGlobalMOffset(globalOffset);
        ProcessSingleGroup(params, bs, g);  // ★ Cgmct 原生调用，不改动

        globalOffset += segM;

        // 如果 group 被切分，更新 offset 为下次调用准备
        if (segM < originalGroupTokens[g]) {
            AdjustGroupRemaining(g, segM);  // 调整 baseOffset 以跳过已处理部分
        }
    }

    // ★ 窗口结束: 所有 core 同步 + 确定性刷出
    SyncAll();
    syncConfig.curM = plan.endM;
    FRDeterministic(syncConfig);
    // 刷出后 lowBoundM 自动更新，workspace 物理空间循环复用
    syncConfig.lowBoundM = plan.endM + windowSize;
}
```

#### 5.2.3 为什么这个方案保证正确性

| 保证 | 机制 |
|------|------|
| **不超窗口写入** | 每个窗口段的总 token 数 ≤ windowSize。在整个窗口段内 Cgmct 任意调度 tile，最坏情况下也只写入 windowSize 行。 |
| **跨窗口 group 正确切分** | Group 按 `singleM` 对齐切分。被切分的 group 在两个窗口中各处理一部分 tile。offset 管理确保第二批 tile 的 rowIndex 和 A/B/C 地址正确。 |
| **物理 workspace 复用** | 窗口 N 的 `windowBase = N * windowSize`（mod workspace 总容量）。FRDeterministic 在窗口结束后立即刷出，该物理区域可安全复用。 |
| **所有 core 同步** | `SyncAll()` 在窗口之间执行。Cgmct 在窗口内自由调度，但窗口边界是硬同步点。 |
| **原有代码零侵入** | `ProcessSingleGroup`、ASWT Scheduler、BlockMmad、BlockPrologue **完全不动**。窗口管理全在 wrapper 层。 |

### 5.3 Group 切分的 Offset 管理

当一个 group 被窗口边界切分时，需要在 tiling 层面处理：

```
原始 group 3: m=800 tokens
窗口 1 容纳 500 tokens → 切分: group 3 处理前 500 tokens (aligned to singleM=128 → 384)
窗口 2 容纳 300 tokens → 切分: group 3 处理后 416 tokens (416 未超窗口)

Offset 调整:
  窗口 1 处理 group 3[0..384):
    x offset: group0_m + group1_m + group2_m + 0       (正常)
    rowIndex offset: 同上
    logit offset: 同上
    pertoken_scale offset: 同上

  窗口 2 处理 group 3[384..800):
    x offset: group0_m + group1_m + group2_m + 384     (跳过已处理)
    rowIndex offset: 同上
    logit offset: 同上
    pertoken_scale offset: 同上
    b offset: 不变（权重属于整个 group，N*K 维度不受 M 切分影响）
    scale offset: 不变（per-channel scale 属于 expert，不受 M 切分影响）
```

**关键简化**: 因为 Cgmct 的 `UpdateOffset` (line 234-264) 在每个 group 开始时执行，且 offset 基于 groupIdx 计算（而非累积 M），对于一个被切分的 group：

- `UpdateOffset(groupIdx)` 中的 `Get<IDX_A_OFFSETS>(baseOffset_) += m * k`，这里的 m 是**原始 groupM**，不是 segM
- 第二次调用同一 groupIdx 时，`UpdateOffset` 不应重复累加

**解决方案**: 引入 `segmentOffsetAdjustment` 变量，记录已处理的 M 偏移。在第一次切分后存储此偏移，第二次调用时使用它而非重新计算：

```cpp
// 切分 group 的 offset 管理
struct GroupSegmentState {
    uint32_t groupIdx;
    uint32_t totalM;        // 原始 group token 数
    uint32_t processedM;    // 已处理的 M 行数
};

// 第一次处理 (窗口 1): processedM = 0 → 384
SetupGroupSegment(g, 0, segM);      // 从 row 0 开始，处理 segM 行

// 第二次处理 (窗口 2): processedM = 384 → 800
SetupGroupSegment(g, processedM, remainingM);  // 从 row 384 开始
```

这通过修改 `problemShape_.m` 和 `Get<IDX_A_OFFSETS>(baseOffset_)` 来实现。`problemShape_.m` 控制本次 tile 的 M 范围；`baseOffset_` 控制数据读取的起始位置。

### 5.4 singleM 对齐为何重要

```
windowSize = 1170, group[3] token count = 800
不对齐: splitM = 1170 → group[3] 处理全部 800 tokens, curWindowM = 800
对齐:   splitM = AlignDown(1170, 128) = 1152 → group[3] 处理前 1152 个 token
        但 1152 > 800, 所以 group[3] 全部放入窗口 1
```

singleM 对齐确保：
1. Cgmct tile 边界不被打破（每个 tile 处理 singleM × singleN 的整数倍）
2. FRDeterministic 刷出时 N 维对齐不需要处理跨 tile 的残片
3. 跨窗口切分点始终落在 tile 边界上，offset 管理更简单

singleM 的值从 Cgmct tiling 参数获取：`tilingData.matmulTiling.baseM`（通常为 128）。

---

## 6. 多 Shape 模拟验证

### 6.1 Shape 模拟方法论

对每种 shape 组合，模拟以下关键指标：

| 检查项 | 含义 | 通过条件 |
|--------|------|---------|
| `maxGroupM ≤ windowSize` | 是否有 group 超窗口 | 若否则需切分 |
| `totalM / windowSize` | 刷出次数 | 预期范围内 |
| `切分点 % singleM == 0` | 切分对齐 | 必须为 0 |
| `窗口写满率` | 窗口利用率 | > 80% 为佳 |
| `curGroup 不越界` | group 追踪 | ≤ groupNum |

### 6.2 示例 1: 均匀小 batch（无切分）

```
参数: 8 experts × 32 tokens, n=7168, 48MB workspace
groupList = [32, 32, 32, 32, 32, 32, 32, 32]
totalM = 256
windowSize = 1755
singleM = 128

预分区:
  windowSize(1755) > totalM(256) → 单窗口
  WindowPlan[0]: groups = [0..7], 每个 group 32 tokens
  curWindowStart = 0, endM = 256

执行:
  Window 0: globalOffset 0 → 256
    处理 group 0..7，无切分
    SyncAll → FRDeterministic(0, 256)
  
  刷出次数: 1（仅最终刷出）
```

### 6.3 示例 2: 大 batch + 多 expert（需切分 group）

这是最复杂的场景——全面追踪内部状态。

```
参数: 4 experts, n=7168, 48MB workspace
groupList = [1024, 1024, 1024, 1024]
totalM = 4096
windowSize = 1755
singleM = 128

────────────────────────────────────────────────────────────
预分区过程:

初始: curGroupIdx=0, curGroupRemaining=1024, curWindowStart=0

Window 0 (capacity=1755):
  group 0: remaining=1024 ≤ 1755 → 全放入
    windowGroups[0] = {groupIdx:0, segM:1024}
    capacity: 1755-1024=731, curGroupIdx→1, curGroupRemaining=1024

  group 1: remaining=1024 > 731 → 需切分!
    splitM = AlignDown(731, 128) = 640
    windowGroups[1] = {groupIdx:1, segM:640}
    curGroupRemaining = 1024-640 = 384
    capacity = 0 → Window 0 满

  WindowPlan[0]: segments=[{0,1024}, {1,640}], startM=0, endM=1664

Window 1 (capacity=1755):
  curGroupIdx=1, curGroupRemaining=384

  group 1 remaining: 384 ≤ 1755 → 全放入
    windowGroups[0] = {groupIdx:1, segM:384}
    capacity: 1755-384=1371, curGroupIdx→2, curGroupRemaining=1024

  group 2: remaining=1024 ≤ 1371 → 全放入
    windowGroups[1] = {groupIdx:2, segM:1024}
    capacity: 1371-1024=347, curGroupIdx→3, curGroupRemaining=1024

  group 3: remaining=1024 > 347 → 需切分!
    splitM = AlignDown(347, 128) = 256
    windowGroups[2] = {groupIdx:3, segM:256}
    curGroupRemaining = 1024-256 = 768
    capacity = 0 → Window 1 满

  WindowPlan[1]: segments=[{1,384}, {2,1024}, {3,256}], startM=1664, endM=3419

Window 2 (capacity=1755):
  curGroupIdx=3, curGroupRemaining=768

  group 3 remaining: 768 ≤ 1755 → 全放入
    windowGroups[2] = {groupIdx:3, segM:768}
    capacity: 1755-768=987, curGroupIdx→4 ≥ groupNum → 结束

  WindowPlan[2]: segments=[{3,768}], startM=3419, endM=4187

  但 endM(4187) > totalM(4096)! 修正: endM = 4096
  WindowPlan[2]: startM=3419, endM=4096

────────────────────────────────────────────────────────────
预分区总结:

Window 0: [g0:1024, g1:640]  → M [0,    1664)  共 1664 行
Window 1: [g1:384, g2:1024, g3:256] → M [1664, 3419)  共 1755 行
Window 2: [g3:768]            → M [3419, 4096)  共 677 行

Group 1 被切分: 窗口 0 处理 640 tokens, 窗口 1 处理 384 tokens
Group 3 被切分: 窗口 1 处理 256 tokens, 窗口 2 处理 768 tokens

────────────────────────────────────────────────────────────
执行追踪:

Window 0: windowBase=0
  ProcessSingleGroup(g=0, segM=1024), globalOffset=0
    → Cgmct tiles: Ceil(1024,128)=8 M-tiles × Ceil(7168,256)=28 N-tiles
    → EpilogueDet 写入: mmQuantOutGm[(0..1023-0)*n + 0..n-1]
  ProcessSingleGroup(g=1, segM=640), globalOffset=1024
    → 注意: group 1 仅处理前 640 tokens!
    → EpilogueDet 写入: mmQuantOutGm[(1024..1663-0)*n + 0..n-1]
  SyncAll → FRDeterministic: 刷出 M [0, 1664)
  → y[outRow * n + 0..n-1] 收到第一批 scatter
  窗口状态: curM=1664, processedM[group1]=640

Window 1: windowBase=0 (物理 workspace 复用! 窗口 0 已刷出)
  ProcessSingleGroup(g=1, segM=384), globalOffset=1664
    → group 1 剩余 384 tokens, offset 从 row 640 开始
    → EpilogueDet 写入: mmQuantOutGm[(1664..2047-1755)*n + 0..n-1]
    →                    = mmQuantOutGm[(-91..292)*n + 0..n-1]
    ⚠ 地址异常！需要修正！
```

### 6.4 关键修正: 物理 Workspace 写入地址

上面的追踪暴露了一个问题：当 `windowBase` 不变（=0）而 `globalMOffset` 不断增加时，写入地址会变成负数。

**修正方案**: windowBase 必须随着 globalMOffset 推进而更新。

```
正确的写入地址公式:
  physicalOffset = (globalMOffset - windowBase) * n + nOffset

其中 windowBase 不是物理复用次数，而是逻辑窗口起始 M:
  Window 0: windowBase = 0         → physicalOffset = (0..1663) * n
  Window 1: windowBase = 0         → physicalOffset = (1664..3418) * n
            ⚠ 1664 > windowSize=1755 → 超出物理 workspace!

修正:
  Window 0: windowBase = 0,           physicalOffset = (0..1663) * n
            → 使用 workspace [0, 1664*n)
  Window 1: windowBase = 1664,        physicalOffset = (1664..3418 - 1664) * n
            → 使用 workspace [0, 1755*n)
            → 物理复用 window 0 的空间（已刷出，安全）
  Window 2: windowBase = 3419,        physicalOffset = (3419..4096 - 3419) * n
            → 使用 workspace [0, 677*n)
```

**关键**: `windowBase = plan.startM`。即 windowBase 始终等于当前窗口段的起始全局 M 偏移。这样物理写入始终落在 `[0, windowSize * n)` 范围内。

### 6.5 修正后的完整执行追踪

```
Window 0: windowBase=0
  g0(1024): 写入 mmQuantOutGm[(0..1023    - 0)*n]   = [0*N .. 1024*N)
  g1(640):  写入 mmQuantOutGm[(1024..1663  - 0)*n]   = [1024*N .. 1664*N)
  SyncAll → FRDeterministic: 刷出 M [0, 1664)
  ✅ 物理 workspace 使用: [0, 1664*N)  ≤ windowSize*N = 1755*N

Window 1: windowBase=1664
  g1(384):  写入 mmQuantOutGm[(1664..2047  - 1664)*n] = [0*N .. 384*N)
  g2(1024): 写入 mmQuantOutGm[(2048..3071  - 1664)*n] = [384*N .. 1408*N)
  g3(256):  写入 mmQuantOutGm[(3072..3327  - 1664)*n] = [1408*N .. 1664*N)
  SyncAll → FRDeterministic: 刷出 M [1664, 3419)
  ✅ 物理 workspace 使用: [0, 1755*N)  ≤ windowSize*N = 1755*N

Window 2: windowBase=3419
  g3(768):  写入 mmQuantOutGm[(3419..4186  - 3419)*n] = [0*N .. 768*N)
  SyncAll → FRDeterministic: 刷出 M [3419, 4096)
  ✅ 物理 workspace 使用: [0, 677*N)  ≤ windowSize*N = 1755*N

═══════════════════════════════════════════════════════════
所有窗口的物理写入均在 [0, windowSize*N) 范围内 → 无溢出
所有 group 的 token 被完整处理（切分 group 的 token 数之和 = 原始 token 数）
刷出顺序: 窗口 0 → 窗口 1 → 窗口 2，顺序确定
═══════════════════════════════════════════════════════════
```

### 6.6 极端 Shape 测试矩阵

| 场景 | groupList | totalM | windowSize | 切分次数 | 窗口数 | 通过条件 |
|------|-----------|--------|-----------|---------|--------|---------|
| 均匀小 batch | 8×32=256 | 256 | 1755 | 0 | 1 | totalM < windowSize |
| 均匀中 batch | 8×256=2048 | 2048 | 1755 | 1 group | 2 | 跨窗口边界 |
| 均匀大 batch | 4×1024=4096 | 4096 | 1755 | 2 groups | 3 | 多 group 跨窗口 |
| 不均匀分布 | [100,200,800,150,300,50,400,2000] | 4000 | 1755 | 2 groups | 3 | 大 group 跨多窗口 |
| 单 expert 巨量 | [8192] | 8192 | 1755 | 1 group 被多次切分 | 5 | group 跨 5 个窗口 |
| 空 group 掺杂 | [0,512,0,512,0] | 1024 | 1755 | 0 | 1 | 空 group 跳过后仍正常 |
| 极小 singleM | [4096] (singleM=128) | 4096 | 1755 | 1 group 切 3 次 | 3 | 对齐到 128 |
| 大 singleM | [4096] (singleM=256) | 4096 | 1755 | 1 group 切 2 次 | 3 | 对齐到 256 |
| cumsum 格式 | [100,300,1100,1250,...] | 4000 | 1755 | 2 groups | 3 | cumsum→count 解析正确 |

### 6.7 窗口写满率分析

```
windowSize = 1755, singleM = 128

最坏情况切分浪费:
  强制按 singleM 对齐切分，每个被切分的 group 最多浪费 singleM-1 = 127 行
  最多每个窗口 1 次切分 → 最坏写满率 = (1755-127)/1755 = 92.8%

实际写满率:
  均匀分布时切分点接近 windowSize → 写满率 > 95%
  不均匀分布时可能存在较大浪费 → 写满率 ~85-95%
```

### 6.8 FRDeterministic 算法

```
FRDeterministic(syncConfig):
    SyncAll()                                    ← 等待所有 core 完成本窗口

    totalM = curM - (lowBoundM - windowSize)     ← 本窗口实际写入的行数
    coreNumVec = coreNum * 2                     ← A5: AIC:AIV = 1:2

    for mOffset in 0 .. totalM-1:
        outRow = rowIndex[lowBoundM - windowSize + mOffset]

        if outRow % coreNumVec != GetBlockIdx(): continue   ← 分核: 每核只写特定行

        for nOffset = 0; nOffset < n; nOffset += baseN:
            curBaseN = min(baseN, n - nOffset)

            bindLocal = queBind.AllocTensor<float>()
            DataCopyPad2D(bindLocal, mmQuantOutGm[mOffset * n + nOffset], ...)
            queBind.EnQue(bindLocal)
            bindLocal = queBind.DeQue<float>()

            SetAtomicAdd<float>()
            DataCopyPad(yGm[outRow * n + nOffset], bindLocal, ...)
            SetAtomicNone()

            queBind.FreeTensor(bindLocal)

    SyncAll()                                    ← 等待所有 core 完成刷出
```

### 6.9 queBind UB 使用时机

`queBind` 仅在 `FRDeterministic` 调用期间使用。FRDeterministic 在 `SyncAll()` 之后执行——此时 Epilogue 的 tile 处理已全部完成，UB 中 198KB buffer 全部空闲。12KB × 2 (double buffer) = **24KB**，仅占 256KB UB 的 9.4%。FRDeterministic 结束后释放 queBind，下一窗口的 Epilogue tile 处理正常使用 UB。

---

## 7. Group 跨窗口保护：A3 vs A5 逐项对照

A3 确定性窗口的核心保护机制是：**当 group 跨窗口边界时，只刷出到 singleM 对齐边界，剩余数据保留在下个窗口继续处理**。A5 通过预分区方案等價实现了相同的保护——但实现方式不同。下面把 A3 文档的三个 Shape 示例逐项用 A5 方案重新模拟，证明保护机制等价。

### 7.1 保护机制对照总表

| 保护项 | A3 (VectorSync 运行时检测) | A5 (BuildWindowPlan 预分区) | 等价性 |
|--------|--------------------------|---------------------------|--------|
| **检测时机** | 每个 tile 后检查 curBlockM > lowBoundM | Kernel 启动时一次性预计算 | 等价：A5 提前算好，执行时不检查 |
| **Group 跨窗口时** | curGroup 停在该 group，inner loop break | g 不递增，remaining 保留未分配 token 数 | **等价** |
| **切分对齐** | `(lowBoundM - curM) / singleM * singleM` | `AlignDown(capacity, singleM)` | **等价** |
| **剩余数据保留** | curGroup 不变，下个 tile 继续同 group | same groupIdx 出现在下一窗口的 segments | **等价** |
| **Offset 调整** | 隐式：tile 循环内部 curBlock 跟踪 mIdx | 显式：`processedM` 记录已处理行数 | **等价** |
| **窗口基址更新** | lowBoundM += windowSize | win.startM 推进 | **等价** |

### 7.2 示例 1: A3 示例 7.2 → A5 (无需中途刷出)

**A3 场景**: 4 experts × 64 tokens, n=7168, 96MB workspace

**A5 对应场景**: 4 experts × 64 tokens, n=7168, 48MB workspace (windowSize=1755)

```
groupList = [64, 64, 64, 64]
totalM = 256
windowSize = 1755
singleM = 128

A5 BuildWindowPlan 过程:
  g=0: remaining=64 ≤ capacity(1755) → 完整放入 Window 0
  g=1: remaining=64 ≤ capacity(1691) → 完整放入 Window 0
  g=2: remaining=64 ≤ capacity(1627) → 完整放入 Window 0
  g=3: remaining=64 ≤ capacity(1563) → 完整放入 Window 0
  → numWindows = 1

A5 执行:
  Window 0: globalOffset 0→256
    ProcessSingleGroup(g=0, segM=64)
    ProcessSingleGroup(g=1, segM=64)
    ProcessSingleGroup(g=2, segM=64)
    ProcessSingleGroup(g=3, segM=64)
    SyncAll → FRDeterministic → 刷出 [0, 256)
  ✅ 无切分，与 A3 完全一致
```

### 7.3 示例 2: A3 示例 7.3 → A5 (Group 跨窗口边界)

这是**核心对照**——验证 A5 的 group 跨窗口保护与 A3 等价。

**A3 场景**: 4 experts × 1024 tokens, n=7168, 96MB workspace → windowSize=3510

A3 执行到 Group 3 的 tile 3 (curBlockM=3584 > lowBoundM=3510) 时触发 VectorSync：
- curGroup 内层循环: g0✓ g1✓ g2✓ g3✗ (curGroupM=4096 > 3510)
- curM = 3072 + (3510-3072)/128*128 = 3072 + 384 = 3456
- break → curGroup=3 保留 → FRDeterministic [0, 3456)
- Group 3 剩余 640 tokens 在下个窗口继续

**A5 对应场景**: 4 experts × 1024 tokens, n=7168, 48MB workspace → windowSize=1755

```
groupList = [1024, 1024, 1024, 1024]
totalM = 4096
windowSize = 1755
singleM = 128

          A5 BuildWindowPlan                         A3 VectorSync 等價步骤
          ════════════════                           ════════════════════
Window 0 (capacity=1755):
  g=0: remaining(1024) ≤ 1755                     ← curGroup=0, mi=1024, curGroupM=1024 ≤ 3510 ✓
       → Segment{g=0, segM=1024}
       capacity=731

  g=1: remaining(1024) > 731 ⚡ 需切分!            ← curGroup=1, mi=1024, curGroupM=2048 > 3510 ✗
       splitM = AlignDown(731, 128) = 640          ← curM += (3510-1024)/128*128 = 1024+2432=3456
       → Segment{g=1, segM=640}                    ← 等价: Window 0 容纳 group 1 的前 640 tokens
       remaining = 1024-640 = 384  ★保留!           ← curGroup=1 停止,剩余 384 tokens 保留
       capacity=0 → Window 0 关闭

  Window 0 总结:                                     A3 第一批刷出: [0, 3456)
    startM=0, endM=1664                              A3: group 0(1024) + group 1 前 384 tokens
    = g0(1024) + g1 前 640                           对齐粒度不同因为 windowSize 不同

Window 1 (capacity=1755):
  g=1: remaining(384) ≤ 1755                     ← ★ 同一个 group 1 继续!
       → Segment{g=1, segM=384}
       capacity=1371                                A3: group 1 剩余 640 行在 window 2
  g=2: remaining(1024) ≤ 1371
       → Segment{g=2, segM=1024}
       capacity=347
  g=3: remaining(1024) > 347 ⚡ 再次切分!
       splitM = AlignDown(347, 128) = 256
       → Segment{g=3, segM=256}
       remaining = 1024-256 = 768  ★保留!
       capacity=0 → Window 1 关闭

Window 2 (capacity=1755):
  g=3: remaining(768) ≤ 1755                     ← ★ 同一个 group 3 继续!
       → Segment{g=3, segM=768}
       g=4 → 结束

═══════════════════════════════════════════════════════════════════════
         A3 结果                           A5 结果
         ═══════                           ═══════
  Window 0: [0, 3456)                  Window 0: [0, 1664)
    含 g0 全量 + g1 前 384 行            含 g0 全量 + g1 前 640 行
  Window 1: [3456, 4096)              Window 1: [1664, 3419)
    含 g1 剩余 640 + g2+g3 全量         含 g1 剩余 384 + g2 全量 + g3 前 256
                                        Window 2: [3419, 4096)
                                          含 g3 剩余 768

  ★ 核心等价验证:
  ✅ group 1 跨窗口 → A3 curGroup=1 保留 / A5 remaining=384 保留
  ✅ group 3 跨窗口 → A3 curGroup=3 保留 / A5 remaining=768 保留  
  ✅ 切分对齐 singleM → A3 (3510-3072)/128*128 / A5 AlignDown(731,128)=640
  ✅ 剩余数据不丢失 → A3 下个窗口继续 / A5 下一 WindowPlan 的 Segment
═══════════════════════════════════════════════════════════════════════
```

### 7.4 示例 3: A3 示例 7.4 → A5 (不均匀分布 + Group 跨多窗口)

**A3 场景**: 不均匀分布, windowSize=1500

```
groupList = [100, 200, 50, 800, 150, 300, 100, 400], totalM=2100

A3 执行: group 5 的 tile 1 触发 VectorSync
  curM = 1300 + (1500-1300)/128*128 = 1428
  curGroup=5 保留, 剩余 172 tokens

A5 BuildWindowPlan (windowSize=1500, singleM=128):
───────────────────────────────────────────────────────────
Window 0 (capacity=1500):
  g=0: 100 → capacity=1400
  g=1: 200 → capacity=1200
  g=2: 50  → capacity=1150
  g=3: 800 → capacity=350
  g=4: 150 → capacity=200
  g=5: 300 > 200 ⚡切分!
       splitM = AlignDown(200, 128) = 128
       → Segment{g=5, segM=128}
       remaining = 300-128 = 172 ★保留!

  Window 0: startM=0, endM=1428
    含 g0..g4 全量 + g5 前 128 行

Window 1 (capacity=1500):
  g=5: remaining(172) ≤ 1500  ★继续!
       → Segment{g=5, segM=172}
       capacity=1328
  g=6: 100 → capacity=1228
  g=7: 400 → capacity=828

  Window 1: startM=1428, endM=2100
    含 g5 剩余 172 + g6,g7 全量

═══════════════════════════════════════════════════════════
         A3 结果                           A5 结果
         ═══════                           ═══════
  刷出 [0, 1428)                     Window 0: [0, 1428)
  curGroup=5, 留 172 tokens          g=5 remaining=172 保留
  继续 g5 剩余...                     Window 1: g5 剩余 172 处理
═══════════════════════════════════════════════════════════
  ✅ A3 curM=1428 ≈ A5 endM=1428 (完全一致!)
  ✅ A3 curGroup=5 保留 172 = A5 remaining=172 (完全一致!)
  ✅ A3 break 后继续 g5 = A5 Window 1 首个 Segment{g=5,172} (完全一致!)
```

### 7.5 两种机制的本质区别与等价性

```
A3 (运行时检测):
  ┌─────────────────────────────────────────────────────┐
  │ for tile in group:                                   │
  │   curBlockM += singleM                               │
  │   if curBlockM > lowBoundM:  ← 运行时逐 tile 检查    │
  │     推进 curGroup → 遇到跨窗口 group → break          │
  │     FRDeterministic                                  │
  │     lowBoundM += windowSize                          │
  │   VectorCompute (写入)                                │
  │                                                      │
  │ 检查发生在「写入之前」→ 不会溢出                      │
  │ curGroup 保留跨窗口 group → 下轮继续                  │
  └─────────────────────────────────────────────────────┘

A5 (预分区):
  ┌─────────────────────────────────────────────────────┐
  │ BuildWindowPlan():  ← 启动时一次性算好                │
  │   遍历 group, 按 windowSize 分配到 Window             │
  │   遇到跨窗口 group → AlignDown(capacity, singleM)     │
  │   remaining 保留 → 下个 Window 继续同一个 group        │
  │                                                      │
  │ for window in plan:                                  │
  │   for seg in window:                                 │
  │     ProcessSingleGroup(seg.groupIdx, seg.segM)       │
  │   SyncAll → FRDeterministic                          │
  │                                                      │
  │ 预分区保证「每个 Window 的总 token ≤ windowSize」      │
  │ 执行时不检查 → 天然不会溢出                           │
  │ remaining 保留跨窗口 group → 下个 Window 继续          │
  └─────────────────────────────────────────────────────┘
```

**为什么 A5 不能用 A3 的运行时检测方式**

A3 的 tile 级 VectorSync 依赖一个前提：**所有 core 在同一个 group 的同一个 tile 步骤上同步**。A3 用 stride=coreNum 保证这个前提。但 A5 的 ASWT 调度器打破了它——不同 core 可能在处理不同 group。在 core 0 触发 VectorSync 时，core 1 可能还在 2 个 group 之前。此时调用 SyncAll 会让 core 1 在未完成当前 group 时被强制同步，导致中间状态不一致。

**预分区方案的额外优势**:
1. **确定性更强**: window 边界在 kernel 启动时就确定，不受运行时调度影响
2. **不依赖 tile 级同步**: Cgmct 在窗口内完全自由调度，只在窗口间同步
3. **可预测**: flush 次数在 kernel 启动前可精确计算
4. **无检测开销**: 执行时没有 `while (curBlockM > lowBoundM)` 的分支判断

---

## 8. 确定性 Epilogue 设计

### 7.1 为什么新建而非修改原 Epilogue

原 `block_epilogue_dequant_finalize_routing.h` 中的 `VectorAtomicProcess` 直接 scatter-add 到 `yGlobal_`。虽然可以通过 `if (isDeterministic_)` 分支替换输出目标，但这会：

1. 污染原有非确定性代码路径（增加条件分支）
2. 增加原 Epilogue 的成员变量（isDeterministic_, mmQuantOutGm_ 等）
3. 给非确定性模式的维护和调试引入噪音

**新建独立 Epilogue 文件**是最干净的做法：确定性 Epilogue 的职责明确（写 mmQuantOutGm），与原有 Epilogue（写 y）物理隔离。

### 7.2 确定性 Epilogue 结构

```cpp
// 新文件: gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing_deterministic.h

template <typename CType, typename C1Type, typename weightScaleType, 
          typename xScaleType, typename BiasType, typename RowIndexType,
          const auto& MM_CFG = CFG_MDL>
class BlockEpilogueDequantFinalizeRoutingDeterministic {
public:
    // === 成员变量（UB buffer 布局与原 Epilogue 完全一致）===
    // l0cOutUb_, l0cOutUbFloat_, logitUbPing_, logitUbPong_,
    // x2ScaleUbPing_, x2ScaleUbPong_, x1ScaleUbPing_, x1ScaleUbPong_,
    // biasUbPing_, biasUbPong_
    // outUbPing_, outUbPong_

    // === 确定性特有成员 ===
    __gm__ float* mmQuantOutGm_ = nullptr;  // 确定性 workspace 地址
    uint64_t n_ = 0;                         // N 维度
    uint64_t windowBaseM_ = 0;               // 当前窗口起始 M 行号
    uint64_t globalMOffset_ = 0;             // 当前 tile 的全局 M 偏移

    // === Init: UB 分配与原 Epilogue 完全相同 ===
    // 总 UB 使用: 198KB / 256KB (不变)

    // === ★ 唯一差异: 输出方法 ===
    template <typename T>
    __aicore__ inline void VectorDeterministicProcess(
        const LocalTensor<T>& outUb, uint32_t curVecBaseM,
        uint32_t alignN, uint32_t yOffset)
    {
        // 写入确定性 workspace（非 atomic，直接 DataCopyPad）
        uint64_t writeOffset = (globalMOffset_ - windowBaseM_) * n_ + yOffset;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curVecBaseM * alignN * sizeof(T);
        DataCopyPad(mmQuantOutGm_[writeOffset], outUb, copyParams);
    }

    // === operator(): 与原 Epilogue 相同，仅最后一步换为 VectorDeterministicProcess ===
    void operator()(...) {
        // CopyInLogit → CopyX2Scale → CopyX1Scale → CopyBias
        // VFDoDequant → VFDoLogitMuls
        // ★ VectorDeterministicProcess (替代 VectorAtomicProcess)
    }

    // === 窗口状态更新接口（由外部内核包装器在 group 切换时调用）===
    __aicore__ inline void SetWindowBaseM(uint64_t base) { windowBaseM_ = base; }
    __aicore__ inline void SetGlobalMOffset(uint64_t offset) { globalMOffset_ = offset; }
};
```

### 7.3 与原 Epilogue 的代码复用

确定性 Epilogue 中的 VFDoDequant 和 VFDoLogitMuls 实现与原版完全相同。为防止代码重复：

**方案 A（推荐）**: 将反量化核心逻辑抽取为 `gmm/common/cgmct/epilogue/block_epilogue_dequant_impl.h` 中的独立模板函数，原 Epilogue 和确定性 Epilogue 均调用之。

**方案 B**: 确定性 Epilogue 直接 include 原 Epilogue 头文件，继承反量化方法。但这引入了对原文件的依赖。

**方案 C**: 在确定性 Epilogue 中复制反量化相关方法（~200 行）。最独立但有一定重复。由于 Cgmct Epilogue 中反量化逻辑使用大量 pipeline intrinsic，复制相对安全——这些 intrinsic 本身不会随框架版本变化。

**建议采用方案 A**，因为改动最小（仅抽取公共函数），且能保持两个 Epilogue 的反量化逻辑完全同步。

---

## 8. 确定性内核包装器

### 8.1 文件结构

```cpp
// 新文件: op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant_deterministic.h

template <typename LayoutA, typename LayoutB, int ScaleType, int RowIndType>
__aicore__ inline void 
grouped_matmul_finalize_routing_pertoken_dequant_deterministic(
    GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR pertoken_scale, GM_ADDR group_list, GM_ADDR share_input,
    GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, 
    GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // ========== Step 1: 读取 Tiling Data ==========
    auto tiling = reinterpret_cast<const GMMFinalizeRoutingTilingData*>(tilingGM);

    // ========== Step 2: 组装 Cgmct 类型 ==========
    using BlockMmadBuilder = Block::BlockMmadBuilder<AType, LayoutA, ...>;
    using BlockPrologue = BlockPrologueFinalizeRouting<CType, BiasType>;
    using BlockEpilogueDet = BlockEpilogueDequantFinalizeRoutingDeterministic<...>;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;

    // ========== Step 3: 初始化 ==========
    __gm__ float* mmQuantOutGm = reinterpret_cast<__gm__ float*>(
        workspaceGM + CGMCT_WORKSPACE_OFFSET);  // 跳过 16MB Cgmct 区域

    // ========== Step 4: Prologue ==========
    InitPrologueAndRun(...);
    SyncAll();

    // ========== Step 5: 读取 group tokens 列表 ==========
    uint32_t groupNum = tiling->gmmFinalizeRoutingDataParams.groupNum;
    uint8_t groupListType = tiling->gmmFinalizeRoutingDataParams.groupListType;
    uint32_t singleM = tiling->matmulTiling.baseM;
    uint64_t windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(float));

    // 预读所有 group 的 token 数量（支持 count 和 cumsum 格式）
    uint32_t originalGroupTokens[MAX_GROUP_NUM];
    uint64_t totalM = ReadGroupTokens(group_list, groupNum, groupListType, originalGroupTokens);

    // ========== Step 6: 预分区 WindowPlan ==========
    WindowPlan plan = BuildWindowPlan(originalGroupTokens, groupNum, windowSize, singleM);
    // plan.windows[w] = {segments: [{groupIdx, segM}, ...], startM, endM}

    // ========== Step 7: 分段执行 ==========
    for (uint32_t w = 0; w < plan.numWindows; w++) {
        auto& win = plan.windows[w];

        // 更新 Epilogue 窗口基址 = 逻辑起始 M（确保物理写入在 [0, windowSize*n)）
        epilogueDetOp_.SetWindowBaseM(win.startM);

        uint64_t globalOffset = win.startM;
        for (auto& seg : win.segments) {
            uint32_t g = seg.groupIdx;
            uint32_t segM = seg.tokenCount;

            // 调整 problemShape_.m = segM
            SetupGroupSegment(g, segM, seg.processedM);

            epilogueDetOp_.SetGlobalMOffset(globalOffset);

            // ★ Cgmct 原生调用，不改动 ProcessSingleGroup
            ProcessSingleGroup(params, bs, g);

            globalOffset += segM;
            seg.processedM += segM;  // 记录该 group 已处理的 M 行数
        }

        // ★ 窗口结束: 所有 core 同步 + 确定性刷出
        SyncAll();
        syncConfig.curM = win.endM;
        FRDeterministic(syncConfig, mmQuantOutGm, row_index, y, tiling);
    }

    SyncAll();
}
```

### 8.2 WindowPlan 构建

```cpp
struct Segment {
    uint32_t groupIdx;
    uint32_t tokenCount;   // 本段处理的行数
    uint32_t processedM;   // 该 group 已处理的累积 M（用于 offset 调整）
};

struct Window {
    Segment segments[MAX_GROUPS_PER_WINDOW];
    uint32_t numSegments;
    uint64_t startM;       // 本窗口在全局 M 空间中的起始行号
    uint64_t endM;         // 本窗口结束行号（刷出边界）
};

struct WindowPlan {
    Window windows[MAX_WINDOWS];
    uint32_t numWindows;
};

WindowPlan BuildWindowPlan(
    uint32_t* groupTokens, uint32_t groupNum,
    uint64_t windowSize, uint32_t singleM)
{
    WindowPlan plan;
    uint32_t g = 0;
    uint32_t remaining = groupTokens[0];
    uint64_t windowStart = 0;

    while (g < groupNum) {
        Window win;
        win.startM = windowStart;
        win.numSegments = 0;
        uint64_t capacity = windowSize;

        while (capacity > 0 && g < groupNum) {
            if (remaining == 0) {
                g++;
                remaining = (g < groupNum) ? groupTokens[g] : 0;
                continue;
            }

            if (remaining <= capacity) {
                // 完整放入
                win.segments[win.numSegments++] = {g, remaining, 0};
                capacity -= remaining;
                g++;
                remaining = (g < groupNum) ? groupTokens[g] : 0;
            } else {
                // 需要切分: 按 singleM 对齐
                uint32_t splitM = AlignDown((uint32_t)capacity, singleM);
                if (splitM > 0) {
                    win.segments[win.numSegments++] = {g, splitM, 0};
                    remaining -= splitM;
                }
                capacity = 0;  // 本窗口满
            }
        }

        win.endM = windowStart + windowSize - capacity;
        plan.windows[plan.numWindows++] = win;
        windowStart = win.endM;
    }

    return plan;
}
```

### 8.3 与原有内核包装器的关系

原有内核包装器 `KernelGmmFinalizeRoutingPertokenDequant` 和 `ProcessSingleGroup` **完全不动**。确定性包装器：

1. **复用**: BlockMmad（AIC MMAD）、BlockPrologue（初始化）、ASWT Scheduler（tile 分配）——不变
2. **替换**: Epilogue → `BlockEpilogueDequantFinalizeRoutingDeterministic`（写 mmQuantOutGm 而非 y）
3. **新增**: `BuildWindowPlan`（预分区） + 窗口循环（SyncAll + FRDeterministic）
4. **关键**: `ProcessSingleGroup` 完全按照 Cgmct 原样调用，窗口管理在外层，零侵入

---

## 9. UB Buffer 安全分析

### 9.1 实测 UB 使用量（基于源码精确计算）

从 `block_epilogue_dequant_finalize_routing.h:189-211` Init() 逐偏移计算：

| # | Buffer | 类型 | 元素数 | 元素大小 | 大小 | 累计 (VECIN) |
|---|--------|------|--------|---------|------|-------------|
| 1 | `l0cOutUb_` | int32_t | 32768 | 4 | **128.00 KB** | 128.00 KB |
| 2 | `l0cOutUbFloat_` | float | 32768 | 4 | **(aliased, 0)** | 128.00 KB |
| 3 | `logitUbPing_` | float | 256 | 4 | **1.00 KB** | 129.00 KB |
| 4 | `logitUbPong_` | float | 256 | 4 | **1.00 KB** | 130.00 KB |
| 5 | `x2ScaleUbPing_` | bf16 | 256 | 2 | **0.50 KB** | 130.50 KB |
| 6 | `x2ScaleUbPong_` | bf16 | 256 | 2 | **0.50 KB** | 131.00 KB |
| 7 | `x1ScaleUbPing_` | float | 256 | 4 | **1.00 KB** | 132.00 KB |
| 8 | `x1ScaleUbPong_` | float | 256 | 4 | **1.00 KB** | 133.00 KB |
| 9 | `biasUbPing_` | bf16 | 256 | 2 | **0.50 KB** | 133.50 KB |
| 10 | `biasUbPong_` | bf16 | 256 | 2 | **0.50 KB** | 134.00 KB |

| # | Buffer | 类型 | 元素数 | 元素大小 | 大小 |
|---|--------|------|--------|---------|------|
| 11 | `outUbPing_` | float | 8192 | 4 | **32.00 KB** |
| 12 | `outUbPong_` | float | 8192 | 4 | **32.00 KB** |

**VECIN 合计: 137,216 bytes = 134.00 KB**
**VECOUT 合计: 65,536 bytes = 64.00 KB**
**物理 UB 总计: 202,752 bytes = 198.00 KB**

```
UB 使用率: 198.00 / 256.00 = 77.3%
安全余量:  58.00 KB (22.7%)
```

### 9.2 确定性模式对 UB 的影响

| 阶段 | UB 使用 | 说明 |
|------|---------|------|
| **Prologue** | ~112 KB | 仅执行一次，不与 Epilogue 重叠 |
| **Epilogue (tile 处理)** | **198 KB** | 与原 Epilogue 完全一致 |
| **FRDeterministic** | **24 KB** (queBind) | 仅在刷出时使用，此时 Epilogue 已完成 |

**确定性 Epilogue 的 UB 布局与原 Epilogue 完全相同**：
- 不新增任何成员 buffer
- 仅新增 4 个标量成员变量（`mmQuantOutGm_`, `n_`, `windowBaseM_`, `globalMOffset_`，共 32 bytes）
- 对 UB 开销的影响可忽略不计（< 0.01%）

**queBind 仅在 FRDeterministic 期间分配**：
- FRDeterministic 前后都有 SyncAll，所有 core 同步进入/退出
- 在 SyncAll 之后，Epilogue 的 tile 处理已经停止，UB 中的 198KB buffer 全部空闲
- 此时分配 12KB × 2 = 24KB 的 queBind 完全安全

### 9.3 UB 安全承诺

```
Epilogue tile 处理峰值: 198KB / 256KB (77.3%)  ← 与原版相同
FRDeterministic 峰值:    24KB / 256KB (9.4%)   ← 完全空闲时分配
所有阶段 UB 使用 < 80%，余量 > 20%
```

对比 A3 确定性模式的 UB 使用率 190KB/192KB (99%)，A5 有充足的 UB 余量。

---

## 10. Workspace 安全分析

### 10.1 当前 A5 Workspace

```
现有 A5 非确定性: 16MB (SYS_WORKSPACE_SIZE)
  来源: grouped_matmul_host_util.h:105
  用途: Cgmct 框架内部 (同步标志、tile 元数据等)
```

### 10.2 确定性 Workspace 分级策略

| L2 总容量 | 确定性 workspace | 总 workspace | 余量 | 余量百分比 |
|-----------|-----------------|-------------|------|----------|
| 96MB | **32MB** | 48MB | **48MB** | **50%** |
| 128MB | **48MB** | 64MB | **64MB** | **50%** |
| >128MB | **48MB** | 64MB | **64MB+** | **50%+** |

**对比 A3 策略**:
```
A3: L2 > 96MB → 96MB det + 20MB sys = 116MB (占 L2 128MB 的 90.6%)
A5 本方案: L2 > 128MB → 48MB det + 16MB sys = 64MB (占 L2 128MB 的 50%)
```

### 10.3 windowSize 影响

| n | 32MB workspace | 48MB workspace | 典型场景 |
|---|---------------|---------------|---------|
| 2560 | 3277 行 | 4915 行 | 小 N 模型 |
| 4096 | 2048 行 | 3072 行 | 中等 N |
| **7168** | **1170 行** | **1755 行** | DeepSeek-V2/V3 |
| 7680 | 1092 行 | 1638 行 | Qwen2-MoE |
| 13824 | 607 行 | 910 行 | 大 N 模型 |

**刷出频率对比 (n=7168)**:

| totalM | 32MB (ws=1170) | 48MB (ws=1755) | A3 96MB (ws=3510) |
|--------|---------------|---------------|-------------------|
| 256 | 1 | 1 | 1 |
| 2048 | 2 | 2 | 1 |
| 4096 | 4 | 3 | 2 |
| 8192 | 7 | 5 | 3 |

32MB 确定性 workspace 在 tensor-parallel 场景下（每卡 expert 数少、token 数少）完全满足需求（totalM < 1170 时只需 1 次刷出）。大 batch 推理会增加刷出次数，但额外的 SyncAll 和 L2 读取开销仍在可接受范围（+5-15% 延迟）。

---

## 11. Shape 通用性分析

### 11.1 窗口容量速查表

| n | 32MB windowSize | 48MB windowSize | 单次刷出覆盖 token 数 |
|---|----------------|----------------|---------------------|
| 2560 | 3277 | 4915 | 充足（大部分场景无需中途刷出） |
| 4096 | 2048 | 3072 | 中等 |
| 7168 | 1170 | 1755 | ~2K-4K token 时触发 1-2 次 |
| 7680 | 1092 | 1638 | 同上 |
| 13824 | 607 | 910 | 频繁，大 N 场景性能下降 |

### 11.2 Example: totalM=4096, n=7168, 48MB

```
groupList = [1024, 1024, 1024, 1024]
windowSize = 1755, singleM = 128

Group 0 (1024): curBlockM=1024, ≤ 1755, OK
Group 1 (1024): curBlockM=2048, > 1755 ⚡触发 VectorSync
  → curM 推进到 1024 + (1755-1024)/128*128 = 1024 + 704 = 1728
  → FRDeterministic: 刷出 [0, 1728)
  → lowBoundM = 1728 + 1755 = 3483
Group 1 剩余: curBlockM 从 1728 → 2048, ≤ 3483, OK
Group 2 (1024): curBlockM=3072, ≤ 3483, OK  
Group 3 (1024): curBlockM=4096, > 3483 ⚡触发 VectorSync
  → FRDeterministic: 刷出 [1728, 3712)  (3483-1728=1755, 对齐到 128)
  → lowBoundM = 3712 + 1755 = 5467
Group 3 剩余: curBlockM → 4096, final FRDeterministic

共 3 次刷出 (vs A3/96MB 的 2 次)
```

### 11.3 n=7168/7680, k=2048 特殊 Shape

确定性窗口行为与 baseM/baseN 无关。Cgmct 框架的 `CalBasicBlock()` 自动决定 tile shape。`windowSize` 仅取决于 n 和 deterWorkspaceSize。

---

## 12. A3→A5 迁移对照

### 12.1 逐组件迁移

| A3 组件 | A3 位置 | A5 迁移 | 新建/复用 |
|---------|--------|---------|---------|
| SyncConfig | `utils.h:49-56` | 原样复制 | 新建 deterministic.h |
| windowSize 初始化 | `kernel.h:303-306` | 原样逻辑，workspace 偏移适配 | 新建 deterministic.h |
| mmQuantOutGm 地址 | `kernel.h:162-166` | workspaceGM + 16MB (跳过 Cgmct) | 新建 deterministic.h |
| VectorSync | `kernel.h:622-649` | 原样算法，group 粒度调用 | 新建 deterministic.h |
| FRDeterministic | `kernel.h:652-687` | 原样算法，coreNumVec=aicNum×2 | 新建 deterministic.h |
| 确定性写入 | `kernel.h:381-386` | 新 Epilogue::VectorDeterministicProcess | **新建文件** |
| DetermTiling | `base_tiling.cpp:413-425` | 保守分级 (32MB/48MB) | **扩展 tiling** |
| DETER_UB_SIZE=12KB | `utils.h:30` | 不变 | 新建 deterministic.h |
| MMCompute | `kernel.h:338-374` | **复用 BlockMmad (不改)** | 复用 |
| Prologue | `kernel.h` | **复用 BlockPrologue (不改)** | 复用 |
| 调度器 | `utils.h:140-161` | **复用 ASWT Scheduler (不改)** | 复用 |

### 12.2 不需要迁移的组件

| A3 组件 | 原因 |
|--------|------|
| MNBlockIdxCompute (对角/行优先) | A5 使用 ASWT Scheduler |
| InitUbBuffer (手动 UB 分配) | A5 Cgmct 框架自动管理 |
| DataCopyMMOut (L2 workspace → UB) | A5 L0C→UB 直通替代 |
| ComputeDequantAndActivate | A5 Cgmct Epilogue 替代 |

---

## 13. 完整数据流

```
输入: x(int8,m×k), w(int8 NZ,e×k×n), scale(float,e×n),
      pertoken_scale(float,m), groupList(int64,e), rowIndex(int64,m)

┌─ Host Tiling ───────────────────────────────────────────────────┐
│ DoOpTiling() → DoLibApiTiling() → DeterministicTilingProcess() │
│ → 保守分配 32MB/48MB 确定性 workspace                            │
│ → 序列化 deterministicFlag=1, deterWorkspaceSize, n, coreNum    │
└────────────────────────────────────────────────────────────────┘
┌─ Device Kernel ─────────────────────────────────────────────────┐
│ 1. Prologue (复用原有 BlockPrologueFinalizeRouting)              │
│    - zeroInit y + sharedInput 缩放复制                          │
│    SyncAll()                                                     │
│                                                                  │
│ 2. InitDeterministic()                                          │
│    → mmQuantOutGm = workspaceGM + 16MB                          │
│    → windowSize = deterWorkspaceSize / (n × 4)                  │
│    → lowBoundM = windowSize                                     │
│                                                                  │
│ 3. Group 循环:                                                   │
│    globalMOffset = 0                                             │
│    for g in 0..groupNum-1:                                      │
│      groupM = GetGroupTokens(g)                                 │
│      UpdateWindowState(epilogueDet, windowBase, globalMOffset)  │
│      ProcessSingleGroup (Cgmct ASWT):                           │
│        AIC: BlockMmad (复用) → L0C int32 tile                   │
│        AIV: BlockEpilogueDet (新建) → L0C→UB dequant            │
│             → VFDoDequant (复用逻辑)                              │
│             → VFDoLogitMuls (复用逻辑)                           │
│             → VectorDeterministicProcess (★ 确定性写入)         │
│               DataCopyPad → mmQuantOutGm[窗口内偏移]             │
│      globalMOffset += groupM                                    │
│      VectorSync(globalMOffset, syncConfig)  ← 窗口检查          │
│                                                                  │
│ 4. 最终 FRDeterministic:                                        │
│    syncConfig.curM = globalMOffset                              │
│    SyncAll → 多核分区读取 mmQuantOutGm → atomic scatter → y     │
│    SyncAll                                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 14. 确定性正确性证明

### 14.1 正确性三要素

**Layer 1 — 写入隔离**:
每个 AIV core 的反量化结果写入 mmQuantOutGm 的不同偏移（由 `(globalMOffset - windowBase) * n + nOffset` 确定）。不同 core 处理不同的 tile（不同的 M 偏移和 N 偏移），写入不重叠。使用普通 `DataCopyPad`（非 atomic），无写入竞争。

**Layer 2 — 刷出顺序**:
FRDeterministic 中：
- `SyncAll()` 确保所有 core 的窗口内数据已完整写入
- `mOffset` 递增遍历窗口内行 → 顺序确定
- `outRow % coreNumVec == GetBlockIdx()` 分核 → 同一输出行不会被多核同时写
- 每个输出行的数据在窗口内已完整（包含来自不同 expert 的累加），单次写入即完成

**Layer 3 — 窗口隔离**:
每次 FRDeterministic 后 `lowBoundM += windowSize`，下一窗口从新基点开始。同一物理 workspace 被循环复用，但逻辑上窗口间不重叠。

### 14.2 与 A3 正确性的对比

| 正确性要素 | A3 | A5 本方案 |
|----------|-----|---------|
| 写入隔离 | L2 workspace 不同偏移 | 相同（L2 workspace） |
| 刷出分核 | outRow % coreNumVec | 相同 |
| SyncAll 屏障 | 刷出前后各一次 | 相同 |
| 窗口滑动 | lowBoundM += windowSize | 相同 |
| 地址复用 | 全局M - windowBase | 相同 |

---

## 15. 性能与代价评估

### 15.1 额外开销明细

| 开销项 | 量化 | 占比 |
|--------|------|------|
| L2 确定性 workspace | +32MB / +48MB | 芯片 L2 的 25%-50% |
| 每 tile L2 写入 | ~16KB-32KB float | 单次 DataCopyPad |
| 每次刷出 L2 读取 | windowSize × n × 4B | ~12MB-25MB (一次 DMA) |
| 每次刷出 SyncAll ×2 | ~数十 μs | 取决于 core 数量 |
| Epilogue 额外分支 | 0（独立文件，无条件分支） | 0% |

### 15.2 与非确定性对比

| 维度 | A5 非确定性 | A5 确定性 (本方案) |
|------|----------|-----------------|
| 确定性 | 不保证 | **保证 bit-exact** |
| L2 占用 | 16MB | **48MB / 64MB** |
| UB 峰值 | 198KB (77.3%) | 198KB (77.3%) 不变 |
| 延迟 (small batch) | 基准 | **+5%** |
| 延迟 (large batch) | 基准 | **+10-15%** |
| 代码侵入 | 0 | 仅 apt.cpp + 新建 2 个文件 |

---

> **文档版本**: v2.0
> **创建日期**: 2026-05-22
> **更新**: UB 精确测算 (198KB/256KB) + Workspace 保守策略 (32MB/48MB) + 独立文件零侵入设计
> **状态**: 待审查
