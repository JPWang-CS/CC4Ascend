# GMMFR W8A8 A3 确定性 → A5 迁移实现方案

> **目标**: 将 A3 (910B/910C) 的 GMMFR W8A8 确定性滑窗回写机制完整迁移到 A5 (950) 芯片
> **基准**: A3 确定性实现已验证可行，A5 当前仅有非确定性路径
> **核心策略**: 保持 A3 确定性机制的五要素不变，逐层适配 A5 的 CGMCT 组件化架构

---

## 1. 架构差异总览 & 迁移映射

### 1.1 A3 → A5 迁移映射表

```
┌──────────────────────────┬─────────────────────────────────┬──────────────────────────────────┐
│     A3 确定性组件          │     A3 位置 (文件:行号)           │     A5 迁移目标 (文件:函数)        │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ TilingData 字段           │ grouped_matmul_finalize_        │ arch35/grouped_matmul_           │
│ deterministicFlag         │ routing_tiling.h:59             │ finalize_routing_tiling_data.h   │
│ deterWorkspaceSize        │ (扁平宏展开 struct)              │ GMMFinalizeRoutingDataParams     │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ DeterministicTilingPro-   │ base_tiling.cpp:413-425         │ quant_tiling.cpp:                │
│ cess (workspace分配)      │ GroupedMatmulFinalizeRouting-   │ GroupedMatmulFinalizeRouting-    │
│                           │ BaseTiling                      │ QuantTiling::GetWorkspaceSize()  │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ SyncConfig 结构体         │ grouped_matmul_finalize_        │ kernel_gmm_finalize_routing_     │
│                           │ routing_utils.h:49-56           │ pertoken_dequant.h (新增)        │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ mmQuantOutGm 指针         │ grouped_matmul_finalize_        │ BlockEpilogueDequantFinalize-    │
│                           │ routing.h:162-166 (Init)        │ Routing (新增成员)               │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ queBind UB缓冲 (12KB)     │ grouped_matmul_finalize_        │ BlockEpilogueDequantFinalize-    │
│                           │ routing.h:186-188 (InitUbBuffer) │ Routing::Init() (新增 LocalTensor)│
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ windowSize 计算           │ grouped_matmul_finalize_        │ KernelGmmFinalizeRouting-        │
│                           │ routing.h:303-304 (Process)     │ PertokenDequant::operator()      │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ MNBlockIdxCompute         │ grouped_matmul_finalize_        │ ASWT Scheduler 行优先模式         │
│ 行优先调度                │ routing_utils.h:140-161         │ (新增 deterministic 参数)         │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ VectorSync 滑窗检查       │ grouped_matmul_finalize_        │ KernelGmmFinalizeRouting-        │
│                           │ routing.h:622-650               │ PertokenDequant (新增成员函数)    │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ FRDeterministic 回写      │ grouped_matmul_finalize_        │ BlockEpilogueDequantFinalize-    │
│                           │ routing.h:652-687               │ Routing (新增成员函数)            │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ VectorAtomicProcess       │ grouped_matmul_finalize_        │ BlockEpilogueDequantFinalize-    │
│ 确定性分支(线性写)         │ routing.h:381-386              │ Routing::VectorAtomicProcess     │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ SyncAll 全局同步          │ grouped_matmul_finalize_        │ 已有 SyncAll<false>()            │
│                           │ routing.h:294,658,686           │ operator():344                   │
└──────────────────────────┴─────────────────────────────────┴──────────────────────────────────┘
```

### 1.2 确定性五要素在 A5 中的实现路径

```
① 工作空间分配      → Tiling 层: GetWorkspaceSize() 增加 deterWorkspaceSize (64/96MB)
② 行优先调度        → Scheduler: ASWT 增加行优先模式 (deterministicFlag=1 时强制)
③ 线性写入          → Epilogue: VectorAtomicProcess 增加 mmQuantOutGm 分支
④ 滑窗触发+批量回写  → Kernel: ProcessSingleGroup 后 VectorSync + FRDeterministic
⑤ 全局同步          → 已有 SyncAll<false>() + 新增 FRDeterministic 首尾 SyncAll
```

---

## 1.5 整体流程总图 (Kernel 侧)

> 本节回答三个核心问题：
> 1. 确定性模式下，Kernel 的 AIC 和 AIV 各自执行什么？和当前非确定性有什么不同？
> 2. 数据从输入到最终写回 yGm，经过了哪些存储位置？
> 3. 滑窗的生命周期是什么——何时初始化、何时填数据、何时触发 flush、何时前移？

### 1.5.1 非确定性 vs 确定性 全链路对比

```
═══════════════════════════════════════════════════════════════════════════════
                    非确定性 (当前 A5 W8A8)
═══════════════════════════════════════════════════════════════════════════════

  AIV                          AIC
  ───                          ───
  Prologue: yGm归零+shared     mmadOp_.Init()
       │                       l0cOutUb_ = GetL0c2UbTensor()
  SyncAll<false>() ─────────── SyncAll<false>()
  epilogueDequantOp_.Init()    │
       │                       │
  ═══ for each group ══════════════════════════════════════════════════════
  ProcessSingleGroup():        ProcessSingleGroup():
                               │
    ┌─ while(tile) ────────┐   ┌─ while(tile) ──────────────┐
    │                      │   │                             │
    │  WaitForCube()       │   │  WaitForVector()            │
    │       │              │   │  MatMul(A,B) → l0cOutUb_   │
    │       ▼              │   │       int8×int8 → int32     │
    │  Epilogue:           │   │                             │
    │    MTE2: load scales │   │  NotifyVector() ──────────→ │
    │    V: Dequant+Logit  │   │                             │
    │    MTE3:             │   │                             │
    │  ★ AtomicAdd→yGm     │   │                             │
    │       │              │   │                             │
    │  NotifyCube() ──────→│   │                             │
    └──────────────────────┘   └─────────────────────────────┘
  ═══ end for ═══════════════════════════════════════════════════════════
  End()                        End(): WaitForVector()


═══════════════════════════════════════════════════════════════════════════════
                    确定性 (迁移后 A5 W8A8)
═══════════════════════════════════════════════════════════════════════════════

  AIV                                        AIC
  ───                                        ───
  Prologue: yGm归零+shared                   mmadOp_.Init()
       │                                     l0cOutUb_ = GetL0c2UbTensor()
  SyncAll<false>() ────────────────────────── SyncAll<false>()
  ★ InitDeterministic(mmQuantOutGm)          │
  ★ epilogueDequantOp_.Init(deter)           │
  ★ windowSize = deterWsSize / (N × 4)       │
  ★ lowBoundM = windowSize                    │
       │                                     │
  ═══ for each group ════════════════════════════════════════════════════
  ProcessSingleGroup():                      ProcessSingleGroup()
  (调度器=行优先)                              (调度器=行优先)
                                             │
    ┌─ while(tile) ──────────┐               ┌─ while(tile) ──────────────┐
    │                        │               │                             │
    │  WaitForCube()         │               │  WaitForVector()            │
    │       │                │               │  MatMul(A,B) → l0cOutUb_   │
    │       ▼                │               │       int8×int8 → int32     │
    │  Epilogue:             │               │                             │
    │    MTE2: load scales   │               │  NotifyVector() ──────────→ │
    │    V: Dequant+Logit    │               │                             │
    │    MTE3:               │               │                             │
    │  ★ DataCopy2D          │               │                             │
    │    → mmQuantOutGm      │  (线性写,无scatter)                       │
    │       │                │               │                             │
    │  NotifyCube() ────────→│               │                             │
    └────────────────────────┘               └─────────────────────────────┘
       │
       ▼
  ★ VectorSync(groupM):                     ← 每个 group 完成后检查
      if curGroupM + groupM > lowBoundM:
          推进 curM 到 lowBoundM 对齐
          ★ FRDeterministic():               ← 滑窗 flush
              SyncAll<false>()               ← 所有核同步
              for mOffset in [0..totalM):    ← 逐行
                  outRow = rowIndex[mOffset]
                  if outRow % coreNumVec == myCore:
                      DMA: mmQuantOutGm → deterDmaUb_ → AtomicAdd → yGm
              SyncAll<false>()               ← flush 完成
          lowBoundM += windowSize            ← 窗口前移
  ═══ end for ═══════════════════════════════════════════════════════════
  ★ FRDeterministic(): 最终收尾 flush       ← 处理窗口剩余数据
  End()                                      End(): WaitForVector()
```

**一眼看清差异**：确定性 vs 非确定性只有两个变化点：
1. **Epilogue 写回**：`AtomicAdd→yGm` (scatter) 变为 `DataCopy2D→mmQuantOutGm` (线性)
2. **每个 group 后**：新增 `VectorSync` + `FRDeterministic` 批量 flush

其余全部不变——Prologue、MatMul、Dequant、Logit、调度器的 tile 循环逻辑完全相同。

### 1.5.2 AIC/AIV 完整调用时序 (单个 group, 确定性)

```
以 group 0 (M=2048, N=7168, K=2048, baseM=128, baseN=256) 为例
4 个 AIC core, 8 个 AIV sub-core (taskRation=2)

时间 ─────────────────────────────────────────────────────────────────────────▶

AIC-0        AIC-1        AIC-2        AIC-3        AIV-0        AIV-1  ...  AIV-7
│            │            │            │            │            │            │
│ SyncAll ◀──┴────────────┴────────────┴───── SyncAll ───┴────────────┴──── │
│            │            │            │            │            │            │
│ ═ tile(0,0) M=128 ═     │            │            │            │            │
│ WaitVector()           │            │            │            │            │
│ MatMul(128×256)        │            │            │            │            │
│ NotifyVector() ──────────────────────────────→  │            │            │
│            │            │            │   WaitCube()          │            │
│            │            │            │   Epilogue:            │            │
│            │            │            │     MTE2: load scale   │            │
│            │            │            │     V: Dequant+Logit   │            │
│            │            │            │     MTE3: ★DataCopy2D  │            │
│            │            │            │       →mmQuantOutGm    │            │
│            │            │            │   NotifyCube() ────────│────────→   │
│ WaitVector()←──────────────────────────────────              │            │
│            │            │            │            │            │            │
│ ═ tile(0,1) M=128 ═    │            │            │            │            │
│ ...        │            │            │            │            │            │
│            │            │            │            │            │            │
│ ═ tile(1,0) M=128 ═    │            │            │            │            │
│ ...        │            │            │            │            │            │
│            │            │            │            │            │            │
│ ══ ... 共 16×28=448 tiles, 每个 AIC 处理 112 ══ │            │            │
│            │            │            │            │            │            │
│            │            │            │            │            │            │
│ ◀── group 0 完成 ───────┴────────────┴───────────┴────────────┴────────── │
│            │            │            │            │            │            │
│            │            │            │  ★ VectorSync(2048):                │
│            │            │            │    2048 ≤ 3511 (lowBoundM)          │
│            │            │            │    → 不触发 flush                    │
│            │            │            │    → 继续 group 1                   │
│            │            │            │            │            │            │
│ ══ group 1 (M=512) ═══════════════  │            │            │            │
│ ...        │            │            │            │            │            │
│            │            │            │  ★ VectorSync(512):                 │
│            │            │            │    2560 ≤ 3511 → 不触发              │
│            │            │            │            │            │            │
│ ══ group 2 (M=1024) ══════════════  │            │            │            │
│ ...        │            │            │            │            │            │
│            │            │            │  ★ VectorSync(1024):                │
│            │            │            │    3584 > 3511 → ★ 触发!            │
│            │            │            │            │            │            │
│  ◀────── SyncAll ──────┴────────────┴──── SyncAll ─┴────────────┴──── │
│            │            │            │            │            │            │
│            │            │            │  ★ FRDeterministic:                │
│            │            │            │    totalM=3456 行                    │
│            │            │            │    for mOffset=0..3455:             │
│            │            │            │      outRow=rowIndex[mOffset]       │
│            │            │            │      if outRow%8 == myId:           │
│            │            │            │        DMA: mmQuantOutGm → UB       │
│            │            │            │        AtomicAdd: UB → yGm          │
│            │            │            │            │            │            │
│  ◀────── SyncAll ──────┴────────────┴──── SyncAll ─┴────────────┴──── │
│            │            │            │            │            │            │
│            │            │            │  lowBoundM = 3456+3511 = 6967      │
│            │            │            │            │            │            │
│ ══ group 3 (M=2560) ═══════════════  │            │            │            │
│ ...        │            │            │            │            │            │
│            │            │            │  ★ VectorSync(2560):                │
│            │            │            │    6144 ≤ 6967 → 不触发              │
│            │            │            │            │            │            │
│            │            │            │  ★ 最终 FRDeterministic:            │
│            │            │            │    curM = 6144                       │
│            │            │            │    totalM = 6144-(6967-3511) = 2688  │
│            │            │            │    flush 2688 行 → yGm              │
│            │            │            │            │            │            │
│  ◀────── SyncAll ──────┴────────────┴──── SyncAll ─┴────────────┴──── │
│            │            │            │            │            │            │
│ End()      │            │            │  End()     │            │            │
```

### 1.5.3 数据存储位置流转 (单个 token)

```
一个 token 的数据从输入到最终写入 yGm 的完整路径：

                                  存储位置                  确定性/非确定性
                                  ════════                  ══════════════
① 输入读取                       GM → L1                   相同
  x[M×K] int8  ────────→  L1 缓冲
  w[N×K] int8  ────────→  L1 缓冲

② Cube 矩阵乘法                  L0                       相同
  int8 × int8  ────────→  L0A/L0B → Cube MAC

③ Cube 输出                      L0C                      相同
  int32 累加   ────────→  L0C 寄存器

④ L0C → UB 直通 (A5 特有)        VECIN UB                  相同
  int32        ────────→  l0cOutUb_ [MAX_SINGLE_MNS=128×256]
                            ↑ AIC 写, AIV 读 (同一物理地址)

⑤ VF 反量化                      VF 寄存器                相同
  int32        ────────→  Cast → float32
  × x2Scale    ────────→  Mul (per-channel)
  × x1Scale    ────────→  Mul (BRC_B32 广播 per-token)
  + bias       ────────→  Add (可选)

⑥ VF × Logit                    VF 寄存器                相同
  × logit      ────────→  Mul (BRC_B32 广播 per-token)
                     │
                     ▼
              ┌──────┴──────┐
              │  分叉点!!!   │
              └──────┬──────┘
                     │
    非确定性路径              确定性路径
    ════════════              ══════════
                     │
⑦-a VECOUT UB                VECOUT UB
    float        ──→ outUbPing/Pong        float       ──→ outUbPing/Pong
                     │                                   │
⑧-a MTE3 DMA                 MTE3 DMA
    AtomicAdd    ──→ yGm[outRow×N]         DataCopy2D   ──→ mmQuantOutGm[yOffset]
    (scatter,    (目标: 输出行×N)           (线性,       (目标: 按token顺序排列)
     无序)                                       有序)
                     │                                   │
⑨-a 结束                     ⑨-b 等待窗口 flush
    结果已在 yGm                                   │
                                               SyncAll
                                                   │
⑩-b FRDeterministic: DMA bounce
    mmQuantOutGm[m×N] ──→ deterDmaUb_ (12KB UB)
                                                   │
⑪-b AtomicAdd
    deterDmaUb_   ──→ yGm[outRow×N + nOffset]
    (单核顺序执行, 累加顺序确定)
                                                   │
⑫-b 结束
    结果已在 yGm
```

**关键结论**: 确定性路径比非确定性多走 `mmQuantOutGm → deterDmaUb_ → yGm` 三跳，
代价是 64~96MB workspace + 2×SyncAll/窗口 + 行优先调度，收益是浮点累加顺序确定。

### 1.5.4 滑窗生命周期

```
初始化 (operator() 入口):
  windowSize = deterWorkspaceSize / (N × sizeof(float))
             = 96MB / (7168 × 4) ≈ 3511 行
  lowBoundM  = windowSize = 3511    ← 窗口下界
  curM       = 0                    ← 已处理行数
  curGroup   = 0                    ← 当前 group 索引
  curGroupM  = 0                    ← curGroup 前的累积行数

Phase 1: 窗口内线性写入 (每个 tile 的 Epilogue)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  mmQuantOutGm 布局:                                                │
  │  ┌─────────┬─────────┬─────────┬─────────┬─────┬─────────────────┐│
  │  │ Token 0 │ Token 1 │ Token 2 │ Token 3 │ ... │ Token windowSize││
  │  │ [0..N-1]│ [0..N-1]│ [0..N-1]│ [0..N-1]│     │ [0..N-1]       ││
  │  └─────────┴─────────┴─────────┴─────────┴─────┴─────────────────┘│
  │  写入: DataCopy2D(outUb, mmQuantOutGm[yOffset], dimParams, N)      │
  │  yOffset = 全局 M 行偏移 × N + N 列偏移                             │
  │  即: 按 token 在所有 group 中的线性顺序连续写入                       │
  └─────────────────────────────────────────────────────────────────────┘

Phase 2: 窗口触发检查 (每个 group 完成后的 VectorSync)
  curGroupM + groupM > lowBoundM ?
      │
      ├─ No → 继续, 写入下一个 group
      │
      └─ Yes → 触发 flush:
            │
            ▼
  ┌─── FRDeterministic ───────────────────────────────────────────────┐
  │  ① SyncAll<false>()                                              │
  │     → 等所有核完成窗口内的计算                                      │
  │                                                                   │
  │  ② totalM = curM - (lowBoundM - windowSize)                      │
  │     → 计算窗口中实际有效行数                                        │
  │                                                                   │
  │  ③ for mOffset = 0 to totalM:                                    │
  │       outRow = rowIndexGm[windowStart + mOffset]                  │
  │       if outRow % coreNumVec != myCore: skip                      │
  │       ★ 按行号取模分配, 每行只有一个核写入                            │
  │       for nOffset = 0 to N step baseN:                            │
  │         DMA: mmQuantOutGm[mOffset×N + nOffset] → deterDmaUb_     │
  │         AtomicAdd: deterDmaUb_ → yGm[outRow×N + nOffset]         │
  │                                                                   │
  │  ④ SyncAll<false>()                                              │
  │     → flush 完成, 所有核可以继续下一轮计算                           │
  └───────────────────────────────────────────────────────────────────┘
            │
            ▼
  窗口前移:
    lowBoundM = curM + windowSize
    mmQuantOutGm 开头可以复用 (已 flush 的数据不再需要)

最终收尾 (所有 group 处理完):
  curM = 全局总 M
  FRDeterministic(): flush 最后一个窗口的剩余数据
```

---

## 2. TilingData 结构变更

### 2.1 目标文件: `arch35/grouped_matmul_finalize_routing_tiling_data.h`

当前结构需要新增 `deterministicFlag` 和 `deterWorkspaceSize`。由于 `GMMFinalizeRoutingDataParams` 已有 `reserved1` 和 `reserved2`，可以复用或扩展：

```cpp
// 变更: arch35/grouped_matmul_finalize_routing_tiling_data.h:26-47

#pragma pack(push, 8)
struct GMMFinalizeRoutingDataParams {
    uint32_t groupNum = 0;
    uint32_t batch = 0;
    uint32_t sharedInputOffset = 0;
    uint32_t sharedInputLen = 0;
    float residualScale = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint32_t biasDtype = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint8_t deterministicFlag = 0;     // ★ NEW: 确定性标志 (复用 reserved1 低8位)
    uint8_t reserved1 = 0;             // 缩减为 8 位
    uint32_t deterWorkspaceSize = 0;   // ★ NEW: 确定性工作空间字节数 (复用 reserved2)
};
#pragma pack(pop)
// 总大小不变 (40字节), deterministicFlag 从 reserved1 拆分
```

**兼容性**: `reserved1` 原为 `uint16_t`，拆分为 `deterministicFlag:8 + reserved1:8`。`reserved2` 原为 `uint32_t`，改为 `deterWorkspaceSize`。总 struct 大小不变，内存布局兼容。

### 2.2 Kernel 侧读取

在 `grouped_matmul_finalize_routing_pertoken_dequant.h` 中通过已有的 `tilingData.gmmFinalizeRoutingDataParams` 直接访问：

```cpp
auto detFlag = tilingData.gmmFinalizeRoutingDataParams.deterministicFlag;
auto deterWsSize = tilingData.gmmFinalizeRoutingDataParams.deterWorkspaceSize;
```

---

## 3. Tiling 层变更 (Host)

### 3.1 目标文件: `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

#### 3.1.1 DoOpTiling() — 新增确定性参数填充

```cpp
// quant_tiling.cpp:445-460 — DoOpTiling() 末尾增加:

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoOpTiling()
{
    // ... 现有代码 (line 447-458) ...

    // ★ NEW: 确定性参数
    if (context_->GetDeterministic() == 0) {
        tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 0;
        tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = 0;
    } else {
        tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
        // 确定性工作空间在 GetWorkspaceSize() 中计算, 这里先标记
    }

    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}
```

#### 3.1.2 GetWorkspaceSize() — 覆写基类，增加确定性空间

当前 A5 的 `GetWorkspaceSize()` 继承自 `GroupedQmmTiling`，仅分配 `SYS_WORKSPACE_SIZE` (约 20MB 用于 System RPC)。需要覆写：

```cpp
// quant_tiling.h 声明中增加:
ge::graphStatus GetWorkspaceSize() override;

// quant_tiling.cpp 实现:

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;  // 保留基类的 System RPC 空间

    // ★ NEW: 确定性工作空间
    if (context_->GetDeterministic() != 0) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t l2_size;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);

        constexpr uint64_t DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024;       // 96MB
        constexpr uint64_t DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024; // 64MB

        uint32_t deterWsSize = l2_size > DETER_WORK_SPACE_SIZE
                               ? DETER_WORK_SPACE_SIZE
                               : DETER_WORK_SPACE_LOWER_SIZE;

        tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWsSize;
        workspaces[0] += deterWsSize;  // 叠加到总 workspace
    }

    return ge::GRAPH_SUCCESS;
}
```

### 3.2 Workspace 布局 (确定性 vs 非确定性)

A5 的 workspace 布局本质上比 A3 简单 — A5 **不使用 GM workspace 做 Cube MatMul 中转** (L0C→VECIN 直通)，因此 workspace 仅用于 System RPC。

```
A5 Workspace 布局 (非确定性):
┌──────────────────────────┐
│  System RPC (~20MB)      │
└──────────────────────────┘

A5 Workspace 布局 (确定性):
┌──────────────────────────┬──────────────────────────────────┐
│  System RPC (~20MB)      │  确定性工作空间 (64MB or 96MB)     │
│                          │  float32 [windowSize × N]        │
│                          │  用于存储 stage1 线性中间结果      │
└──────────────────────────┴──────────────────────────────────┘
                           ▲
                   mmQuantOutGm 指向此处
                   偏移 = SYS_WORKSPACE_SIZE
```

**关键差异** vs A3: A3 的 workspace 前缀有 `parallNum × baseM × baseN × sizeof(int32) × coreNum` 的 Cube MatMul 输出缓冲。A5 无需此缓冲，mmQuantOutGm 直接从 `workspace + SYS_WORKSPACE_SIZE` 开始。

---

## 4. Kernel 层变更 — 入口 & 调度包装

### 4.1 目标文件: `arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`

#### 4.1.1 GMMTiling 结构体扩展

```cpp
// pertoken_dequant.h:73-78 — GMMTiling 增加确定性字段:

GMMTiling gmmParams{
    gmmFinalizeRoutingQuantParams_.groupNum,
    gmmFinalizeRoutingQuantParams_.groupListType,
    matmulTiling_.baseM,
    matmulTiling_.baseN,
    matmulTiling_.baseK,
    gmmFinalizeRoutingQuantParams_.hasBias,
    /* ★ NEW: */ gmmFinalizeRoutingQuantParams_.deterministicFlag,
    /* ★ NEW: */ gmmFinalizeRoutingQuantParams_.deterWorkspaceSize,
    /* ★ NEW: */ matmulTiling_.usedCoreNum,   // for coreNumVec in FRDeterministic
    /* ★ NEW: */ matmulTiling_.N              // for N in address calculation
};
```

同步更新 `KernelGmmFinalizeRoutingPertokenDequant::GMMTiling` (在 `kernel_gmm_finalize_routing_pertoken_dequant.h:138-155`)：

```cpp
struct GMMTiling {
    uint32_t groupNum;
    uint8_t groupListType;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    uint8_t hasBias;
    // ★ NEW:
    uint8_t deterministicFlag;
    uint32_t deterWorkspaceSize;
    uint32_t coreNum;          // aicNum (不带 taskRation)
    uint32_t nSize;            // = N (输出列数)
    const TCubeTiling *__restrict matmulTiling;

    __aicore__ GMMTiling() {}
    __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_,
                         int32_t baseN_, int32_t baseK_, uint8_t hasBias_,
                         uint8_t detFlag_, uint32_t deterWsSize_,
                         uint32_t coreNum_, uint32_t nSize_)
        : groupNum(groupNum_), groupListType(groupListType_),
          baseM(baseM_), baseN(baseN_), baseK(baseK_), hasBias(hasBias_),
          deterministicFlag(detFlag_), deterWorkspaceSize(deterWsSize_),
          coreNum(coreNum_), nSize(nSize_) {}
};
```

#### 4.1.2 传递 workspace 地址

当前 A5 的 pertoken_dequant 函数签名接收 `workspaceGM` 但未使用。需要传递给 Kernel 和 Epilogue：

```cpp
// pertoken_dequant.h:89 — params 中增加确定性参数:
Params params = {
    {1, 1, 1, 1},
    {x, w, y, bias, group_list},
    {share_input, y, ...},
    {y, w_scale, x_scale, bias, logit, row_index,
     matmulTiling_.baseM, matmulTiling_.baseN,
     /* ★ NEW: */ workspaceGM,                                    // 确定性工作空间
     /* ★ NEW: */ gmmFinalizeRoutingQuantParams_.deterministicFlag,
     /* ★ NEW: */ gmmFinalizeRoutingQuantParams_.deterWorkspaceSize},
    gmmParams};
```

---

## 5. Kernel 层变更 — 核心流水

### 5.1 目标文件: `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h`

#### 5.1.1 新增 SyncConfig 结构体

```cpp
// 在 namespace {} 块中新增 (line 41-62 之后):

// ★ NEW: 确定性滑窗同步配置
struct DeterSyncConfig {
    uint64_t curM = 0;          // 窗口中已处理的有效数据行数
    uint64_t curGroup = 0;      // 已处理到的 group 索引
    uint64_t curGroupM = 0;     // curGroup 之前的累积行数
    uint64_t lowBoundM = 0;     // 窗口下界
    uint64_t windowSize = 0;    // 窗口最大行容量
    uint64_t baseN = 0;         // flush 时的 baseN (对齐到128)
};
```

#### 5.1.2 类成员变量新增

```cpp
// 在 KernelGmmFinalizeRoutingPertokenDequant 类中 (line 123-136 之后):

    // ★ NEW: 确定性相关成员
    AscendC::GlobalTensor<float> mmQuantOutGm_;   // 确定性工作空间 (float输出)
    DeterSyncConfig deterSyncConfig_;              // 滑窗状态
    bool isDeter_ = false;                         // 确定性模式标记
```

#### 5.1.3 operator() 变更

```cpp
// operator() (line 332-363) — 增加确定性初始化:

__aicore__ inline void operator()(const Params &params)
{
    // === 现有代码: Prologue ===
    if ASCEND_IS_AIV {
        prologueOp_.Init(params.prologueParams);
        prologueOp_();
    }

    // ★ NEW: 确定性初始化 — mmQuantOutGm 指针
    if (params.gmmParams.deterministicFlag == 1) {
        isDeter_ = true;
        if ASCEND_IS_AIV {
            // mmQuantOutGm = workspace + SYS_WORKSPACE_SIZE
            // 偏移量由 Tiling 层计算: System RPC 空间大小
            constexpr uint64_t SYS_WS_OFFSET = 20 * 1024 * 1024; // 20MB (与 SYS_WORKSPACE_SIZE 一致)
            mmQuantOutGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(
                    params.epilogueParams.workspaceGM + SYS_WS_OFFSET));
        }

        // 计算窗口大小
        deterSyncConfig_.windowSize =
            params.gmmParams.deterWorkspaceSize /
            (params.gmmParams.nSize * sizeof(float));      // DTYPE_OUT = float
        deterSyncConfig_.lowBoundM = deterSyncConfig_.windowSize;
        deterSyncConfig_.curM = 0;
        deterSyncConfig_.curGroup = 0;
        deterSyncConfig_.curGroupM = 0;

        // baseN: N方向flush分段大小
        uint64_t nTimes = Ceil(params.gmmParams.nSize,
                               DETER_UB_SIZE / sizeof(float));
        deterSyncConfig_.baseN = Ceil(Ceil(params.gmmParams.nSize, nTimes), 128) * 128;
    }

    // === 现有代码: AIC Init ===
    if ASCEND_IS_AIC {
        mmadOp_.Init(const_cast<TCubeTiling *__restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
        l0cOutUb_ = epilogueDequantOp_.GetL0c2UbTensor();
    }

    // === 现有代码: 全局初始化 ===
    InitParamsAndTensor(params);
    BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
    SyncAll<false>();

    // ★ NEW: AIV Epilogue Init — 传入确定性参数
    if ASCEND_IS_AIV {
        epilogueDequantOp_.Init(params.epilogueParams);
        if (isDeter_) {
            epilogueDequantOp_.InitDeterministic(mmQuantOutGm_, params.gmmParams.nSize);
        }
    }

    // ... 现有 group 循环 ...
    uint32_t groupNum = params.gmmParams.groupNum;
    // ... SetTailAlign ...

    // ★ 确定性模式: 设置调度器为行优先
    if (isDeter_) {
        bs.SetRowPriority(true);
    }

    for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
        if (!UpdateGroupParams(params, groupIdx)) continue;
        ProcessSingleGroup(params, bs, groupIdx);

        // ★ NEW: 确定性滑窗检查
        if (isDeter_ && ASCEND_IS_AIV) {
            uint64_t m = Get<MNK_M>(problemShape_);
            VectorSync(m, params);
        }
    }

    // ★ NEW: 确定性最终收尾 flush
    if (isDeter_ && ASCEND_IS_AIV) {
        FRDeterministic(params);
    }

    End();
}
```

#### 5.1.4 VectorSync — 滑窗检查 (新增成员函数)

```cpp
// ★ NEW: 在 KernelGmmFinalizeRoutingPertokenDequant 类中新增:

__aicore__ inline void VectorSync(uint64_t groupM, const Params &params)
{
    if (!isDeter_) return;

    // curBlockM = 累积M + 当前group的M (即该group完成后的上界)
    uint64_t curBlockM = deterSyncConfig_.curGroupM + groupM;

    while (curBlockM > deterSyncConfig_.lowBoundM) {
        // 推进 curGroup 和 curM 到 lowBoundM
        while (deterSyncConfig_.curGroup < params.gmmParams.groupNum) {
            uint32_t mi = GetGroupM(deterSyncConfig_.curGroup);
            if (deterSyncConfig_.curGroupM + mi <= deterSyncConfig_.lowBoundM) {
                deterSyncConfig_.curGroupM += mi;
                deterSyncConfig_.curM = deterSyncConfig_.curGroupM;
                deterSyncConfig_.curGroup++;
            } else {
                // 对齐到 baseM
                deterSyncConfig_.curM +=
                    (deterSyncConfig_.lowBoundM - deterSyncConfig_.curM)
                    / params.gmmParams.baseM * params.gmmParams.baseM;
                break;
            }
        }
        // 执行窗口 flush
        FRDeterministic(params);
        // 窗口下界前移
        deterSyncConfig_.lowBoundM = deterSyncConfig_.curM + deterSyncConfig_.windowSize;
    }
    // 更新 curGroupM (当前 group 已在窗口中)
    deterSyncConfig_.curGroupM += groupM;
}
```

**注意**: A5 的 VectorSync 在 group 级别触发 (而非 A3 的 tile 级别)，这是因为 A5 的 ASWT 调度器在 group 内自动管理 tile 分配。在 group 完成后检查窗口边界是安全的，因为 group 内所有 tile 的 M 坐标必定连续。

#### 5.1.5 GetGroupM — 辅助函数

```cpp
// ★ NEW: 读取 group M 值的辅助函数
__aicore__ inline uint32_t GetGroupM(uint32_t groupIdx)
{
    // 直接通过 groupListGm_ 读取 (groupListType=1 直接值, groupListType=0 差分值)
    int32_t offset = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
    int32_t preOffset = (groupIdx > 0) ? static_cast<int32_t>(groupListGm_.GetValue(groupIdx - 1)) : 0;
    // 简化: 假设 groupListType=1 (直接值) 或 groupListType=0 (差分)
    // 实际实现需要根据 params.gmmParams.groupListType 区分
    return (groupListType_ == 0) ? (offset - preOffset) : offset;
}
```

#### 5.1.6 FRDeterministic — 批量回写 (新增成员函数)

```cpp
// ★ NEW: 确定性批量 flush 函数
__aicore__ inline void FRDeterministic(const Params &params)
{
    if (!isDeter_) return;
    if ASCEND_IS_AIC return;

    SyncAll<false>();  // ★ 全局同步: 等所有核完成窗口内计算

    uint64_t totalM = deterSyncConfig_.curM -
                      (deterSyncConfig_.lowBoundM - deterSyncConfig_.windowSize);
    uint64_t coreNumVec = params.gmmParams.coreNum * GetTaskRation();
    uint64_t n = params.gmmParams.nSize;
    uint64_t windowStart = deterSyncConfig_.lowBoundM - deterSyncConfig_.windowSize;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        // 读路由信息
        auto outRow = static_cast<uint64_t>(
            epilogueDequantOp_.GetRowIndex(windowStart + mOffset));

        // 按行号取模分配核
        if (outRow % coreNumVec != GetBlockIdx()) continue;

        // N 方向分段搬运
        uint64_t curVecBaseN = deterSyncConfig_.baseN;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += deterSyncConfig_.baseN) {
            if (nOffset + deterSyncConfig_.baseN >= n) {
                curVecBaseN = n - nOffset;
            }

            // 使用 Epilogue 的 DMA ping-pong 缓冲
            epilogueDequantOp_.DeterministicFlushRow(
                mmQuantOutGm_[mOffset * n + nOffset],   // 源: 确定性工作空间
                outRow * n + nOffset,                    // 目标: yGm
                curVecBaseN);                            // 搬运大小
        }
    }

    SyncAll<false>();  // ★ flush 完成
}
```

---

## 6. Epilogue 层变更 — 写回逻辑

### 6.1 目标文件: `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h`

#### 6.1.1 新增成员变量

```cpp
// 在 BlockEpilogueDequantFinalizeRouting 类的 private 段 (line 147-158 之后):

    // ★ NEW: 确定性相关
    bool isDeter_ = false;
    AscendC::GlobalTensor<float> mmQuantOutGm_;    // 确定性工作空间
    AscendC::GlobalTensor<DataTypeRowIndex> rowIndexGmFull_;  // 完整 rowIndex (用于FRDeterministic)
    uint64_t nFull_;                                 // 全局 N
    AscendC::LocalTensor<DataTypeOut> deterDmaUb_;   // 确定性 DMA bounce buffer
    static constexpr uint32_t DETER_UB_SIZE = 12 * 1024; // 12KB (与A3一致)
```

#### 6.1.2 Init() 变更 — 增加确定性 UB 分配

```cpp
// Init() (line 181-212) — 在现有 UB 分配末尾增加:

__aicore__ inline void Init(const Params &params)
{
    // ... 现有代码 (line 185-211) ...

    // ★ NEW: 确定性 DMA 缓冲
    // 在 outUbPong_ 之后分配
    if (params.deterministicFlag == 1) {
        uint32_t afterOutPong = afterBiasPong + 2 * HALF_DB_MAX_SINGLE_MNS * sizeof(DataTypeOut);
        deterDmaUb_ = AscendC::LocalTensor<DataTypeOut>(
            AscendC::TPosition::VECOUT, afterOutPong, DETER_UB_SIZE / sizeof(DataTypeOut));
        isDeter_ = true;
    }
}
```

**UB 布局变更**:

```
确定性模式下 A5 Epilogue UB 布局 (VECIN + VECOUT):

VECIN (输入侧, 不变):
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│l0cOutUb_ │l0cOutUb  │logitPing │logitPong │x2Scale   │x2Scale   │x1Scale   │x1Scale   │
│int32     │Float_    │float     │float     │Ping      │Pong      │Ping      │Pong      │
│128KB     │128KB     │256B      │256B      │256B      │256B      │256B      │256B      │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│biasPing  │biasPong  │
│256B      │256B      │
└──────────┴──────────┘

VECOUT (输出侧, 扩展):
┌──────────┬──────────┬──────────────────────┐
│outUbPing │outUbPong │ deterDmaUb_          │  ★ NEW
│float     │float     │ float                │
│32KB      │32KB      │ 12KB                 │
└──────────┴──────────┴──────────────────────┘

VECOUT 总计: 32KB + 32KB + 12KB = 76KB
UB 总计: ~256KB (VECIN) + 76KB (VECOUT) ≈ 332KB
```

#### 6.1.3 Params 扩展

```cpp
// Params (line 72-82) — 增加确定性参数:

struct Params {
    GM_ADDR yGMAddr{nullptr};
    GM_ADDR x2ScaleGmAddr{nullptr};
    GM_ADDR x1ScaleGmAddr{nullptr};
    GM_ADDR biasGmAddr{nullptr};
    GM_ADDR logitGmAddr{nullptr};
    GM_ADDR rowIndexGmAddr{nullptr};
    int32_t baseM = 256;
    int32_t baseN = 256;
    // ★ NEW:
    GM_ADDR workspaceGM{nullptr};        // 确定性工作空间
    uint8_t deterministicFlag = 0;       // 确定性标志
    uint32_t deterWorkspaceSize = 0;     // 确定性工作空间大小
    Params() = default;
};
```

#### 6.1.4 InitDeterministic — 新增初始化函数

```cpp
// ★ NEW: 在 operator() 之前新增:
__aicore__ inline void InitDeterministic(
    AscendC::GlobalTensor<float> &mmQuantOutGm, uint64_t n)
{
    if ASCEND_IS_AIC return;
    mmQuantOutGm_ = mmQuantOutGm;
    nFull_ = n;
    isDeter_ = true;
}
```

#### 6.1.5 VectorAtomicProcess — 增加确定性分支

```cpp
// VectorAtomicProcess (line 262-274) — 增加确定性分支:

__aicore__ inline void VectorAtomicProcess(
    uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM,
    uint64_t yOffset, LocalTensor<DataTypeOut> &yLocal)
{
    if (isDeter_) {
        // ★ 确定性路径: 线性写入 mmQuantOutGm (无 AtomicAdd)
        DataCopy2DDimParams dimParams{
            curVecBaseM,
            curBaseN,
            static_cast<uint32_t>(alignN_)};
        // 偏移计算: yGmOffset1 是当前 tile 在全局输出中的线性偏移
        // 减去窗口起始偏移得到确定性工作空间中的偏移
        DataCopyPad2D(
            mmQuantOutGm_[yOffset],   // 源: 全局线性偏移
            yLocal,                    // 目标: UB 中的 dequant+logit 结果
            dimParams,
            n_);                      // pitch = N (全局列数)
        return;
    }

    // === 现有非确定性代码 (不变) ===
    SetAtomicAdd<float>();
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
    for (uint32_t i = 0; i < curVecBaseM; i++) {
        auto outRow = static_cast<uint64_t>(rowIndexGlobal_.GetValue(offsetM + i));
        DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN_], paramsOut);
    }
    SetAtomicNone();
}
```

**注意**: A5 的 `yOffset` 等价于 A3 的 `yGmOffset1`，它表示当前 tile 在全局输出矩阵中的线性偏移 (M_global × N + N_offset)。在确定性模式下，这个值直接作为 `mmQuantOutGm_` 的下标，因为确定性工作空间按 token 线性顺序存储。

#### 6.1.6 DeterministicFlushRow — 新增行回写函数

```cpp
// ★ NEW: 确定性单行 flush (在 FRDeterministic 中逐行调用)

__aicore__ inline void DeterministicFlushRow(
    __gm__ float *src,       // mmQuantOutGm[mOffset * n + nOffset]
    uint64_t dstRowOffset,    // outRow * n + nOffset
    uint64_t nElements)     // curVecBaseN
{
    // Step 1: DMA 搬入: mmQuantOutGm → deterDmaUb_
    DataCopy2DDimParams copyInParams{
        static_cast<uint32_t>(1),           // 1 行
        static_cast<uint32_t>(nElements),   // N 方向元素数
        static_cast<uint32_t>(nElements)};  // pitch
    DataCopyPad2D(deterDmaUb_, src, copyInParams);

    // Step 2: 等待 DMA 完成
    // 在 A5 中, DataCopyPad2D 是同步的 (取决于具体API), 如需异步使用 SetFlag/WaitFlag

    // Step 3: AtomicAdd 写回: deterDmaUb_ → yGm
    SetAtomicAdd<float>();
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(nElements * sizeof(float)), 0, 0, 0};
    DataCopyPad(yGlobal_[dstRowOffset], deterDmaUb_, paramsOut);
    SetAtomicNone();
}
```

#### 6.1.7 GetRowIndex — 辅助函数

```cpp
// ★ NEW: 暴露 rowIndex 读取接口 (供 FRDeterministic 使用)

__aicore__ inline uint64_t GetRowIndex(uint64_t globalOffset)
{
    return static_cast<uint64_t>(rowIndexGlobal_.GetValue(globalOffset));
}
```

**设计说明**: A5 不使用 A3 的 `queBind` (TQueBind 是 A3 特有 API)。改用 `deterDmaUb_` 作为手动 DMA bounce buffer：从 `mmQuantOutGm` 搬入 UB，再 AtomicAdd 到 `yGm`。这在功能上与 A3 的 queBind Ping/Pong 流水等价，但更贴近 A5 的编程模型。

---

## 7. Scheduler 适配 — 行优先模式

### 7.1 必要性分析

A3 确定性要求 **行优先调度** (`MNBlockIdxCompute` 中 `(curBlock-count)/blockDimN`)，以确保窗口内的行在 flush 前已被所有核处理完毕。

A5 的 `GroupedMatmulAswtWithTailSplitScheduler` 默认使用 ASWT (Adaptive Split With Tail) 策略，tile 分配顺序不一定严格行优先。对于确定性模式，需要在 `GetTileIdx` 中强制按行优先分配。

### 7.2 改动方案: 增加 SetRowPriority 接口

#### 目标文件: `gmm/common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h`

```cpp
// 在 SchedulerOp 类中新增成员和方法:

class SchedulerOp {
    // ... 现有成员 ...
    bool rowPriority_ = false;  // ★ NEW

public:
    // ★ NEW: 设置行优先模式
    __aicore__ inline void SetRowPriority(bool enable) {
        rowPriority_ = enable;
    }
};

// GetTileIdx 中增加行优先分支:
__aicore__ inline bool GetTileIdx(BlockCoord &blockCoord)
{
    if (rowPriority_) {
        return GetTileIdxRowPriority(blockCoord);
    }
    // ... 现有 ASWT 逻辑 ...
}

// ★ NEW: 行优先 tile 分配 (模仿 A3 的 MNBlockIdxCompute)
__aicore__ inline bool GetTileIdxRowPriority(BlockCoord &blockCoord)
{
    // 简化实现: 按行优先顺序分配 tile
    // curBlock: 当前 core 已处理的 tile 数
    // totalTiles = blockDimM * blockDimN (含尾块)
    // 每个 core 步长 = coreNum

    uint64_t totalTiles = blockDimM_ * blockDimN_;
    uint64_t curBlock = coreRoundIdx_ * coreNum_ + coreIdx_;

    if (curBlock >= totalTiles) return false;

    uint64_t mIdx = curBlock / blockDimN_;  // row-priority
    uint64_t nIdx = curBlock % blockDimN_;

    // 处理 tail split (M/N 方向尾块不足 baseM/baseN)
    uint64_t curSingleM = (mIdx == blockDimM_ - 1 && mTail_ > 0)
                          ? mTail_ : baseM_;
    uint64_t curSingleN = (nIdx == blockDimN_ - 1 && nTail_ > 0)
                          ? nTail_ : baseN_;

    blockCoord = BlockCoord{mIdx, nIdx,
                             (mIdx == blockDimM_ - 1 && mTail_ > 0) ? 1 : 0,
                             (nIdx == blockDimN_ - 1 && nTail_ > 0) ? 1 : 0};
    coreRoundIdx_++;
    return true;
}
```

**替代方案**: 如果修改 ASWT scheduler 风险过高，可以在 `ProcessSingleGroup` 的确定性分支中 bypass scheduler，手动循环 tile 索引：

```cpp
// 替代方案: 在 ProcessSingleGroup 确定性分支中手动计算 tile:

if (isDeter_ && ASCEND_IS_AIV) {
    // 手动行优先调度: 每个 core 独立计算自己的 tile 序列
    uint64_t blockDimM = CeilDiv(m, params.gmmParams.baseM);
    uint64_t blockDimN = CeilDiv(n, params.gmmParams.baseN);
    uint64_t totalTiles = blockDimM * blockDimN;
    uint64_t coreIdx = GetBlockIdx();
    uint64_t coreNum = params.gmmParams.coreNum;  // AIC 数

    for (uint64_t curBlock = coreIdx; curBlock < totalTiles; curBlock += coreNum) {
        uint64_t mIdx = curBlock / blockDimN;
        uint64_t nIdx = curBlock % blockDimN;
        // 计算实际单 tile 大小 (含尾块)
        // 执行 AIC: WaitForVector + MatMul + NotifyVector
        // 执行 AIV: WaitForCube + Epilogue + NotifyCube
    }
}
```

**推荐**: 使用 scheduler 的 `SetRowPriority` 方案，避免绕过 ASWT 的 tail-split 处理逻辑。

---

## 8. 同步机制适配

### 8.1 A3 vs A5 同步原语对照

```
操作                A3 (mode=2)                    A5 (mode=4)                   确定性用途
────────────────────────────────────────────────────────────────────────────────────────
全局同步            SyncAll()                      SyncAll<false>()              窗口 flush 首尾
AIC→AIV 通知        CrossCoreSetFlag<2,PIPE_FIX>(5) CrossCoreSetFlag<4,PIPE_FIX>(4)  MatMul→Dequant
                                                        + CrossCoreSetFlag(20)
AIV→AIC 通知        CrossCoreSetFlag<2,PIPE_MTE2>(3) CrossCoreSetFlag<4,PIPE_V>(6)   Dequant→MatMul
AIC 等待 AIV        CrossCoreWaitFlag<2,PIPE_FIX>(3) CrossCoreWaitFlag(6)+WaitFlag(22) workspace复用
AIV 等待 AIC        CrossCoreWaitFlag<2,PIPE_FIX>(5) CrossCoreWaitFlag<4,PIPE_V>(4)   等MatMul完成
```

### 8.2 新增同步点

确定性模式下新增 3 个 `SyncAll<false>()` 同步点：

```
operator() 流程中的同步:

  Prologue (AIV)
  ──────── SyncAll<false>() ────────  ← 已有 (line 344)
  for each group:
      ProcessSingleGroup()             ← AIC/AIV tile 级同步 (已有 WaitForVector/NotifyVector/WaitForCube/NotifyCube)
      VectorSync()                     ← 确定性滑窗检查
          ─── SyncAll<false>() ───     ← ★ NEW: 等所有核完成窗口内计算
          FRDeterministic()            ← 批量 flush
          ─── SyncAll<false>() ───     ← ★ NEW: flush 完成, 窗口释放
  FRDeterministic()                    ← 最终收尾 flush
      ─── SyncAll<false>() ───         ← ★ NEW
      ... flush ...
      ─── SyncAll<false>() ───         ← ★ NEW
  End()
```

### 8.3 不影响的现有同步

- A5 的 MTE2/V/MTE3 三级流水事件 (`SetFlag<MTE2_V>`, `WaitFlag<V_MTE3>`, 等) **保持不变**
- tile 级别的 AIC↔AIV 同步 (mode=4, flag=4/6/20/22) **保持不变**
- Prologue 的 `SyncAll<false>()` **保持不变**

---

## 9. 完整数据流图 (确定性模式)

### 9.1 两阶段数据流

```
阶段1: 计算 + 线性写入 (每个 tile 重复)
════════════════════════════════════════

AIC Core k                    AIV Core j (subBlockIdx 0/1)
──────────                    ──────────────────────────
WaitForVector()               WaitForCube()
  (等AIV消费完前一tile)         (等AIC完成当前tile MatMul)
MatMul(A, B) → l0cOutUb_      │
  int8×int8 → int32           │ L0C→VECIN 直通
NotifyVector()                │
                              ▼
                              MTE2: CopyInLogit(logitUb)
                              MTE2: CopyX2Scale(x2ScaleUb)
                              MTE2: CopyX1Scale(x1ScaleUb)
                              MTE2: CopyBias(biasUb)
                              SetFlag<MTE2_V> / WaitFlag<MTE2_V>
                              
                              V: VFDoDequant (int32→float, ×scale, +bias)
                                 → l0cOutUbFloat_
                              V: VFDoLogitMuls (×logit by BRC_B32)
                                 → outUbPing/Pong
                              
                              ★ 确定性分支:
                              VectorAtomicProcess(deterministic):
                                DataCopyPad2D(
                                  mmQuantOutGm_[yOffset],  ← 线性地址
                                  outUb,                     ← UB 中结果
                                  dimParams, n_)             ← 整块拷贝
                                → 无 AtomicAdd
                                → 写回 mmQuantOutGm (顺序排列)

                              SetFlag<V_MTE3> / WaitFlag<V_MTE3>
NotifyCube()


阶段2: 滑窗触发 + 批量回写 (窗口满时触发)
════════════════════════════════════════

VectorSync 触发条件:
  curGroupM + groupM > lowBoundM
  
  ── SyncAll<false>() ──  ← 所有 AIV 核同步
  
  FRDeterministic:
    for mOffset in [0..totalM):
      outRow = rowIndexGm_[windowStart + mOffset]
      if outRow % coreNumVec == GetBlockIdx():
        for nOffset in [0..n step baseN]:
          DMA: mmQuantOutGm_[mOffset*n + nOffset] → deterDmaUb_
          AtomicAdd: deterDmaUb_ → yGm[outRow * n + nOffset]
  
  ── SyncAll<false>() ──  ← flush 完成
  
  lowBoundM += windowSize  ← 窗口前移
```

### 9.2 滑窗数值模拟 (A5)

```
场景: n=7168, groupNum=4, M=[2048,512,1024,2560]
      workspace=96MB → windowSize=3511 行
      baseM=128, baseN=256 (CalBasicBlock 动态计算)
      coreNum=4 (aicNum), taskRation=2 → coreNumVec=8

Timeline:
────────────────────────────────────────────────────────────────
Group 0 (M=2048):  ProcessSingleGroup → curBlockM=2048 ≤ 3511 → 不触发
  AIV 写入 mmQuantOutGm[0..2048×7168-1]
  
Group 1 (M=512):   ProcessSingleGroup → curBlockM=2560 ≤ 3511 → 不触发
  AIV 写入 mmQuantOutGm[2048×7168..2560×7168-1]
  
Group 2 (M=1024):  ProcessSingleGroup → curBlockM=3584 > 3511 → ★ 触发!
  
  VectorSync:
    curGroup: G0(2048) + G1(512) = 2560 → curM=2560
    G2 partial: lowBoundM(3511) - 2560 = 951
    对齐到 baseM(128): curM = 2560 + 896 = 3456
    
    FRDeterministic:
      SyncAll()
      totalM = 3456 - (3511-3511) = 3456 行
      → 8 个 AIV 子核: 按 outRow%8 分配
      → 每个行: DMA→deterDmaUb_→AtomicAdd→yGm
      SyncAll()
    
    lowBoundM = 3456 + 3511 = 6967
    
  Group 2 剩余 (128 行) 继续处理 → 写入 mmQuantOutGm[开头复用]
  
Group 3 (M=2560):  ProcessSingleGroup → curBlockM=6144 ≤ 6967 → 不触发

最终收尾:
  FRDeterministic:
    SyncAll()
    totalM = 6144 - (6967-3511) = 2688 行
    → 逐行 flush
    SyncAll()
```

---

## 10. 实施步骤 (按顺序)

### Phase 1: TilingData & Tiling (Host)

| 步骤 | 文件 | 变更 | 风险 |
|------|------|------|------|
| 1.1 | `arch35/grouped_matmul_finalize_routing_tiling_data.h` | `GMMFinalizeRoutingDataParams` 增加 deterministicFlag + deterWorkspaceSize | 低 — 复用 reserved 字段 |
| 1.2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 声明 `GetWorkspaceSize()` override + 成员变量 `deterWsSize_` | 低 |
| 1.3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 实现 `GetWorkspaceSize()` + `DoOpTiling()` 增加确定性参数填充 | 中 — workspace 大小计算需验证 |
| 1.4 | 编译 Host 侧 | 验证 tiling 参数正确下发 | — |

### Phase 2: Kernel 入口 & 数据结构

| 步骤 | 文件 | 变更 | 风险 |
|------|------|------|------|
| 2.1 | `arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | GMMTiling 扩展 + EpilogueParams 扩展 | 低 |
| 2.2 | `kernel_gmm_finalize_routing_pertoken_dequant.h` | GMMTiling 结构体扩展 + SyncConfig 定义 + 新增成员变量 | 低 |

### Phase 3: Epilogue 核心变更

| 步骤 | 文件 | 变更 | 风险 |
|------|------|------|------|
| 3.1 | `block_epilogue_dequant_finalize_routing.h` | Params 扩展 + 新增成员变量 | 低 |
| 3.2 | 同上 | `Init()` 增加 deterDmaUb_ 分配 | 中 — UB 布局需验证不冲突 |
| 3.3 | 同上 | `VectorAtomicProcess()` 增加确定性分支 | 中 — 地址计算需精确 |
| 3.4 | 同上 | 新增 `InitDeterministic()`, `DeterministicFlushRow()`, `GetRowIndex()` | 中 — DMA 流水需正确 |

### Phase 4: Kernel 核心流水

| 步骤 | 文件 | 变更 | 风险 |
|------|------|------|------|
| 4.1 | `kernel_gmm_finalize_routing_pertoken_dequant.h` | `operator()` 增加确定性初始化 + VectorSync 调用 | 高 — 同步时序关键 |
| 4.2 | 同上 | 新增 `VectorSync()`, `FRDeterministic()`, `GetGroupM()` | 高 — 窗口计算需精确 |

### Phase 5: Scheduler 适配

| 步骤 | 文件 | 变更 | 风险 |
|------|------|------|------|
| 5.1 | `block_scheduler_gmm_aswt_with_tail_split.h` | 增加 `SetRowPriority()`, `GetTileIdxRowPriority()` | 高 — 可能影响非确定性路径 |

### Phase 6: 集成测试

| 步骤 | 内容 | 验证方法 |
|------|------|----------|
| 6.1 | 小 shape 单 group 测试 | M=128, N=256, 输出对比 golden |
| 6.2 | 多 group 窗口不触发测试 | M_sum < windowSize, 验证一次 flush |
| 6.3 | 多 group 窗口多次触发测试 | M_sum >> windowSize, 验证多次 flush |
| 6.4 | 多核路由冲突测试 | 多个 token 路由到同一 outRow |
| 6.5 | 10 次重复运行结果对比 | bitwise 完全一致 |
| 6.6 | 性能回归测试 | vs 非确定性路径延迟 |

---

## 11. 风险分析与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **UB 布局冲突**: deterDmaUb_ 可能超出 VECOUT 容量 | 编译/运行失败 | 先静态计算 UB 总使用量，确认在 VECOUT 限制内 |
| **地址计算错误**: mmQuantOutGm 偏移与 A3 不同 (无 Cube workspace 前缀) | 数据错位 | 显式使用 SYS_WORKSPACE_SIZE 常量，单元测试先行 |
| **ASWT 行优先兼容性**: 修改 scheduler 可能影响 tail-split 逻辑 | 尾部 tile 分配错误 | 优先使用"手动行优先循环"替代方案 |
| **SyncAll 性能影响**: 每次窗口 flush 增加 2 个全局同步 | 延迟增加 | 增大 windowSize 减少 flush 频率 (L2 允许的范围内) |
| **AIC/AIV 分工**: AIC 不参与 FRDeterministic，窗口内的 rowIndex 读取和 yGm 写入全在 AIV | AIC 空闲等待 | 可在 AIC 等待窗口 flush 期间预加载下一窗口数据 |
| **无 workspace 的 Cube 中转**: A5 L0C→VECIN 直通意味着没有 workspace slot 轮转 | 与 A3 不同 | 确认同步机制不需要修改 — A5 的 WaitForVector/WaitForCube 已处理 tile 间依赖 |
| **SyncConfig 跨 group 状态**: curGroup/curGroupM 依赖 groupListGm | 组边界判断错误 | groupListType=0/1 分别处理，参考 A3 的 groupTokensGm 读取逻辑 |

---

## 12. 代价与收益估算

| 维度 | A5 非确定性 (当前) | A5 确定性 (迁移后) | 增量 |
|------|---------------------|----------------------|------|
| **额外 Workspace** | 0 | 64~96 MB L2 | +64~96 MB |
| **额外 UB** | 0 | 12 KB (deterDmaUb_) | +12 KB |
| **额外同步** | 0 | 2×SyncAll / 窗口 | ~2-4 次 SyncAll (典型值) |
| **调度策略** | ASWT 自适应 | 行优先 (确定性模式) | L2 缓存命中率可能下降 |
| **写回方式** | 即时 AtomicAdd scatter | 延迟批量 flush | 首 token 延迟增加 |
| **数值确定性** | ✗ | ✓ | — |
| **新增代码量** | — | ~300 行 (估计) | — |

---

## 13. 关键常量速查

| 常量 | 建议值 | 说明 |
|------|--------|------|
| `DETER_WORK_SPACE_SIZE` | 96 × 1024 × 1024 (96MB) | 大 L2 (>96MB) 时的确定性工作空间 |
| `DETER_WORK_SPACE_LOWER_SIZE` | 64 × 1024 × 1024 (64MB) | 小 L2 (≤96MB) 时的确定性工作空间 |
| `DETER_UB_SIZE` | 12 × 1024 (12KB) | 确定性 DMA bounce buffer |
| `SYS_WORKSPACE_SIZE` | 20 × 1024 × 1024 (20MB) | System RPC 保留空间 (含 msg/notify/pipe) |

---

## 14. 与 A3 确定性实现的差异一览

```
┌─────────────────────┬──────────────────────────────┬──────────────────────────────┐
│      维度            │     A3 确定性                  │     A5 确定性 (迁移后)         │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ mmQuantOutGm 偏移    │ workspace + parallNum×baseM   │ workspace + SYS_WORKSPACE_   │
│                     │ ×baseN×sizeof(int32)×coreNum  │ SIZE (20MB)                  │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ UB DMA 机制          │ queBind (TQueBind VECIN+VECOUT)│ deterDmaUb_ (手动 DMA bounce) │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 调度器               │ MNBlockIdxCompute (手动公式)  │ ASWT + SetRowPriority        │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ VectorSync 触发点    │ 每个 tile 后 (while 循环内)    │ 每个 group 后 (for 循环内)    │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ Cube 输出            │ GM workspace 中转              │ L0C→VECIN 直通 (零搬运)       │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 同步 mode            │ mode=2                       │ mode=4                       │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 反量化计算位置        │ UB 级 (AscendDequant API)     │ VF 寄存器级 (MicroAPI)        │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 双缓冲               │ TQue 队列 (6个队列)            │ 手动 ping-pong (HardEvent)    │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ Dequant + Logit     │ 在 UB 上 Mul 合并              │ VFDoDequant + VFDoLogitMuls  │
│                     │                              │ (寄存器级分离)               │
└─────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

---

## 附录A: 涉及修改的文件清单

| 文件 | 变更类型 | 代码量 (估计) |
|------|----------|---------------|
| `arch35/grouped_matmul_finalize_routing_tiling_data.h` | 修改 GMMFinalizeRoutingDataParams | +2 fields |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 新增 GetWorkspaceSize 声明 | +1 method |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 实现 GetWorkspaceSize + DoOpTiling 修改 | ~30 行 |
| `arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | GMMTiling + Params 扩展 | ~20 行 |
| `kernel_gmm_finalize_routing_pertoken_dequant.h` | GMMTiling + SyncConfig + 新增成员函数 | ~150 行 |
| `block_epilogue_dequant_finalize_routing.h` | Params + UB + VectorAtomicProcess + 新函数 | ~100 行 |
| `block_scheduler_gmm_aswt_with_tail_split.h` | SetRowPriority + GetTileIdxRowPriority | ~50 行 |
| **总计** | | **~350 行** |

## 附录B: 调试检查点

1. **Tiling 参数验证**: 打印 `deterministicFlag` / `deterWorkspaceSize` 确认正确下发
2. **Workspace 地址验证**: 在 Kernel 中打印 mmQuantOutGm 地址，确认在 workspace + SYS_WORKSPACE_SIZE 位置
3. **UB 布局验证**: 确认 deterDmaUb_ 的物理地址不与其他缓冲区重叠
4. **窗口边界验证**: 打印 `lowBoundM` / `curM` / `windowSize` 确认窗口计算正确
5. **FRDeterministic 逐行验证**: 打印前 10 个 outRow 确认路由正确
6. **确定性验证**: 重复运行 10 次，输出 bitwise 对比
