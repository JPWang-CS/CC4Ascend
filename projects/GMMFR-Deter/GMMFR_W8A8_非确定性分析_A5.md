# GMMFR (GroupedMatmulFinalizeRouting) W8A8 INT8 非确定性实现分析 — A5

> 分析目标: A5 (Ascend 950 / DAV3510 / arch35) 平台上的 W8A8 (INT8 activation × INT8 weight) 模式
> 源码位置: `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/`
>
> **分析范围**: A5 (arch35) 的 W8A8 全量化路径，通过 `GroupedMatmulFinalizeRoutingQuantTiling` (registerList={1}) 分发，内核入口为 `grouped_matmul_finalize_routing_apt.cpp` ( `__CCE_AICORE__ == 310` )。核心采用 Cgmct 模板框架，**无确定性窗口机制**。

---

## 目录
1. [算子总体架构](#1-算子总体架构)
2. [Tiling 策略与 Key 分发](#2-tiling-策略与-key-分发)
3. [Cgmct 内核执行流程](#3-cgmct-内核执行流程)
4. [Epilogue 反量化与 Scatter-Add 详解](#4-epilogue-反量化与-scatter-add-详解)
5. [非确定性根因分析](#5-非确定性根因分析)
6. [A5 硬件 Buffer 分配分析](#6-a5-硬件-buffer-分配分析)
7. [与 A3 确定性实现的对比](#7-与-a3-确定性实现的对比)
8. [代码路径索引](#8-代码路径索引)

---

## 1. 算子总体架构

### 1.1 功能定义

A5 平台的 GMMFR W8A8 与 A3 版本功能一致——GroupedMatmul + MoeFinalizeRouting 融合算子：
- 多个 expert 的权重矩阵与输入 token 的矩阵乘法 (GroupedMatmul)
- 对 Matmul int32 输出执行 **per-channel + per-token 反量化** 得到 float32
- 乘以 logit 后进行 **scatter-add** 写入输出 (FinalizeRouting)

关键区别：A5 版本**不保证确定性**。

### 1.2 执行模型

采用 **AIC + AIV 混合编程模型** (`KERNEL_TYPE_MIX_AIC_1_2`)，AIC:AIV 核数比为 1:2：

```
┌──────────────────────────────────────────────────────────────┐
│                    AIC (Cube 核) × N                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  BlockMmad: INT8 Matmul (MMAD 指令)                    │  │
│  │  x(int8, ND) × w(int8, NZ) → l0cOut(int32)            │  │
│  │  通过 BlockScheduler 按 group 迭代 tile               │  │
│  └────────────────────────────────────────────────────────┘  │
│                            │                                  │
│          CrossCoreSetFlag (SYNC_AIC_TO_AIV)                  │
│                            ▼                                  │
│                    AIV (Vector 核) × 2N                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  BlockPrologueFinalizeRouting (一次):                   │  │
│  │    - 输出张量置零 (InitOutputWithZeros)                 │  │
│  │    - sharedInput 缩放复制 (VFDoSharedCastAndMuls)       │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  BlockEpilogueDequantFinalizeRouting (每 tile):         │  │
│  │    1. CopyInLogit / x2Scale / x1Scale / bias → UB      │  │
│  │    2. VFDoDequant: int32→fp32, ×x2Scale, ×x1Scale,     │  │
│  │       +bias                                             │  │
│  │    3. VFDoLogitMuls: ×logit (per-row 广播)             │  │
│  │    4. VectorAtomicProcess: atomic scatter-add → y      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 Cgmct 框架层次

A5 实现完全基于 **Cgmct (CANN Generic Matrix Compute Toolkit)** 模板框架，而非手写 AscendC：

```
Host (CPU):
  GroupedMatmulFinalizeRoutingQuantTiling
    └── GroupedQmmTiling
          └── TilingBaseClass
          ├── CalBasicBlock() → 计算 L0 Tile Shape
          ├── CalL1Tiling()   → 计算 L1 分块策略
          └── GetTilingKey()  → 生成 TilingKey

Device (NPU):
  grouped_matmul_finalize_routing_apt.cpp (入口)
    └── grouped_matmul_finalize_routing_pertoken_dequant<layoutA, layoutB, scaleType, rowIndType>()
          └── KernelGmmFinalizeRoutingPertokenDequant
                ├── BlockMmadBuilder → BlockMmad (AIC MMAD)
                ├── BlockPrologueFinalizeRouting (AIV 初始化)
                ├── BlockEpilogueDequantFinalizeRouting (AIV 反量化)
                └── GroupedMatmulAswtWithTailSplitScheduler (调度器)
```

### 1.4 关键源文件布局

```
op_kernel/arch35/
├── grouped_matmul_finalize_routing_pertoken_dequant.h  ★ W8A8 内核入口
├── grouped_matmul_finalize_routing.h                   (MX/Float8 入口)
├── grouped_matmul_finalize_routing_tiling_data.h       TilingData 结构体
├── grouped_matmul_finalize_routing_tiling_key.h        TilingKey 模板参数
├── common/
│   ├── tool.h                                          ★ 常量和工具函数
│   ├── basic_block_config.h                            WqmmConfig 等配置
│   └── basic_block_vf_mx.h                             MX 向量函数参数
└── weight_quant_basic_block/                           (W8A4/W4A8 场景)

op_host/op_tiling/arch35/
├── grouped_matmul_finalize_routing_quant_tiling.cpp    ★ W8A8 Tiling 实现
├── grouped_matmul_finalize_routing_quant_tiling.h      ★ Tiling 类声明
├── grouped_matmul_finalize_routing_weight_quant_tiling.cpp (权重量化 Tiling)
└── grouped_matmul_finalize_routing_weight_quant_tiling.h

gmm/common/cgmct/
├── kernel/kernel_gmm_finalize_routing_pertoken_dequant.h ★ Cgmct 内核
├── kernel/kernel_gmm_finalize_routing.h                  (MX 内核)
├── epilogue/block_epilogue_dequant_finalize_routing.h    ★ 反量化 Epilogue
├── epilogue/block_epilogue_finalize_routing.h            (无 dequant Epilogue)
├── prologue/block_prologue_finalize_routing.h            ★ Prologue
└── block/block_mmad_builder.h                            ★ MMAD Builder
```

---

## 2. Tiling 策略与 Key 分发

### 2.1 Tiling 分发入口

`op_host/grouped_matmul_finalize_routing_tiling.cpp:48-64` 根据 NPU 架构分发：

```cpp
if (compileInfoPtr->npuArch == NpuArch::DAV_3510) {
    // A5 路径
    if (GetSizeByDataType(xDType) != GetSizeByDataType(weightDtype)) {
        // x 和 w 不同位宽 → 权重量化路径 (如 FP8×FP4)
        tilingRegisterList = {WEIGHT_QUANT_TILING};
    } else {
        // x 和 w 相同位宽 → 全量化路径 (W8A8 属于此路径)
        tilingRegisterList = {1};  // GroupedMatmulFinalizeRoutingQuantTiling
    }
} else {
    tilingRegisterList = {0};  // A2/A3 BaseTiling
}
```

W8A8 (INT8 × INT8) 中 x 和 w 都是 1 byte，走 `registerList={1}` 即 `GroupedMatmulFinalizeRoutingQuantTiling`。

### 2.2 TilingKey 编码

与 A3 的 20-bit 模式不同，A5 使用 4 个独立模板参数生成 TilingKey：

```cpp
// op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h:23-32
ASCENDC_TPL_ARGS_DECL(GroupedMatmulFinalizeRouting,
    ASCENDC_TPL_UINT_DECL(ATRANS,      2bit, [0,1]),           // transpose_x
    ASCENDC_TPL_UINT_DECL(BTRANS,      2bit, [0,1]),           // transpose_w
    ASCENDC_TPL_UINT_DECL(SCALETYPE,   2bit, [0,1,2]),         // scale dtype
    ASCENDC_TPL_UINT_DECL(ROWINDEXTYPE, 2bit, [0,1]));         // row_index dtype
```

| 参数 | 取值 | 含义 |
|------|------|------|
| ATRANS | 0 | 不转置 (W8A8 固定) |
| BTRANS | 0/1 | weight 格式: 0=NZ, 1=Zn |
| SCALETYPE | 0/1/2 | 0=fp8_e8m0(MX), 1=float, 2=bf16 |
| ROWINDEXTYPE | 0/1 | 0=int64, 1=int32 |

SCALETYPE 由 host 端根据 scale 输入 dtype 决定：

```cpp
// quant_tiling.cpp:117-121
if (inputParams_.scaleDtype == ge::DT_FLOAT) {
    scaleType_ = 1;
} else if (inputParams_.scaleDtype == ge::DT_BF16) {
    scaleType_ = 2;  // 2 represents bf16
}
```

**W8A8 有效组合**：ATRANS=0 × BTRANS∈{0,1} × SCALETYPE∈{1,2} × ROWINDEXTYPE∈{0,1} = **8 种**。

### 2.3 Tiling 参数计算

`GroupedMatmulFinalizeRoutingQuantTiling` 继承自 `GroupedQmmTiling`，核心流程：

```
GetShapeAttrsInfo() → AnalyzeInputs() → DoOpTiling() → DoLibApiTiling() → PostTiling()
```

**DoOpTiling** (quant_tiling.cpp:445-459)：填充 GMM 元数据到 `tilingData_`：
- `groupNum`, `batch`, `sharedInputOffset`, `sharedInputLen`, `residualScale`
- `aQuantMode`, `bQuantMode`, `hasBias`, `groupListType`

**DoLibApiTiling** (quant_tiling.cpp:468-509)：填充 Matmul Tiling 到 `tilingData_.matmulTiling`：
- 继承自父类 `GroupedQmmTiling::CalL1Tiling()` 计算 `baseM/baseN/baseK`
- 设置 `stepM/stepN/stepKa/stepKb` 等 L1 分块步长
- 设置 `depthA1/depthB1` (L1→L0 传输深度)
- 设置 `singleCoreM/singleCoreN/singleCoreK` (各 core 分到的数据量)
- **double buffer 全开**: `dbL0A=2, dbL0B=2`

```cpp
// quant_tiling.cpp:492-494
tilingData_.matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
tilingData_.matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
tilingData_.matmulTiling.dbL0C = basicTiling_.dbL0c;
```

关键点：**没有确定性相关参数** — 没有 `deterWorkspaceSize`、没有 `windowSize`、没有 `DETER_UB_SIZE`。

### 2.4 Shape 信息提取

```cpp
// quant_tiling.cpp:403
SetMKN(xShape, wShape)
// 从 x shape (m, k) 和 w shape (e, k, n) 提取:
//   mSize = xShape[0]  (总 token 数)
//   kSize = xShape[1]  (hidden dim)
//   nSize = wShape[2]  (intermediate dim)
//   groupNum = wShape[0]  (expert 数)
```

---

## 3. Cgmct 内核执行流程

### 3.1 入口分发

`op_kernel/grouped_matmul_finalize_routing_apt.cpp:94-102`，对 W8A8 (INT8，非 FP8/FP4)，进入 PerTokenDequant 路径：

```cpp
// apt.cpp:94
if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == 1 && ROWINDEXTYPE == 0) {
    grouped_matmul_finalize_routing_pertoken_dequant<
        Cgmct::Gemm::layout::RowMajor,   // LayoutA
        Cgmct::Gemm::layout::Nz,         // LayoutB
        1,                                // scaleType=float
        0>(                               // rowIndType=int64
        x, w, scale, bias, pertoken_scale, group_list, share_input,
        logit, row_index, offset, y, workspaceGM, tilingGM);
}
```

模板参数决定：
- **LayoutA**: 固定 `RowMajor` (ND 格式)
- **LayoutB**: `Nz` (BTRANS=0) 或 `Zn` (BTRANS=1)
- **SCALETYPE**: `1`=float, `2`=bf16
- **ROWINDEXTYPE**: `0`=int64, `1`=int32

### 3.2 Kernel 类型组装

`grouped_matmul_finalize_routing_pertoken_dequant.h:38-91` 组装类型链：

```cpp
// Step 1 — 数据类型映射
using AType = DTYPE_X;                     // int8_t (编译期宏)
using BType = DTYPE_W;                     // int8_t (编译期宏)
using CType = DTYPE_Y;                     // float (编译期宏)
using C1Type = int32_t;                    // Matmul 中间输出
using weightscaleType = float 或 bfloat16_t;
using xscaleType = float;                  // per-token scale
using rowIndexType = int32_t 或 int64_t;
using BiasType = bfloat16_t;

// Step 2 — BlockScheduler
using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;

// Step 3 — BlockMmadBuilder (AIC 部分)
using BlockMmadBuilder = Block::BlockMmadBuilder<
    AType, LayoutA, BType, LayoutB, C1Type, LayoutC,
    BiasType, LayoutBias, L1TileShape, L0TileShape,
    BlockScheduler, MatmulMultiBlock<>,
    Tile::TileCopy<Arch::DAV3510, CopyInAndCopyOutSplitMWithParams>>;

// Step 4 — Prologue + Epilogue
using BlockPrologue = BlockPrologueFinalizeRouting<CType, BiasType>;
using BlockEpilogueDequant = BlockEpilogueDequantFinalizeRouting<
    CType, C1Type, weightscaleType, xscaleType, BiasType, rowIndexType>;

// Step 5 — 完整 Kernel
using GmmKernel = KernelGmmFinalizeRoutingPertokenDequant<
    ProblemShape, BlockMmadBuilder, BlockPrologue,
    BlockEpilogueDequant, BlockScheduler>;
```

### 3.3 Kernel 执行主循环

`KernelGmmFinalizeRoutingPertokenDequant::operator()` 核心流程：

```
┌────────────────────────────────────────────────────────┐
│ 1. AIV 执行 Prologue                                   │
│    - zeroInit + sharedInput 缩放复制                    │
│    - SyncAll 等待所有 core 完成                         │
├────────────────────────────────────────────────────────┤
│ 2. 按 group 迭代 (for groupIdx in 0..groupNum-1)       │
│  ┌──────────────────────────────────────────────────┐  │
│  │ a. UpdateGroupParams: 读 groupList 获取当前 group │  │
│  │    的 M 大小 (cumsum 或 direct size)              │  │
│  │ b. UpdateOffset: 推进 A/B/C/Bias/Scale 的 GM     │  │
│  │    基地址                                         │  │
│  │ c. ProcessSingleGroup: 按 tile 迭代               │  │
│  │    ┌────────────────────────────────────────────┐ │  │
│  │    │ AIC:                                        │ │  │
│  │    │   - BlockMmad: int8×int8 MMAD              │ │  │
│  │    │   - 输出 int32 → l0cOutUb (L0C→UB mapped)   │ │  │
│  │    │   - CrossCoreSetFlag → AIV                  │ │  │
│  │    ├────────────────────────────────────────────┤ │  │
│  │    │ AIV:                                        │ │  │
│  │    │   - CrossCoreWaitFlag ← AIC                │ │  │
│  │    │   - BlockEpilogueDequant: 反量化+logit      │ │  │
│  │    │     +原子scatter                           │ │  │
│  │    │   - CrossCoreSetFlag → AIC                  │ │  │
│  │    └────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────┤
│ 3. SyncAll 结束                                        │
└────────────────────────────────────────────────────────┘
```

### 3.4 BlockScheduler 的调度策略

`GroupedMatmulAswtWithTailSplitScheduler` 是关键的调度器：

- **ASWT** = Adaptive Split With Tail：自适应的尾部分裂策略
- 将每个 group 的 M×N 矩阵分块为 baseM×baseN 的 tile
- **Tail Split**: 当 M 或 N 方向有尾块 (不能整除 baseM/baseN 的剩余部分) 时，动态调整 tile 尺寸以覆盖尾块
- 不同 core 并行处理不同 group 的不同 tile
- **调度是非确定性的**: 哪个 core 处理哪个 tile 取决于硬件调度和完成顺序

这意味着：对于同一输入，不同运行中 tile 的完成顺序可能不同，导致后续的 atomic scatter-add 到达顺序不同。

---

## 4. Epilogue 反量化与 Scatter-Add 详解

### 4.1 数据流向

```
L0C (MMAD 输出)              UB (Vector 工作区)                GM (输出)
┌──────────────┐            ┌─────────────────────┐         ┌──────────┐
│ int32[128][256]│ ─CopyIn→ │ l0cOutUb (int32)     │         │          │
│ (M×N tile)    │           │        │              │         │          │
└──────────────┘           │        ▼              │         │          │
                           │ Cast: int32→float     │         │          │
GM (Scale/Bias)            │   × x2Scale (per-N)   │         │          │
┌──────────────┐           │   × x1Scale (per-M)   │         │          │
│ x2Scale[N]   │ ─CopyIn→  │   + bias (per-N)      │         │          │
│ x1Scale[M]   │ ─CopyIn→  │        │              │         │          │
│ bias[N]      │ ─CopyIn→  │        ▼              │         │          │
│ logit[M]     │ ─CopyIn→  │ dequant float[M][N]   │         │          │
└──────────────┘           │        │              │         │          │
                           │        ▼              │         │          │
                           │  × logit (per-M)      │         │          │
                           │        │              │         │          │
                           │        ▼              │         │          │
                           │ Atomic scatter-add ──→│ y[rows][N] │
                           └─────────────────────┘         └──────────┘
```

### 4.2 VFDoDequant 反量化微架构

`BlockEpilogueDequantFinalizeRouting` 中有四种反量化路径 (根据是否有 x1Scale 和 bias)：

```
VFDoDequant<hasBias>:
  for each row m in tile:
    load l0cOutReg = l0cOutUb[m]  (int32)
    cast l0cOutReg → float
    load x2ScaleVec = x2Scale[0..N-1]  (float/bf16)
    l0cOutReg *= x2ScaleVec            (per-channel 乘)
    load x1ScaleScalar = x1Scale[m]    (float)
    l0cOutReg *= broadcast(x1ScaleScalar)  (per-token 乘, DIST_BRC_B32)
    if hasBias:
      load biasVec = bias[0..N-1] (bf16)
      cast biasVec → float
      l0cOutReg += biasVec
    store l0cOutReg → l0cOutUbFloat[m]
```

对于 W8A8 的典型场景 (有 per-token scale 和 bias)，走 `VFDoDequant<true>` 路径。

### 4.3 VFDoLogitMuls

```cpp
// 对每一行 m
logitScalar = logit[m];                    // per-row logit 值
l0cOutUbFloat[m][0..N-1] *= logitScalar;  // 广播乘
```

通过 `DIST_BRC_B32` 将标量广播到向量所有元素一次完成。

### 4.4 VectorAtomicProcess — 非确定性关键

```cpp
// 伪代码
for each row m in tile:
    rowIdx = rowIndex[m];             // 此行对应的全局输出行号
    globalAddr = y + rowIdx * N;      // 目标地址
    AtomicAdd(globalAddr, l0cOutUbFloat[m]);  // ★ 原子加法
```

**所有 core 并行执行 VectorAtomicProcess，向同一个 `y` 张量做 scatter atomic-add**。

### 4.5 BlockPrologue 的 sharedInput 处理

在 kernel 启动时，Prologue 先完成输出张量的准备：

1. **ZeroInit**: 将 `y[0..batch-1][0..N-1]` 以下三部分置零：
   - `[0, sharedInputOffset)` — sharedInput 之前
   - `[sharedInputOffset + sharedInputLen, batch)` — sharedInput 之后

2. **SharedInput**: 将 `y[sharedInputOffset..sharedInputOffset+sharedInputLen][0..N-1]` 赋值为 `shareInput × residualScale`

多 core 分工：通过 `totalElements / aivNum` 平分工作量。

---

## 5. 非确定性根因分析

### 5.1 非确定性来源总结

A5 W8A8 的非确定性来自以下 **三个层次**：

| 层次 | 非确定性来源 | 影响 |
|------|------------|------|
| **调度层** | `GroupedMatmulAswtWithTailSplitScheduler` 并行调度，tile 完成顺序不确定 | 不同 core 的 tile 处理顺序不同 |
| **计算层** | 多 core 直接 scatter atomic-add 到同一个 `y` | 浮点加法不满足结合律，累加顺序影响结果末尾精度 |
| **架构层** | **无确定性窗口机制** (无 workspace 缓冲、无 FRDeterministic 同步刷出) | 无法保证写入顺序 |

### 5.2 详细分析

#### 5.2.1 浮点加法的非结合性

核心数学原因：float32 加法不满足结合律。

```
(a + b) + c ≠ a + (b + c)   (可能相差 ULP 级别)
```

多个 expert 的同一 token 输出需要累加到同一个 `y[rowIdx]` 位置。如果 token 被路由到 expert 0 和 expert 2：
- core 0 先做完 expert 0 → atomic-add 先到达
- core 1 先做完 expert 2 → atomic-add 后到达

顺序交换虽然数学上 "a+b=b+a"，但浮点 `(y + expert0_out) + expert2_out` 可能 ≠ `(y + expert2_out) + expert0_out`。

#### 5.2.2 Multi-Core 并发 Atomic Scatter

```
时间线 (非确定性):
  Core 0: --[Group 0 Tile 0]--[Group 0 Tile 1]----------------
  Core 1: --[Group 1 Tile 0]--[Group 1 Tile 1]--[Group 2 Tile 0]--
  Core 2: ----[Group 2 Tile 0]--[Group 2 Tile 1]---------------
  
  AtomicAdd 到达 rowIdx=5 的顺序可能每次运行都不同
```

#### 5.2.3 无确定性 Workspace

A5 的关键架构差异：**没有 `deterWorkspace`**。

对照 A3 实现：
```
A3:  MMCompute → DeterWorkspace (L2) → VectorSync 窗口管理
     → FRDeterministic 有序刷出 → Fixed-order scatter
     
A5:  MMAD (L0C) → Epilogue 反量化 → 直接 Atomic Scatter
     (无中间缓冲，无窗口管理，无有序刷出)
```

在 A5 的 TilingData 中：
```cpp
// arch35/grouped_matmul_finalize_routing_tiling_data.h:43-46
struct GMMFinalizeRoutingTilingData {
    GMMFinalizeRoutingDataParams gmmFinalizeRoutingDataParams;
    TCubeTiling matmulTiling;
    // ★ 没有 deterWorkspaceSize
    // ★ 没有 windowSize
    // ★ 没有 SyncConfig
};
```

Workspace 仅用于 Cgmct 框架内部 (如 AIV/AIC 同步标志、临时 buffer)，不用于确定性缓冲。

### 5.3 非确定性量化评估

| 场景 | 非确定性程度 |
|------|-----------|
| 单 expert、单 core | **可确定** (顺序固定) |
| 单 expert、多 core (不同 M-block) | **可确定** (不同行无交叠) |
| 多 expert、top-1 routing | **可确定** (每 token 路由到唯一 expert) |
| 多 expert、top-k routing (k>1) | **非确定** (同一行多 expert 累加) |
| 多 expert + sharedInput | **非确定** (sharedInput+expert 累加顺序不确定) |

核心判断：只要两个或以上的 core 可能向 **同一个 rowIdx** 做 atomic-add，结果就非确定。

### 5.4 设计意图

A5 选择非确定性是有意为之：
- **性能优先**: 去掉了确定性窗口的内存中转和同步开销
- **L2 节省**: 不需要 64MB/96MB 的确定性 workspace
- **Cgmct 框架适配**: Cgmct 是通用模板框架，天然设计为无状态、并行友好
- **适用场景**: Top-1 routing 仍然是确定性的；Top-k 场景下精度差异在 ULP 级别，对推理质量影响可忽略

---

## 6. A5 硬件 Buffer 分配分析

### 6.1 A5 (950 / DAV3510) 硬件规格

| Buffer | A5 (950) 容量 | A3 (910B) 容量 | 对比 |
|--------|-------------|---------------|------|
| UB | 256 KB | 192 KB | ↑ 1.33× |
| L1 | 512 KB | 512 KB | = |
| L0A | 64 KB | 64 KB | = |
| L0B | 64 KB | 64 KB | = |
| L0C | 256 KB | 128 KB | ↑ 2× |
| L2 | 芯片级共享 | 芯片级共享 | 架构不同 |

关键变化：**L0C 翻倍** (128KB→256KB)，**UB 增大** (192KB→256KB)。这直接导致了 A5 可以实现更大的 baseM/baseN tile，减少了 tiling 开销。

### 6.2 各 Buffer 占用分析

#### L0C (256 KB)

A5 的 L0C 翻倍使得 Matmul 可以使用更大的 L0 tile。Cgmct 框架自动选择最优 tile shape：

```
L0 Tile Shape: M0 × N0 × K0
  - M0: 通常在 64-256 范围
  - N0: 通常在 64-256 范围
  - K0: 通常在 32-128 范围
  - sizeof(int32) × M0 × N0 ≤ 256KB

最大单 tile M×N = 256KB / 4B = 65536 个元素
例如: M0=128, N0=256 → 128×256×4 = 128KB (一半 L0C, double buffer)
      M0=256, N0=256 → 256×256×4 = 256KB (单 buffer)
```

`dbL0C = basicTiling_.dbL0c` 决定是否使用 L0C double buffer。

#### L1 (512 KB)

L1 需要容纳 A 矩阵和 B 矩阵的 L1 tile：

```cpp
// tool.h:53-55
static constexpr int64_t L1_SIZE = 512;             // 512 KB
static constexpr int64_t L1_HALF_SIZE = L1_SIZE / 2; // 256 KB (A/B 各半)
```

在 W8A8 场景下：
- **L1 A tile**: `stepKa × depthA1 × baseK × baseM × sizeof(int8_t)`
- **L1 B tile**: `stepKb × depthB1 × baseK × baseN × sizeof(int8_t)` (NZ 格式可能略有额外 padding)
- `CalL1Depth()` 根据剩余 L1 空间动态调整 depth 参数

#### UB (256 KB)

UB 在 A5 上比 A3 多 64KB (256KB vs 192KB)，这给 Epilogue 的反量化提供了更大的缓冲空间。

BlockEpilogueDequantFinalizeRouting 的 UB 占用估算：

| 缓冲区 | 用途 | 估算大小 |
|--------|------|---------|
| l0cOutUb | MMAD 输出 (ping/pong) | ~64KB (double buffer) |
| l0cOutUbFloat | 反量化后 float 输出 | ~128KB |
| logitBuf | per-row logit (ping/pong) | ~1KB |
| x2ScaleBuf | per-N scale (ping/pong) | ~2KB |
| x1ScaleBuf | per-M scale | ~1KB |
| biasBuf | per-N bias (ping/pong) | ~2KB |
| **合计** | | ~198KB / 256KB |

对比 A3 确定性模式的 UB 预算 (~190KB/192KB)，A5 有更大的余量。

### 6.3 L2 / Workspace

A5 不使用确定性 workspace。Workspace 完全由 Cgmct 框架内部分配：

```cpp
// 编译期通过 compileInfo 获取 L2 大小
ascendcPlatform.GetCoreMemSize(CoreMemType::L2, compileInfoPtr->l2Size);
```

Workspace 用途：
- **Kernel 间同步标志** (AIC↔AIV cross-core flags)
- **Cgmct 内部临时缓冲** (tile scheduling 元数据)
- **无确定性缓冲需求**

L2 的释放量远大于 A3 (不需要 64MB/96MB 确定性工作区)，可分配给其他算子或更大的 batch size。

### 6.4 AIC:AIV 核数比与 L1 分配

```cpp
// quant_tiling.cpp:415-424
CheckCoreNum():
  aicNum > 0
  aivNum == 2 × aicNum    // 固定 1:2 比例
```

```
┌──────────┬──────────┬──────────┐
│  AIC 0   │  AIC 1   │  AIC 2   │  ... (N 个)
│ (Cube)   │ (Cube)   │ (Cube)   │
├──────────┼──────────┼──────────┤
│ AIV 0,1  │ AIV 2,3  │ AIV 4,5  │  ... (2N 个)
│ (Vector) │ (Vector) │ (Vector) │
└──────────┴──────────┴──────────┘
```

每个 AIC 配 2 个 AIV，AIV 负责反量化和 scatter-add 的向量计算部分。

### 6.5 Double Buffer 策略对比

| Buffer | A3 (确定性) | A5 (非确定性) |
|--------|-----------|-------------|
| L0A | db (double buffer) | db=2 (全开) |
| L0B | db | db=2 (全开) |
| L0C | db | 由 `dbL0c` 决定 |
| UB vecInQueue | 32KB | Cgmct 自动管理 |
| UB vecOutQueue | 32KB (含确定性 12KB) | Cgmct 自动管理 |
| DeterWorkspace (L2) | 64MB/96MB | **无** |

---

## 7. 与 A3 确定性实现的对比

### 7.1 架构对比总表

| 维度 | A3 W8A8 确定性 | A5 W8A8 非确定性 |
|------|--------------|----------------|
| **芯片** | 910B/910C (DAV_3510 早期) | 950 (DAV_3510/arch35) |
| **核数比** | AIC:AIV = 1:2 | AIC:AIV = 1:2 |
| **框架** | 手写 AscendC | Cgmct 模板框架 |
| **Tiling 类** | `GroupedMatmulFinalizeRoutingBaseTiling` | `GroupedMatmulFinalizeRoutingQuantTiling` |
| **内核类** | `QuantGroupMatmul` | `KernelGmmFinalizeRoutingPertokenDequant` |
| **Scheduler** | Manual tile loop | `GroupedMatmulAswtWithTailSplitScheduler` |
| **确定性窗口** | 有 (SyncConfig + VectorSync) | **无** |
| **确定性刷出** | FRDeterministic (多 core 分区 scatter) | **无** |
| **输出方式** | 有序 scatter-add (窗口内保证顺序) | 直接 atomic scatter-add (无顺序保证) |
| **中间缓冲** | L2 DeterWorkspace (64MB/96MB) | **无** (L0C → UB → GM 直通) |
| **溢出防护** | 三级: L2 Workspace + UB + L1/L0C | 仅硬件 Buffer 限制 (tiling 时保证) |
| **UB 用量** | ~190KB / 192KB | ~198KB / 256KB (估算) |
| **L0C** | 128KB (baseM×baseN=128×256) | 256KB (tile shape 由框架优化) |
| **同步机制** | VectorSync 自定义同步 + CrossCoreSetFlag | Cgmct 内置 CrossCoreSetFlag |
| **确定性** | 保证 (top-k 场景) | 不保证 (top-k > 1 时) |

### 7.2 同步机制对比

**A3 确定性同步**:
```
VectorSync():
  while (curBlockM > lowBoundM):   ← 窗口溢出检测
    FRDeterministic()               ← 多 core 同步刷出
    lowBoundM = curM + windowSize   ← 窗口滑动
  CrossCoreSetFlag(SYNC_AIC_TO_AIV)
  CrossCoreWaitFlag(SYNC_AIV_TO_AIC)
```

**A5 非确定性同步**:
```
ProcessSingleGroup():
  AIC: MMAD 完成 → CrossCoreSetFlag → AIV
  AIV: 等待 AIC → Epilogue → CrossCoreSetFlag → AIC
  (无窗口检查，无中间刷出)
```

### 7.3 Scatter-Add 实现对比

**A3**:
```
FRDeterministic():
  SyncAll → 多 core 分区负责不重叠的 row 范围
  → 每个 core 只写自己负责的 row
  → 无竞争 → 确定性顺序
```

**A5**:
```
VectorAtomicProcess():
  无 SyncAll → 所有 core 直接 atomic-add
  → 多个 core 可能写同一 row
  → 有竞争 → 顺序不确定
```

### 7.4 性能与精度权衡

| 指标 | A3 确定性 | A5 非确定性 |
|------|---------|-----------|
| L2 占用 | +64MB/96MB | 0 |
| 同步开销 | VectorSync + FRDeterministic (两次 SyncAll) | 仅 tile 级 AIC↔AIV 同步 |
| 中间访存 | 写 workspace + 读 workspace (L2) | 无 |
| 精度 | 完全确定 | ULP 级差异 (浮点累加顺序) |
| 推理质量 | 基准 | 无实际影响 (ULP << 模型量化误差) |

---

## 8. 代码路径索引

### 8.1 Tiling 分发

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_host/grouped_matmul_finalize_routing_tiling.cpp` | 48-64 | DAV_3510 分发逻辑: xDtype==wDtype → registerList={1} |
| `op_host/grouped_matmul_finalize_routing_tiling.cpp` | 89-113 | TilingPrepare: 获取 NPU arch、核数、UB/L1/L0/L2 容量 |
| `op_host/grouped_matmul_finalize_routing_tiling.cpp` | 37 | BaseTiling 注册 (registerList=0, A2/A3 路径) |

### 8.2 W8A8 Tiling 计算

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 67-125 | QuantTiling 类声明 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 39-43 | GetShapeAttrsInfo |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 84-103 | AnalyzeAttrs: 解析 transpose/residualScale 等属性 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 105-168 | AnalyzeDtype: 解析输入 dtype，确定 scaleType/rowIndexType |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 171-355 | AnalyzeInputs + CheckDim + CheckDtype |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 445-459 | DoOpTiling: 填充 GMM 元数据 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 462-466 | GetTilingKey: 4 参数模板 key |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 468-509 | DoLibApiTiling: 填充 matmulTiling |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 512-528 | PostTiling: 序列化 tiling data |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 565 | 模板注册: `REGISTER_OPS_TILING_TEMPLATE(..., 1)` |

### 8.3 TilingKey 与 TilingData 定义

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h` | 23-32 | 4 参数 TilingKey 模板声明 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 26-39 | GMMFinalizeRoutingDataParams 结构体 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 43-46 | GMMFinalizeRoutingTilingData (无确定性字段) |

### 8.4 内核入口

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 16 | `__CCE_AICORE__ == 310` (A5 编译宏) |
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 54-59 | 模板参数声明 |
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 88-180 | Kernel 入口函数: 分发到 perTokenDequant 或 MX 路径 |
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 94-101 | W8A8 8 路 if-constexpr 分发 |

### 8.5 内核实现

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 28-91 | Kernel 入口函数: 类型组装 + GmmKernel 调用 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 38-40 | L1TileShape/L0TileShape (编译期 _0 占位) |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 41-52 | 数据类型映射 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 56-58 | BlockMmadBuilder 组装 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 63-65 | BlockEpilogueDequant 组装 |

### 8.6 Cgmct 框架核心

| 文件 | 内容 |
|------|------|
| `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` | ★ 核心 Kernel 类: group 迭代, tile 调度, AIC↔AIV 同步 |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | ★ 反量化 Epilogue: VFDoDequant + VFDoLogitMuls + VectorAtomicProcess |
| `gmm/common/cgmct/epilogue/block_epilogue_finalize_routing.h` | MX Epilogue: 无 dequant, 直接 logit+scatter |
| `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h` | ★ Prologue: zeroInit + sharedInput 缩放复制 |
| `gmm/common/cgmct/block/block_mmad_builder.h` | ★ MMAD Builder: MatmulType 推导, Tile Shape 提取 |

### 8.7 硬件常量

| 文件 | 行号 | 常量 | 值 |
|------|------|------|------|
| `op_kernel/arch35/common/tool.h` | 53 | L1_SIZE | 512 KB |
| `op_kernel/arch35/common/tool.h` | 55 | L1_HALF_SIZE | 256 KB |
| `op_kernel/arch35/common/tool.h` | 56 | L1_SIZE_WITH_QUANTSCALE | 504 KB |
| `op_kernel/arch35/common/tool.h` | 79-83 | SYNC 标志 | SYNC_AIV_AIC=8, SYNC_AIC_AIV=9 |
| `op_kernel/arch35/common/basic_block_config.h` | 119-131 | UB Buffer 分配 (MXA8W4 场景) | weightLowBit=64KB, highBit=128KB |

### 8.8 对比参考 (A3 确定性)

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_kernel/grouped_matmul_finalize_routing.h` | 303 | windowSize 初始化 |
| `op_kernel/grouped_matmul_finalize_routing.h` | 630 | VectorSync 窗口溢出检测 |
| `op_kernel/grouped_matmul_finalize_routing.h` | 648 | FRDeterministic 确定性刷出 |
| `op_host/grouped_matmul_finalize_routing_base_tiling.cpp` | 413-425 | 确定性 workspace 大小计算 |
| `op_kernel/grouped_matmul_finalize_routing_utils.h` | 30 | DETER_UB_SIZE = 12KB |
| `op_kernel/grouped_matmul_finalize_routing_utils.h` | 49-56 | SyncConfig (确定性窗口配置) |

---

## 附录: A5 W8A8 完整调用链

```
HOST 端 (编译期):
  GroupedMatmulFinalizeRoutingTilingFunc()
    ├── npuArch == DAV_3510
    ├── xDtype == weightDtype (INT8) → registerList={1}
    └── GroupedMatmulFinalizeRoutingQuantTiling (extends GroupedQmmTiling)
          ├── GetShapeAttrsInfo()
          ├── AnalyzeAttrs()
          ├── AnalyzeDtype()         → scaleType_, rowIndexType_
          ├── AnalyzeInputs()
          │     ├── SetGroupNum()
          │     ├── SetMKN()
          │     └── SetQuantModeForGMMFinalizeRouting()
          │           └── bQuantMode = PERCHANNEL_MODE
          │               aQuantMode = PERTOKEN_MODE (有 pertoken_scale 时)
          │               aQuantMode = DEFAULT (无 pertoken_scale 时)
          ├── DoOpTiling()           → gmmFinalizeRoutingDataParams
          ├── DoLibApiTiling()
          │     ├── CalBasicBlock()
          │     ├── CalL1Tiling()
          │     └── → matmulTiling (TCubeTiling)
          ├── GetTilingKey()         → 4 参数模板 Key
          └── PostTiling()           → 序列化到 TilingContext

DEVICE 端 (运行时):
  grouped_matmul_finalize_routing()  [apt.cpp:61]
    └── ORIG_DTYPE_X == DT_INT8 (非 FP8/FP4)
          └── if constexpr 8 路分发:
                grouped_matmul_finalize_routing_pertoken_dequant<
                    RowMajor/Nz, RowMajor/Zn, 1/2, 0/1>()  [pertoken_dequant.h:28]
                  └── KernelGmmFinalizeRoutingPertokenDequant::operator()
                        ├── Prologue:
                        │     BlockPrologueFinalizeRouting
                        │       ├── InitOutputWithZeros (多 core 分工)
                        │       └── VFDoSharedCastAndMuls (sharedInput 缩放)
                        ├── for each group:
                        │     ├── UpdateGroupParams (读 group_list)
                        │     ├── UpdateOffset (推进全局指针)
                        │     └── ProcessSingleGroup:
                        │           ├── for each tile via BlockScheduler:
                        │           │     ├── AIC: BlockMmad (int8×int8 MMAD)
                        │           │     ├── CrossCoreSetFlag(SYNC)
                        │           │     └── AIV: BlockEpilogueDequantFinalizeRouting
                        │           │           ├── CopyInLogit / x2Scale / x1Scale / bias
                        │           │           ├── VFDoDequant: int32→float, ×scales, +bias
                        │           │           ├── VFDoLogitMuls: ×logit
                        │           │           └── VectorAtomicProcess: atomic scatter-add → y
                        └── SyncAll
```
