# GMMFR W8A8 int8 A5 链路解析 —— 从 Tiling 到 Kernel 完整分析

> **芯片**: Ascend 950 (A5, arch35, DAV_3510, `__CCE_AICORE__ == 310`, `__NPU_ARCH__ == 3510`)
> **数据类型**: activation INT8 RowMajor × weight INT8 NZ/Zn → output FLOAT32
> **确定性**: A5 W8A8 无确定性分支，直接非确定性

---

## 1. 文件结构概览 —— A5 vs A3 架构差异

A5 与 A3 的代码组织完全不同。A5 使用 APT (Ascend Programming Template) 框架 + CGMCT (Compound GEMM Compute Template) 组件库：

```
grouped_matmul_finalize_routing/
│
├── op_host/
│   ├── grouped_matmul_finalize_routing_def.cpp       # A5 注册 (ascend950, opFile=apt)
│   ├── grouped_matmul_finalize_routing_tiling.cpp     # 架构调度 (A5→模板#1 QuantTiling)
│   ├── op_tiling/arch35/
│   │   ├── grouped_matmul_finalize_routing_quant_tiling.cpp  # ★ A5 W8A8 Tiling 核心 (566行)
│   │   ├── grouped_matmul_finalize_routing_quant_tiling.h    # Tiling 类声明
│   │   └── grouped_matmul_finalize_routing_weight_quant_tiling.cpp  # MX/W8A4路径
│
├── op_kernel/
│   ├── grouped_matmul_finalize_routing_apt.cpp        # ★ A5 内核入口 (183行)
│   └── arch35/
│       ├── grouped_matmul_finalize_routing_pertoken_dequant.h  # ★ W8A8 调度包装 (92行)
│       ├── grouped_matmul_finalize_routing_tiling_data.h   # GMMFinalizeRoutingTilingData (49行)
│       ├── grouped_matmul_finalize_routing_tiling_key.h    # APT TilingKey 声明 (35行)
│       └── grouped_matmul_finalize_routing.h               # MX FP8/FP4 路径
│
└── gmm/common/cgmct/                              # ★ CGMCT 组件库
    ├── kernel/
    │   └── kernel_gmm_finalize_routing_pertoken_dequant.h   # ★ 核心 Kernel 类 (368行)
    ├── epilogue/
    │   └── block_epilogue_dequant_finalize_routing.h        # ★ Dequant + Route (563行)
    ├── block/
    │   ├── block_scheduler_gmm_aswt_with_tail_split.h       # ASWT 调度器
    │   └── block_mmad_builder.h                             # MatMul Builder
    ├── prologue/
    │   └── block_prologue_finalize_routing.h                # 输出初始化
    ├── tile/
    │   └── tile_copy_policy.h                               # Tile拷贝策略
    └── utils/
        ├── coord_utils.h          # 坐标/偏移工具
        ├── common_utils.h         # 通用工具
        ├── device_utils.h         # 设备工具
        ├── status_utils.h         # 同步事件
        └── tensor_utils.h         # Tensor工具
```

### 1.1 调用层次全景图

```
Host (Tiling)                        Device (Kernel)
══════════════                       ═══════════════

DoOpTiling()                         grouped_matmul_finalize_routing()
  │                                     │
  ├─ DoOpTiling()                       ├─ grouped_matmul_finalize_routing
  │   └─ Fill DataParams                │   _pertoken_dequant<>()
  │                                     │     │
  ├─ DoLibApiTiling()                   │     ├─ REGISTER_TILING_DEFAULT
  │   ├─ CalBasicBlock()                │     ├─ GET_TILING_DATA
  │   │   └─ GroupedQmmTiling基类       │     │
  │   └─ CalL1Tiling()                  │     ├─ 组装 CGMCT 组件:
  │       └─ L1 tiling参数              │     │   BlockMmadBuilder
  │                                     │     │   BlockPrologueFinalizeRouting
  └─ PostTiling()                       │     │   BlockEpilogueDequantFinalizeRouting
      └─ SetBlockDim + SetScheduleMode  │     │
                                        │     └─ GmmKernel(params)
                                        │         │
                                        │         ├─ operator()
                                        │         │   ├─ AIV: Prologue(归零+shared)
                                        │         │   ├─ AIC: mmadOp_.Init()
                                        │         │   ├─ SyncAll()
                                        │         │   └─ for each group:
                                        │         │       ProcessSingleGroup()
                                        │         │         ├─ AIC: WaitForVector
                                        │         │         ├─ AIC: mmadOp_(A,B,l0cOutUb)
                                        │         │         ├─ AIC: NotifyVector
                                        │         │         ├─ AIV: WaitForCube
                                        │         │         ├─ AIV: epilogueDequantOp_
                                        │         │         │   └─ VFDoDequant
                                        │         │         │       CopyInLogit
                                        │         │         │       VFDoLogitMuls
                                        │         │         │       VectorAtomicProcess
                                        │         │         └─ AIV: NotifyCube
```

### 1.2 A5 vs A3 架构关键差异

| 维度 | A3 (910B/910C) | A5 (950) |
|------|---------------|----------|
| **Tiling 模板** | BaseTiling (模板#0) | QuantTiling (模板#1) |
| **TilingData** | `GroupMatmulFRTilingData` (扁平宏) | `GMMFinalizeRoutingTilingData` (POD struct) |
| **TilingKey** | 字符串式 `100...01UL` + 偏移 | APT 模板式 `(ATRANS, BTRANS, SCALETYPE, ROWINDEXTYPE)` |
| **baseM/N/K** | 固定 128/256/128 (有特殊分支) | `CalBasicBlock()` 动态计算 |
| **Kernel 框架** | 手写 `QuantGroupMatmul<P>` | CGMCT 组件化 |
| **Dequant** | `AscendDequant` 高层API | VF微架构寄存器编程 (MicroAPI) |
| **Scheduler** | 手工 `MNBlockIdxCompute` | `GroupedMatmulAswtWithTailSplitScheduler` |
| **L0A/L0B 双缓冲** | 未显式设置 | `dbL0A=2, dbL0B=2` |
| **确定性** | 支持 | **不支持** |
| **同步事件** | mode=2, event=3/5 | mode=4, event=4/6 + flag+16 |
| **UB 布局** | TBuf偏移 96KB | VECIN/VECOUT位置 ~322KB |
| **算子注册** | binary.json | APT框架 |

---

## 2. Tiling 层完整解析

### 2.1 DoOpTiling 调用流程

```
GroupedMatmulFinalizeRoutingQuantTiling 调用链:

Reset()                                    [quant_tiling.cpp:30]
  └─ memset_s → 清零 TilingData

GetShapeAttrsInfo()                        [继承自 GroupedQmmTiling]
  ├─ AnalyzeAttrs()                        [quant_tiling.cpp:84]
  │   ├─ 解析 shareInputWeight (residualScale)
  │   ├─ 解析 transposeX / transposeW
  │   └─ CheckOptionalAttr()
  │       ├─ shareInputOffset ≥ 0?
  │       ├─ outputBS > 0?
  │       ├─ groupListType ∈ {0,1}?
  │       └─ outputDtype ∈ {float32, float16, bfloat16}?
  │
  ├─ AnalyzeDtype()                        [quant_tiling.cpp:105]
  │   ├─ x=int8, w=int8, scale=float/bf16
  │   ├─ 设置 scaleType_ (1=float, 2=bf16)
  │   ├─ 设置 rowIndexType_ (0=int64, 1=int32)
  │   ├─ 检测 hasBias
  │   └─ CheckDtype() 验证所有类型组合
  │
  └─ AnalyzeInputs()                       [quant_tiling.cpp:373]
      ├─ 解析 x[M×K], w[E×N×K], scale[E×N]
      ├─ SetGroupNum() → inputParams_.groupNum
      ├─ SetMKN() → mSize, nSize, kSize
      ├─ CheckInputsShape() → 维度验证
      ├─ CheckOptionalInputsShape() → sharedInput/rowIndex/logit验证
      ├─ SetQuantModeForGMMFinalizeRouting()  ★
      │   ├─ W8A8: bQuantMode=PERCHANNEL(2)
      │   ├─ pertoken_scale存在: aQuantMode=PERTOKEN(4)
      │   └─ pertoken_scale不存在: aQuantMode=DEFAULT(0)
      └─ CheckCoreNum() → aicNum : aivNum = 1 : 2

DoOpTiling()                               [quant_tiling.cpp:445]
  └─ Fill GMMFinalizeRoutingDataParams (12个字段)
      groupNum, batch, sharedInputOffset/Len, residualScale,
      aQuantMode, bQuantMode, biasDtype, groupListType, hasBias

DoLibApiTiling()                           [quant_tiling.cpp:468]
  ├─ CalBasicBlock()                       ★ 基类动态计算
  │   └─ 根据 M,N,K + L1/L0/UB 大小自动确定最优 baseM/baseN/baseK
  ├─ CalL1Tiling()                         ★ L1 tiling参数
  └─ Fill TCubeTiling (18个字段)
      M,N,Ka,Kb, baseM/baseN/baseK,
      singleCoreM/N/K, depthA1/B1, stepM/N/Ka/Kb,
      isBias, iterateOrder, dbL0A=2, dbL0B=2, dbL0C

PostTiling()                               [quant_tiling.cpp:512]
  ├─ SetBlockDim(aicNum)
  ├─ SetScheduleMode(1)   ← 动态调度模式
  └─ memcpy_s → RawTilingData
```

### 2.2 GetTilingKey —— APT TilingKey 编码

```cpp
// quant_tiling.cpp:462-466
uint64_t GetTilingKey() const {
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(inputParams_.transA),    // ATRANS: 固定0
        static_cast<uint64_t>(inputParams_.transB),    // BTRANS: 0(NZ) 或 1(Zn)
        static_cast<uint64_t>(scaleType_),              // SCALETYPE: 1(float) 2(bf16)
        static_cast<uint64_t>(rowIndexType_));          // ROWINDEXTYPE: 0(int64) 1(int32)
}
```

## 3. Kernel 层完整解析

### 3.1 内核入口 —— APT 模板分派

```cpp
// grouped_matmul_finalize_routing_apt.cpp:60-126
// AICORE=310, 编译期展开8个W8A8分支
template <int ATRANS, int BTRANS, int SCALETYPE, int ROWINDEXTYPE>
__global__ __aicore__ void grouped_matmul_finalize_routing(...) {
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // W8A8: ORIG_DTYPE_X=int8 ≠ FP8/FP4 → 进入第一个分支
    // 8种组合通过 constexpr if 编译期展开:
    if constexpr (ATRANS==0 && BTRANS==0 && SCALETYPE==1 && ROWINDEXTYPE==0) {
        grouped_matmul_finalize_routing_pertoken_dequant<RowMajor, Nz, 1, 0>(...);
    } else if constexpr (ATRANS==0 && BTRANS==1 && SCALETYPE==1 && ROWINDEXTYPE==0) {
        grouped_matmul_finalize_routing_pertoken_dequant<RowMajor, Zn, 1, 0>(...);
    }
    // ... 共8个分支 (SCALETYPE=1,2 × BTRANS=0,1 × ROWINDEXTYPE=0,1)
}
```

### 3.2 pertoken_dequant 调度包装 —— CGMCT 组件组装

```cpp
// arch35/grouped_matmul_finalize_routing_pertoken_dequant.h:27-91
template <typename layoutA, typename layoutB, int scaleType, int rowIndType>
void grouped_matmul_finalize_routing_pertoken_dequant(...) {
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    // === Step 1: 类型推导 (W8A8 → int8×int8 → float) ===
    using AType = DTYPE_X;          // = int8_t
    using BType = DTYPE_W;          // = int8_t
    using CType = DTYPE_Y;          // = float
    using C1Type = int32_t;         // MatMul中间累加类型
    using weightscaleType = (scaleType==2) ? bfloat16_t : float;
    using rowIndexType = (rowIndType==1) ? int32_t : int64_t;

    // === Step 2: 组装 CGMCT 组件 ===

    // [2a] BlockMmadBuilder: MatMul计算
    using BlockMmadBuilder = Block::BlockMmadBuilder<
        int8_t, RowMajor,         // AType, LayoutA
        int8_t, Nz|Zn,            // BType, LayoutB (编译期确定)
        int32_t, RowMajorAlign,   // C1Type, LayoutC
        bfloat16_t, RowMajor,     // BiasType
        L1TileShape<0,0,0>,       // L1 Tile形状 (由Scheduler确定)
        L0TileShape<0,0,0>,       // L0 Tile形状
        GroupedMatmulAswtWithTailSplitScheduler,  // Scheduler类型
        MatmulMultiBlock<>,                       // MatMul模式
        TileCopy<DAV3510, CopyInAndCopyOutSplitMWithParams>>;  // Tile拷贝策略

    // [2b] BlockPrologue: 输出初始化
    using BlockPrologue = BlockPrologueFinalizeRouting<float, bfloat16_t>;

    // [2c] BlockEpilogue: 反量化 + 路由写回
    using BlockEpilogueDequant = BlockEpilogueDequantFinalizeRouting<
        float,           // CType (输出)
        int32_t,         // DataTypeIn (MatMul输出)
        weightscaleType, // weight scale
        float,           // per-token scale
        bfloat16_t,      // bias
        rowIndexType>;   // rowIndex

    // [2d] Kernel: 组装完整流水线
    using GmmKernel = KernelGmmFinalizeRoutingPertokenDequant<
        MatmulShape, BlockMmadBuilder, BlockPrologue,
        BlockEpilogueDequant, GroupedMatmulAswtWithTailSplitScheduler>;

    // === Step 3: 填充参数 ===
    GMMTiling gmmParams{groupNum, groupListType, baseM, baseN, baseK, hasBias};
    gmmParams.matmulTiling = &matmulTiling_;

    Params params = {
        {1,1,1,1},                                           // problem shape
        {x, w, y, bias, group_list},                         // mmad params
        {share_input, y, sharedInputOffset, sharedInputLen,  // prologue params
         N, batch, residualScale},
        {y, w_scale, x_scale, bias, logit, row_index,        // epilogue params
         baseM, baseN},
        gmmParams};

    GmmKernel gmm;
    gmm(params);  // → operator()()
}
```

### 3.3 KernelGmmFinalizeRoutingPertokenDequant —— operator() 主入口

```cpp
// kernel_gmm_finalize_routing_pertoken_dequant.h:332-363
void operator()(const Params &params) {
    // === 阶段1: AIV 初始化输出 ===
    if ASCEND_IS_AIV {
        prologueOp_.Init(params.prologueParams);
        prologueOp_();  // → yGm全部归零 + 处理sharedInput
    }

    // === 阶段2: AIC 初始化 MatMul ===
    if ASCEND_IS_AIC {
        mmadOp_.Init(params.gmmParams.matmulTiling, GetTPipePtr());
        l0cOutUb_ = epilogueDequantOp_.GetL0c2UbTensor();
        // l0cOutUb_ 指向 Epilogue 的 VECIN 空间 [MAX_SINGLE_MNS int32]
        // → 这是 AIC MatMul 输出 和 AIV Epilogue 输入 的桥梁 (L0C→UB)
    }

    // === 阶段3: 全局初始化 ===
    InitParamsAndTensor(params);   // 读取 N,K, groupListGm_
    BlockSchedulerOp bs(baseM, baseN, baseK);
    SyncAll<false>();              // ★ AIC/AIV 同步: 确保 prologue 完成

    // === 阶段4: AIV 初始化 Epilogue ===
    if ASCEND_IS_AIV {
        epilogueDequantOp_.Init(params.epilogueParams);
        // → UB分配: logit/logitPong/x2Scale/x2ScalePong/.../outPing/outPong
    }

    // === 阶段5: 设置尾部对齐 ===
    if constexpr (formatB == CubeFormat::NZ) {
        if constexpr (transB) {
            bs.SetTailAlign(1, MATMUL_MNK_ALIGN);       // Zn: 标准16对齐
        } else {
            bs.SetTailAlign(1, MATMUL_MNK_ALIGN_INT8);  // NZ: int8特殊对齐
        }
    }

    // === 阶段6: 逐组处理 ===
    for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
        if (!UpdateGroupParams(params, groupIdx)) continue;  // M=0跳过
        ProcessSingleGroup(params, bs, groupIdx);
    }

    End();  // AIC等待最后的AIV完成
}
```

**operator() 控制流时序**:

```
Timeline:
        AIC                         AIV
        ───                         ───
                                    Prologue: Init yGm=0, sharedInput→yGm
        mmadOp_.Init()              │
        l0cOutUb_ = GetL0c2Ub()     │
        InitParamsAndTensor()       InitParamsAndTensor()
        BlockSchedulerOp(...)       BlockSchedulerOp(...)
        ──────────── SyncAll() ────────────  ★ 屏障
                                    epilogueDequantOp_.Init()
                                        → 分配UB: logit/scale/bias/out ping-pong

        ─── for each group ───      ─── for each group ───
        UpdateGroupParams()         UpdateGroupParams()
        ProcessSingleGroup()        ProcessSingleGroup()
          WaitForVector()             WaitForCube()
          MatMul(A,B,l0cOutUb)        Dequant+Logit+Route
          NotifyVector()              NotifyCube()
        ─── end for ───             ─── end for ───

        End(): WaitForVector()
```

### 3.4 ProcessSingleGroup —— 核心流水循环

```cpp
// kernel_gmm_finalize_routing_pertoken_dequant.h:285-330
void ProcessSingleGroup(params, bs, groupIdx) {
    int64_t m = Get<MNK_M>(problemShape_);
    int64_t n = Get<MNK_N>(problemShape_);
    int64_t k = Get<MNK_K>(problemShape_);

    // 更新 scheduler 和 epilogue 的 problem shape
    bs.UpdateNextProblem({m, n, k, 0});
    epilogueDequantOp_.UpdateNextProblem({m, n, k, 0});

    UpdateGlobalBuffer(params);  // 更新 AIC/AIV 的全局缓冲区指针

    CoordClass coord(m, n, k, baseM, baseN, baseK);
    BlockCoord tileIdx;  // 4维坐标 (mIdx, nIdx, mTailSplit, nTailSplit)

    while (bs.GetTileIdx(tileIdx)) {  // ★ Scheduler 迭代分配 tile
        BlockShape singleShape = bs.GetBlockShape(tileIdx);
        // singleShape: (singleM, singleN, mTailSplitId, nTailSplitId)

        // ★ 计算量化偏移 (PERTOKEN_MODE: per-token A量化 + per-channel B量化)
        blockOffset_ = coord.GetQuantOffset<PERTOKEN_MODE>(
            tileIdx.m, tileIdx.n,
            singleShape.mTail, singleShape.nTail);
        // blockOffset_ = (aOffset, bOffset, x1ScaleOffset, x2ScaleOffset, biasOffset, cOffset, logitOffset)

        // ============ AIC 侧 ============
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {       // 非第一个tile: 等AIV消费完前一个
                WaitForVector();          // ← mode=4, flags=6+22
            }

            // ★ Cube MatMul: int8 × int8 → int32 → l0cOutUb_
            mmadOp_(aGlobal_[aOffset],
                    bGlobal_[bOffset],
                    l0cOutUb_,            // ★ 直接写入UB桥接区
                    {singleM, singleN, k},
                    transA, transB);

            NotifyVector();               // → mode=4, flags=4+20
        }

        isVecSetSyncCom_ = true;          // 标记已完成至少1个tile

        // ============ AIV 侧 ============
        if ASCEND_IS_AIV {
            // 计算epilogue偏移
            int64_t y = Get<IDX_C_OFFSETS>(blockOffset_);   // 线性输出偏移
            int64_t mOffset = y / n;                          // 全局M行偏移
            int64_t nOffset = y - mOffset * n;                // 全局N列偏移

            BlockCoord epiOffset{
                nOffset,                // yGm N偏移
                biasOffset,             // bias偏移
                x2ScaleOffset,          // weight scale偏移
                x1ScaleOffset,          // per-token scale偏移
                mOffset,                // M行偏移
                mOffset};               // logit偏移

            WaitForCube();              // ← mode=4, flag=4

            // ★ Dequant + Logit + Route: 完整后处理
            epilogueDequantOp_(singleShape, epiOffset);

            NotifyCube();               // → mode=4, flag=6
        }
    }
}
```

**ProcessSingleGroup 流水时序**:

```
Tile 0               Tile 1               Tile 2              ...
════════             ════════             ════════

AIC: MatMul ──→     AIC: WaitForVector    AIC: WaitForVector
       │                  ← AIV Tile 0           ← AIV Tile 1
       │ NotifyVector(4,20)  │                     │
       │        │            │ MatMul              │ MatMul
       │        │            │ NotifyVector(4,20)  │ NotifyVector
       ▼        │            ▼                     ▼
AIV: WaitForCube│     AIV: WaitForCube      AIV: WaitForCube
       │        │            │                     │
       │ Epilogue       │ Epilogue            │ Epilogue
       │ NotifyCube(6)  │ NotifyCube(6)       │ NotifyCube(6)
       ▼        ▼       ▼        ▼            ▼        ▼
```

### 3.5 同步机制

A5的同步与A3完全不同——使用 mode=4 和带有 +16 偏移的双组 flags：

```cpp
// kernel_gmm_finalize_routing_pertoken_dequant.h:53-56
constexpr uint8_t  SYNC_AIC_AIV_MODES = 4U;   // A5 mode 4 (vs A3 mode 2)
constexpr uint16_t FLAG_ID_MAXS       = 16U;  // 双缓冲偏移量
constexpr uint16_t AIC_SYNC_AIV_FLAGS = 4U;   // AIC→AIV: flag 4 和 20
constexpr uint16_t AIV_SYNC_AIC_FLAGS = 6U;   // AIV→AIC: flag 6 和 22

// NotifyCube: AIV→AIC (AIC用WaitForVector等待)
void NotifyCube() {
    CrossCoreSetFlag<4, PIPE_V>(6);        // flag 6
}

// WaitForVector: AIC等待AIV
void WaitForVector() {
    CrossCoreWaitFlag<4, PIPE_FIX>(6);     // 等flag 6
    CrossCoreWaitFlag<4, PIPE_FIX>(6+16);  // 等flag 22
}

// NotifyVector: AIC→AIV (AIV用WaitForCube等待)
void NotifyVector() {
    CrossCoreSetFlag<4, PIPE_FIX>(4);      // flag 4
    CrossCoreSetFlag<4, PIPE_FIX>(4+16);   // flag 20
}

// WaitForCube: AIV等待AIC
void WaitForCube() {
    CrossCoreWaitFlag<4, PIPE_V>(4);       // flag 4
}
```

**A3 vs A5 同步对比**:

```
         A3                        A5
         ───                       ───
Mode:    2                         4
Flags:   3 (AIV→AIC)               6+22 (AIV→AIC)
         5 (AIC→AIV)               4+20 (AIC→AIV)

管道:    AIC→AIV: PIPE_FIX         AIC→AIV: PIPE_FIX ×2
         AIV→AIC: PIPE_MTE2        AIV→AIC: PIPE_V

双缓冲:  A3使用单一flag,         A5使用flag+16配对,
         依赖parallNum=4流水      提供更精确的slot级同步
```

### 3.6 UpdateOffset —— 跨组偏移计算

```cpp
// kernel_gmm_finalize_routing_pertoken_dequant.h:234-264
void UpdateOffset(uint32_t groupIdx) {
    if (groupIdx == 0) return;  // 第一组偏移全0

    uint64_t m = problemShape_.M;
    uint64_t n = problemShape_.N;
    uint64_t k = problemShape_.K;

    // A偏移: 累加 M×K (上一组的activation全部跳过)
    Get<IDX_A_OFFSETS>(baseOffset_) += m * k;

    // B偏移: NZ格式下的复杂计算
    if constexpr (formatB == CubeFormat::NZ) {
        if (transB) {
            // Zn: ceil(K/32) × ceil(N/16) × 512
            Get<IDX_B_OFFSETS>(baseOffset_) +=
                CeilDiv(k, 32) * CeilDiv(n, 16) * 512;
        } else {
            // NZ: ceil(N/32) × ceil(K/16) × 512
            Get<IDX_B_OFFSETS>(baseOffset_) +=
                CeilDiv(n, 32) * CeilDiv(k, 16) * 512;
        }
    }

    // x2Scale (weight scale): 每组独立, 跳过 N 个元素
    Get<IDX_X2SCALE_OFFSETS>(baseOffset_) = groupIdx * n;

    // x1Scale (per-token scale): 跨组累加 M
    Get<IDX_X1SCALE_OFFSETS>(baseOffset_) += m;

    // Logit偏移: 同x1Scale (跨组累加M)
    Get<IDX_LOGIT_OFFSETS>(baseOffset_) += m;

    // Bias偏移: 每组独立, 跳过 N
    Get<IDX_BIAS_OFFSETS>(baseOffset_) += n;
}
```

---

## 4. Epilogue 详细解析

### 4.1 UB 内存布局 (精确字节偏移)

从 `block_epilogue_dequant_finalize_routing.h:181-211` 的 `Init()` 方法推导:

```
UB 总布局 (VECIN + VECOUT):

VECIN 空间 (起始地址 = 0, AIV输入):
┌────────────────────────────────────────────────────────────────┐
│ Offset 0:          l0cOutUb_         MAX_SINGLE_MNS × 4       │
│                     int32 [128×256]   = 128KB                 │
│                     ★ AIC MatMul输出→AIV Epilogue输入的桥梁    │
├────────────────────────────────────────────────────────────────┤
│ Offset 128KB:      l0cOutUbFloat_    MAX_SINGLE_MNS × 4       │
│                     float [128×256]   = 128KB                 │
│                     ★ VFDoDequant结果暂存                      │
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB:      logitUbPing_      BLOCKS_BYTES = 256B      │
│                     float [64]        (logit双缓冲#1)         │
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB+256B: logitUbPong_      BLOCKS_BYTES = 256B      │
│                     float [64]        (logit双缓冲#2)         │
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB+512B: x2ScaleUbPing_    BLOCKS_BYTES = 256B      │
│                     float/bf16 [64]   (weight scale双缓冲#1)  │
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB+768B: x2ScaleUbPong_    BLOCKS_BYTES = 256B      │
│                     (weight scale双缓冲#2)                    │
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB+1KB:  x1ScaleUbPing_    BLOCKS_BYTES = 256B      │
│                     float [64]        (per-token scale双缓冲#1)│
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB+1280B:x1ScaleUbPong_    BLOCKS_BYTES = 256B      │
│                     (per-token scale双缓冲#2)                 │
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB+1536B:biasUbPing_       BLOCKS_BYTES = 256B      │
│                     bf16 [128]        (bias双缓冲#1)          │
├────────────────────────────────────────────────────────────────┤
│ Offset 256KB+1792B:biasUbPong_       BLOCKS_BYTES = 256B      │
│                     (bias双缓冲#2)                            │
└────────────────────────────────────────────────────────────────┘

VECOUT 空间 (AIV输出, 在VECIN之后):
┌────────────────────────────────────────────────────────────────┐
│ Offset 256KB+2KB: outUbPing_  HALF_DB_MAX_SINGLE_MNS × 4     │
│                    float [32×256]  = 32KB                     │
│                    ★ logit乘后暂存, 等待MTE3写回 (双缓冲#1)    │
├────────────────────────────────────────────────────────────────┤
│ Offset 288KB+2KB: outUbPong_  HALF_DB_MAX_SINGLE_MNS × 4     │
│                    float [32×256]  = 32KB                     │
│                    (双缓冲#2)                                  │
└────────────────────────────────────────────────────────────────┘

总计 UB 使用: ~322KB (256KB VECIN + 64KB VECOUT + align)
```

**双缓冲 Ping-Pong 机制**:

```
logitPingPongID_ 控制 scale/logit/bias 的 ping-pong:
  ID=0: 使用 Ping (MTE2加载到Ping, V处理Pong)
  ID=1: 使用 Pong (MTE2加载到Pong, V处理Ping)

yPingPongID_ 控制 out 的 ping-pong:
  ID=0: outUbPing_  (MTE3写回Pong, V处理→Ping)
  ID=1: outUbPong_  (MTE3写回Ping, V处理→Pong)

同步标志:
  SetFlag<MTE2_V>(pingPongID):  MTE2→V: scale/logit/bias数据已就绪
  WaitFlag<MTE2_V>(pingPongID): V等待MTE2完成
  SetFlag<V_MTE3>(yPingPongID):  V→MTE3: 数据已就绪, 可以写回GM
  WaitFlag<MTE3_V>(yPingPongID): MTE3等待上次写回完成
  SetFlag<MTE3_V>(yPingPongID): MTE3→V: 输出缓冲区已释放, 可覆盖
```

### 4.2 Epilogue operator() —— 完整后处理流水

```cpp
// block_epilogue_dequant_finalize_routing.h:504-558
void operator()(blockShape, blockCoord) {
    singleM_ = Get<MNK_M>(blockShape);    // 本次tile的M
    singleN_ = Get<MNK_N>(blockShape);    // 本次tile的N

    // AIV子核分工: M方向分两半
    uint32_t halfSingleM = CeilDiv(singleM_, GetTaskRation());  // = Ceil(singleM_/2)
    uint64_t singleMInVec = (subBlockIdx_ == 1)
        ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) return;

    uint64_t mOffset = subBlockIdx_ * halfSingleM;  // 子核的M起始偏移

    vlForFloatNumber_ = BLOCKS_BYTES / sizeof(float);   // = 256/4 = 64 (每次处理64个float)
    repeatTimesRe = CeilDiv(singleN_, vlForFloatNumber_); // N方向循环次数
    uint64_t logitOffset = Get<LOGIT_INDEXS>(blockCoord) + mOffset;
    uint64_t yOffset = Get<Y_IDXS>(blockCoord);

    // 选择当前 ping-pong ID 对应的缓冲区
    auto logitUb   = (logitPingPongID_ == 0) ? logitUbPing_   : logitUbPong_;
    auto x2ScaleUb = (logitPingPongID_ == 0) ? x2ScaleUbPing_ : x2ScaleUbPong_;
    auto x1ScaleUb = (logitPingPongID_ == 0) ? x1ScaleUbPing_ : x1ScaleUbPong_;
    auto biasUb    = (logitPingPongID_ == 0) ? biasUbPing_    : biasUbPong_;

    // === Step 1: MTE2 加载 (DMA) ===
    CopyInLogit(singleMInVec, logitOffset, logitUb);   // per-token logits
    CopyX2ScaleFromGm2Ub(x2ScaleUb, 0);                 // weight scale (per-channel)
    if (params_->x1ScaleGmAddr != nullptr)
        CopyX1ScaleFromGm2Ub(x1ScaleUb, singleMInVec * sizeof(float), mOffset);
    if (params_->biasGmAddr != nullptr)
        CopyBiasFromGm2Ub(biasUb);                      // bias

    // === Step 2: MTE2→V 同步 ===
    SetFlag<MTE2_V>(logitPingPongID_);     // 告诉Vector: 数据就绪
    WaitFlag<MTE2_V>(logitPingPongID_);    // 等待MTE2确认

    // === Step 3: VF Dequant ===
    VFDoDequantWithX1X2Scale(x2ScaleUb, x1ScaleUb, biasUb, singleMInVec);
    // 结果写入 l0cOutUbFloat_ [singleMInVec × alignN_] float

    // === Step 4: 翻转 ping-pong ===
    logitPingPongID_ = (logitPingPongID_ + 1) & 1;
    // 下一次MTE2可以加载下一个tile的scale/logit/bias

    // === Step 5: Logit + Route (每次处理最多32行) ===
    uint32_t loopNumY = CeilDiv(singleMInVec, MAX_OUTPUT_M_UBS); // = ceil(M/32)
    for (uint32_t i = 0; i < loopNumY; i++) {
        // 5a. 等待MTE3释放输出缓冲区
        WaitFlag<MTE3_V>(yPingPongID_);

        // 5b. VF Logit乘法: out = dequant[i*32..i*32+32] × logitUb[i*32..]
        repeatTimesLine_ = (i == loopNumY - 1)
            ? remainRepeatTimesLogit : MAX_OUTPUT_M_UBS;
        __ubuf__ float* outUbAddr = (yPingPongID_ == 0)
            ? (__ubuf__ float*)outUbPing_.GetPhyAddr()
            : (__ubuf__ float*)outUbPong_.GetPhyAddr();

        VFDoLogitMuls(
            i * MAX_OUTPUT_M_UBS * alignN_,     // dequant偏移
            i * MAX_OUTPUT_M_UBS,               // logit偏移
            repeatTimesLine_,                   // logit重复次数(=行数)
            repeatTimesRe,                      // N方向VEC迭代次数
            outUbAddr,
            logitUb);

        // 5c. V→MTE3: 数据就绪
        SetFlag<V_MTE3>(yPingPongID_);

        // 5d. 等待V确认
        WaitFlag<V_MTE3>(yPingPongID_);

        // 5e. ★ Atomic Route 写回
        auto yLocal = (yPingPongID_ == 0) ? outUbPing_ : outUbPong_;
        VectorAtomicProcess(
            singleN_,                                           // N大小
            repeatTimesLine_,                                   // 本次行数
            logitOffset + i * MAX_OUTPUT_M_UBS,                 // 全局M偏移
            yOffset,                                            // 全局N偏移
            yLocal);

        // 5f. MTE3→V: 输出缓冲区已释放
        SetFlag<MTE3_V>(yPingPongID_);

        // 5g. 翻转输出 ping-pong
        yPingPongID_ = (yPingPongID_ + 1) & 1;
    }

    // === Step 6: 清理 ===
    SetFlag<V_MTE2>(logitPingPongID_);      // 释放MTE2缓冲区
    WaitFlag<V_MTE2>(logitPingPongID_);     // 等待MTE2确认
}
```

**Epilogue 流水时序**:

```
Timeline for one tile:
──────────────────────────────────────────────────────────────────────
MTE2 (DMA Engine)        Vector Engine           MTE3 (DMA Engine)
───────────────────────  ─────────────────────   ────────────────────
Load logit/scale/bias    │                       │
→ logitUb/x2ScaleUb/     │                       │
  x1ScaleUb/biasUb       │                       │
                         │                       │
SetFlag<MTE2_V>(0) ───→ │                       │
                    WaitFlag<MTE2_V>(0)          │
                         │                       │
                         │ VFDoDequant:          │
                         │  int32→float Cast     │
                         │  × x2Scale (per-ch)   │
                         │  × x1Scale (per-tok)  │
                         │  + bias               │
                         │ → l0cOutUbFloat_      │
                         │                       │
                         │ ─── for each 32行 ─── │
                         │ VFDoLogitMuls         │
                         │ SetFlag<V_MTE3>(0)─→  │ AttomicAdd(yGm)
                         │                  WaitFlag<MTE3_V>(0)
                         │ ← SetFlag<MTE3_V>(0)  │
                         │ ...                   │ ... (32行循环)
                         │ ─── end for ───       │
                         │                       │
SetFlag<V_MTE2>(1) ──→  │                       │
                   WaitFlag<V_MTE2>(1)           │
──────────────────────────────────────────────────────────────────────
```

### 4.3 VFDoDequant —— 微架构反量化 (寄存器级)

这是A5与A3最大的差异之一——A5不使用 `AscendDequant` API，而是直接用 `AscendC::MicroAPI` 逐寄存器操作：

```cpp
// block_epilogue_dequant_finalize_routing.h:399-469
template <bool isBiasEpilogue>
void VFDoDequant(
    __ubuf__ float *dst,           // 输出: l0cOutUbFloat_
    __ubuf__ DataTypeIn *l0cOut,   // 输入: l0cOutUb_ (int32)
    __ubuf__ DataTypeX2Scale *x2Scale,     // weight scale
    __ubuf__ DataTypeX1Scale *x1Scale,     // per-token scale
    __ubuf__ BiasDtype *bias,              // bias
    uint16_t mSize, uint16_t nSize) {

    uint32_t eleNumPerVf = VECTOR_REG_WIDTH / sizeof(int32_t);  // = 512/32 = 8
    // 每次VF操作处理8个int32/float元素

    uint32_t nSrcUbAligned = Align(nSize, UB_ALIGN_SIZE / sizeof(int32_t));
    // UB中每行的对齐宽度: ceil(N, 32/4) = ceil(N, 8)

    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    // N方向VF循环: ceil(N/8)

    __VEC_SCOPE__ {  // ★ 进入Vector编程域

        // bf16→float双路解交织用的全量mask
        MaskReg maskN4B16 = CreateMask<bf16, ALL>();

        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;

            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                // ─── Step A: int32 → float32 Cast ───
                RegTensor<int32_t> l0cOutReg;
                RegTensor<float> l0cOutRegFloat;
                MaskReg maskN = UpdateMask<int32_t>(elementNum);

                // 从UB加载int32到VF寄存器
                DataCopy(l0cOutReg, l0cOut + mIdx*nSrcUbAligned + vfBlockIdx*8);
                // Cast: int32 → float32 (尾数截断模式)
                Cast<float, int32, ctInt322Fp32ES>(l0cOutRegFloat, l0cOutReg, maskN);
                // l0cOutRegFloat = (float)l0cOutReg  (逐元素)

                // ─── Step B: × x2Scale (per-channel weight scale) ───
                RegTensor<DataTypeX2Scale> scaleReg;
                RegTensor<float> castScaleReg, castScaleOneReg, mulScaleOutReg;

                DataCopy(scaleReg, x2Scale + vfBlockIdx*8);

                if constexpr (IsSameType<DataTypeX2Scale, bfloat16_t>::value) {
                    // bf16→float: 需要双路解交织
                    Cast<float, bf16, ctHalf2Fp32ZeroES>(castScaleReg, scaleReg, maskN);
                    Cast<float, bf16, ctHalf2Fp32OneES>(castScaleOneReg, scaleReg, maskN4B16);
                    Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                    // 解交织后: 8个bf16变成8个float32
                } else {
                    castScaleReg = scaleReg;  // float直接使用
                }

                Mul(mulScaleOutReg, l0cOutRegFloat, castScaleReg, maskN);
                // mulScaleOutReg = l0cOutFloat × weightScale (per-channel)

                // ─── Step C: × x1Scale (per-token activation scale) ───
                RegTensor<DataTypeX1Scale> perTokenScaleReg;
                RegTensor<float> mulPtScaleOutReg;

                // ★ DIST_BRC_B32: 广播加载 — 从x1Scale[mIdx]加载1个float, 广播到整个寄存器
                DataCopy<float, DIST_BRC_B32>(perTokenScaleReg, x1Scale + mIdx);
                // perTokenScaleReg = {scale, scale, scale, ..., scale} (8个相同值)

                Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
                // mulPtScaleOutReg = mulScaleOutReg × perTokenScale[i] (per-token广播)

                // ─── Step D: + bias (可选) ───
                RegTensor<BiasDtype> biasReg;
                RegTensor<float> addBiasOutReg, castBiasReg, castBiasOneReg;

                if constexpr (isBiasEpilogue) {
                    DataCopy(biasReg, bias + vfBlockIdx*8);
                    if constexpr (IsSameType<BiasDtype, bfloat16_t>::value) {
                        Cast<float, bf16, ctHalf2Fp32ZeroES>(castBiasReg, biasReg, maskN);
                        Cast<float, bf16, ctHalf2Fp32OneES>(castBiasOneReg, biasReg, maskN4B16);
                        Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                    } else {
                        castBiasReg = biasReg;
                    }
                    Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN);
                    // addBiasOutReg = mulPtScaleOutReg + bias (可选)
                } else {
                    addBiasOutReg = mulPtScaleOutReg;
                }

                // ─── Step E: 写回UB ───
                DataCopy<float, DIST_NORM_B32>(
                    dst + mIdx*nDstUbAligned + vfBlockIdx*8,
                    addBiasOutReg, maskN);
                // 写回 l0cOutUbFloat_[mIdx][vfBlockIdx*8..vfBlockIdx*8+7]

                elementNum -= eleNumPerVf;  // 递减剩余元素数(用于尾部mask)
            }
        }
    }  // __VEC_SCOPE__结束 — VF寄存器释放
}
```

**VFDoDequant 数据流图**:

```
对于每个元素位置 (mIdx, vfBlockIdx*8 .. vfBlockIdx*8+7):

l0cOutUb_[mIdx × nSrcUbAligned + vfBlock*8]  (UB, int32)
         │
         │ DataCopy (UB→VF Reg)
         ▼
    l0cOutReg  (VF Reg, int32)
         │
         │ Cast<int32→float32>
         ▼
    l0cOutRegFloat  (VF Reg, float)
         │
         │ × scaleReg  (per-channel: 从x2ScaleUb加载, bf16需先Cast+Interleave)
         ▼
    mulScaleOutReg  (VF Reg, float)
         │
         │ × perTokenScaleReg  (per-token: DIST_BRC_B32 广播加载)
         ▼
    mulPtScaleOutReg  (VF Reg, float)
         │
         │ + biasReg  (可选: 从biasUb加载, bf16需先Cast+Interleave)
         ▼
    addBiasOutReg  (VF Reg, float)
         │
         │ DataCopy<DIST_NORM_B32> (VF Reg→UB)
         ▼
l0cOutUbFloat_[mIdx × nDstUbAligned + vfBlock*8]  (UB, float)
```

### 4.4 VFDoLogitMuls —— Logit 广播乘法

```cpp
// block_epilogue_dequant_finalize_routing.h:277-302
void VFDoLogitMuls(offsetRe, offsetLogit, repeatTimesLogit, repeatTimesRe,
                   outUbAddr, logitUb) {
    __ubuf__ float *l0cOutUbAddr = (__ubuf__ float*)l0cOutUbFloat_.GetPhyAddr();
    __ubuf__ float *logitUbAddr = (__ubuf__ float*)logitUb.GetPhyAddr();

    l0cOutUbAddr += offsetRe;       // dequant结果的偏移
    logitUbAddr += offsetLogit;     // logit的偏移

    __VEC_SCOPE__ {
        RegTensor<float> vregLogit, vregRe, vDstReg;
        MaskReg mask;

        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {  // 每行
            // ★ BRC_B32 广播: 加载 logit[i] 并广播到整个寄存器
            DataCopy<float, DIST_BRC_B32>(vregLogit, logitUbAddr + i);
            // vregLogit = {logit_i, logit_i, ..., logit_i} (8个相同值)

            for (uint16_t j = 0; j < repeatTimesRe; ++j) {  // N方向VEC迭代
                mask = UpdateMask<float>(singleN_);

                // 加载 dequant 结果
                DataCopy(vregRe, l0cOutUbAddr + i*alignN_ + j*vlForFloatNumber_);

                // Mul: dequant[i][j*64..j*64+63] × logit[i]
                Mul(vDstReg, vregLogit, vregRe, mask);

                // 写回 outUb
                DataCopy(outUbAddr + i*alignN_ + j*vlForFloatNumber_, vDstReg, mask);
            }
        }
    }
}
```

### 4.5 VectorAtomicProcess —— 路由写回

```cpp
// block_epilogue_dequant_finalize_routing.h:262-274
void VectorAtomicProcess(curBaseN, curVecBaseM, offsetM, yOffset, yLocal) {
    SetAtomicAdd<float>();  // ← MTE3开启原子加模式

    DataCopyExtParams paramsOut{
        1,                                  // blockCount: 每次1行
        curBaseN * sizeof(float),            // blockLen: N×4 bytes
        0, 0, 0};

    for (uint32_t i = 0; i < curVecBaseM; i++) {
        // ★ 标量读取: 从GM路由表读取单个输出行号
        auto outRow = static_cast<uint64_t>(
            rowIndexGlobal_.GetValue(offsetM + i));
        // rowIndexGlobal_: [M] int64/int32, tokenRanks映射表

        // ★ 写回: AtomicAdd到目标行
        DataCopyPad(
            yGlobal_[outRow * n_ + yOffset],    // 目标地址 = 输出行×N + N偏移
            yLocal[i * alignN_],                 // 源 = UB中第i行
            paramsOut);                          // 拷贝curBaseN个float
    }
    SetAtomicNone();  // 关闭原子加模式

    // W8A8 A5: 没有确定性分支 — 直接写回yGm, 无中间暂存
}
```

---

## 5. 关键数据结构

### 5.1 GMMFinalizeRoutingTilingData

```cpp
// arch35/grouped_matmul_finalize_routing_tiling_data.h:26-47
#pragma pack(push, 8)
struct GMMFinalizeRoutingDataParams {
    uint32_t groupNum;           // MoE group数
    uint32_t batch;              // 输出batch
    uint32_t sharedInputOffset;  // sharedInput起始行偏移
    uint32_t sharedInputLen;     // sharedInput长度(行数)
    float residualScale;         // sharedInput缩放因子
    uint32_t aQuantMode;         // 激活量化模式
    uint32_t bQuantMode;         // 权重量化模式
    uint32_t biasDtype;          // bias类型
    uint8_t groupListType;       // 0=差分, 1=绝对值
    uint8_t hasBias;             // bias存在标记
    uint16_t reserved1;
    uint32_t reserved2;
};  // 共40字节

struct GMMFinalizeRoutingTilingData {
    GMMFinalizeRoutingDataParams gmmFinalizeRoutingDataParams;  // 40B
    TCubeTiling matmulTiling;  // Cube matmul tiling参数
};
```

### 5.2 CoordClass::GetQuantOffset —— 量化偏移计算

A5使用 `Coordinate<transA, transB, formatA, formatB, formatC>` 模板类计算各输入/输出的偏移：

```cpp
blockOffset_ = coord.GetQuantOffset<PERTOKEN_MODE>(
    mIdx, nIdx, mTail, nTail);
// 返回 Shape<int64_t, 7>:
//   [0] = aOffset       (activation偏移, 元素)
//   [1] = bOffset       (weight偏移, 元素)
//   [2] = x1ScaleOffset (per-token scale偏移, 元素)
//   [3] = x2ScaleOffset (weight scale偏移, 元素)
//   [4] = biasOffset    (bias偏移, 元素)
//   [5] = cOffset       (输出偏移, 线性索引)
//   [6] = logitOffset   (logit偏移, 元素)
```

PERTOKEN_MODE 的偏移计算逻辑:
- `x1ScaleOffset`: 跨组累加 M 行 (global per-token offset)
- `x2ScaleOffset`: 每组独立，偏移 = groupIdx × N + nIdx × baseN
- `cOffset`: 线性输出偏移 = mGlobalOffset × N + nIdx × baseN

---

## 6. 不同 Shape 场景的数值模拟

### 6.1 场景一: 通用小Shape

```
输入: groupNum=4, m=256 (每group 64), n=1024, k=4096
      scale=float, rowIndex=int64, coreNum=4

CalBasicBlock() 估算:
  baseM ≈ 128, baseN ≈ 256, baseK ≈ 128

ProcessSingleGroup (group 0, m=64):
  scheduler 产生 tiles: blockDimM=1, blockDimN=4
  tile (0,0): singleM=64, singleN=256

  AIC: MatMul 64×256×4096 int8→int32→l0cOutUb_[64×256]
  AIV (subBlockIdx=0, halfSingleM=32):
    singleMInVec=32, mOffset=0
    logitOffset = mOffset_base + 0
    yOffset = 0*1024 + 0 = 0

    CopyInLogit: 32 floats (singleMInVec × sizeof(float) = 128B)
    CopyX2ScaleFromGm2Ub: 256 floats (1KB)
    CopyX1ScaleFromGm2Ub: 32 floats (128B)

    VFDoDequant: mSize=32, nSize=256
      nLoopCnt = ceil(256/8) = 32 (N方向32次VEC迭代)
      mSize=32 → 总共 32×32 = 1024 次VEC操作

    Logit+Route (loopNumY = ceil(32/32) = 1):
      VFDoLogitMuls: repeatTimesLine_=32, repeatTimesRe=4
      VectorAtomicProcess: 32次atomicAdd, 每次256个float

  tile (0,1): singleN=256, yOffset=256
  tile (0,2): singleN=256, yOffset=512
  tile (0,3): singleN=256, yOffset=768
```

### 6.2 场景二: MoE 大Shape

```
输入: groupNum=8, m=1536 (avg=192), n=7168, k=2048
      scale=float, rowIndex=int64

CalBasicBlock() 估算:
  n=7168, k=2048 → baseK可能因k较小而调整
  avg_m=192 → baseM可能增大 (类似A3的交换策略)

Kernel行为:
  group 0 (m=192): blockDimM=ceil(192,baseM), blockDimN=ceil(7168,baseN)
  blockOffset_ 通过 GetQuantOffset<PERTOKEN_MODE> 计算

  偏移计算示例 (tile (0,0)):
    aOffset = 0 (第一组)
    bOffset = 0*7168*2048 + 0*baseN*2048 = 0
    x2ScaleOffset = 0*7168 + 0 = 0
    x1ScaleOffset = 跨组累积 = 0 (第一组)
    cOffset = 0*7168 + 0*baseN = 0
    logitOffset = 0

  VectorAtomicProcess:
    outRow = rowIndexGlobal_.GetValue(offsetM + i)
    AtomicAdd(yGlobal_[outRow * 7168 + yOffset], yLocal[i * alignN], baseN floats)
```

### 6.3 场景三: 大Shape + 多Core

ASWT Scheduler 自动处理:
- M_TAIL_SPLIT_TILEIDXS: M方向尾部不足baseM时自动分割
- N_TAIL_SPLIT_TILEIDXS: N方向尾部不足baseN时自动分割
- 多核负载均衡: 每个core通过scheduler的GetTileIdx轮询获取tile

```
输入: groupNum=1, m=2048, n=4096, k=8192, coreNum=4

CalBasicBlock() 可能产出:
  baseM=128, baseN=256, baseK=128

blockDimM = ceil(2048/128) = 16
blockDimN = ceil(4096/256) = 16
总共: 256个tile, 每core处理64个

ASWT Scheduler 每个core独立调用 GetTileIdx():
  根据 multiBlock策略 + subBlockIdx 分配tile
  自动处理 M=16×128 (恰好整除, 无尾部)
  N=16×256 (恰好整除, 无尾部)

每个tile:
  AIC: MatMul 128×256×8192 → K迭代ceil(8192/128)=64次
  AIV: Dequant 128×256 → 每次32行 → 4次logit循环
```

### 6.4 三种场景对比

```
                    场景一            场景二              场景三
                    (通用小Shape)     (MoE大Shape)        (大Shape多Core)
─────────────────────────────────────────────────────────────────────
n, k                1024, 4096        7168, 2048          4096, 8192
m, groupNum         256, 4           1536, 8             2048, 1
avg_m               64               192                 2048
coreNum             4                4                   4
─────────────────────────────────────────────────────────────────────
baseM (估算)        128              128~256             128
baseN (估算)        256              256                 256
K迭代次数           32               16                  64
N tile数/group      4                28                  16
─────────────────────────────────────────────────────────────────────
AIV M循环/tile      2(32行)          3~6(32行)           4(32行)
VFDoDequant ops     2×32×32=2048     6×28×32=5376       4×16×64=4096
logit循环           2                6                   4
atomicAdd次数/tile  64               192                 128
dbL0A/L0B           双缓冲开启       双缓冲开启          双缓冲开启
同步mode            4                4                   4
```

---

## 7. A3 vs A5 详细对比

```
┌────────────────────┬─────────────────────────┬──────────────────────────┐
│       维度          │        A3 (910B/C)      │        A5 (950)          │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ Tiling 模板        │ BaseTiling (模板#0)     │ QuantTiling (模板#1)     │
│ Tiling 基类        │ 自定义 BaseTiling       │ GroupedQmmTiling         │
│ TilingKey          │ 字符偏移 100...01       │ APT模板 (ATRANS,BTRANS..) │
│ TilingData 结构    │ 宏展开20字段            │ POD struct 嵌套          │
│ baseM/N/K          │ 固定128/256/128+特殊    │ CalBasicBlock() 动态     │
│ special shape分支  │ n=7168/7680 k=2048      │ 继承自基类算法           │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ Kernel 架构        │ 手写类 QuantGroupMatmul │ CGMCT 组件化             │
│ Scheduler          │ MNBlockIdxCompute       │ ASWT Scheduler           │
│ MatMul             │ MatmulImpl 直接调用     │ BlockMmadBuilder 封装    │
│ Prologue           │ PreProcess() 成员函数   │ BlockPrologue 独立组件   │
│ Epilogue           │ 成员函数串联            │ BlockEpilogue 独立组件   │
│ 代码复用           │ 低 (全部在cpp中)        │ 高 (CGMCT组件库共享)     │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ Dequant 实现       │ AscendDequant API       │ MicroAPI VF寄存器编程    │
│   int32→float      │ Cast内部实现            │ Cast<int32,float,ES>     │
│   ×weightScale     │ AscendDequant内部       │ DataCopy+Mul+Interleave  │
│   ×perTokenScale   │ BroadCast+Mul           │ DIST_BRC_B32+Mul         │
│   +bias            │ BroadCast+Add           │ DataCopy+Cast+Add        │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ UB 布局            │ TBuf偏移 ~96KB          │ VECIN/VECOUT ~322KB      │
│ UB 分配方式        │ 手动GetWithOffset       │ 手动绝对地址分配         │
│ TQue 双缓冲        │ 6个队列 (scale/bias/    │ 无TQue — 手动ping-pong   │
│                    │  pertoken/vecIn/vecOut/ │  通过HardEvent标志同步   │
│                    │  queBind)               │                          │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ L0 双缓冲          │ dbL0C=1 (单缓冲)        │ dbL0A=2, dbL0B=2 (双)    │
│   A矩阵            │ depthA1=8 (L1双缓冲)    │ depthA1=base (动态)      │
│   B矩阵            │ depthB1=8               │ depthB1=base (动态)      │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ 同步 mode          │ 2                       │ 4                        │
│ AIC→AIV flags      │ 5 (单flag)              │ 4+20 (双flag)            │
│ AIV→AIC flags      │ 3 (单flag)              │ 6+22 (双flag)            │
│ 同步管道           │ PIPE_FIX / PIPE_MTE2    │ PIPE_FIX / PIPE_V        │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ 确定性             │ 支持                    │ 不支持                   │
│ 确定性workspace    │ 64MB或96MB              │ N/A                      │
│ 确定性UB           │ queBind 12KB×2          │ N/A                      │
│ 确定性写回         │ mmQuantOutGm→FRDeter    │ N/A                      │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ GroupList处理      │ groupTokensGm.GetValue  │ GetSplitValueFromGL      │
│ groupListType=0    │ m -= offsetM (累减)      │ offset-preOffset (差分)  │
│ groupListType=1    │ m = 读值 (直接)          │ m = 读值 (直接)          │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ AIV 子核分工       │ GetOffset (除以2)       │ CeilDiv(M, GetTaskRatio) │
│ 子核处理方式       │ AIV0→上半部, AIV1→下半 │ 同A3: subBlockIdx 0/1    │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ 算子注册           │ binary.json             │ APT 框架 (opFile=apt)    │
│ 编译宏             │ __CCE_AICORE__ == 220   │ __CCE_AICORE__ == 310    │
│ 架构宏             │ __NPU_ARCH__ 未定义     │ __NPU_ARCH__ == 3510     │
└────────────────────┴─────────────────────────┴──────────────────────────┘
```

### 关键架构升级解读

1. **CGMCT 组件化**: A5将所有流水线组件拆分为独立类 — `BlockMmadBuilder` (MatMul)、`BlockEpilogueDequantFinalizeRouting` (反量化+路由)、`BlockPrologueFinalizeRouting` (输出初始化)。组件间通过 UB 的 `l0cOutUb_` 桥接，实现松耦合。

2. **L0 双缓冲**: A5 开启 `dbL0A=2, dbL0B=2`，AIC 可以在 AIV 处理当前 tile 的同时预加载下一个 tile 的 A/B 数据到 L0A/L0B，从而隐藏更多的内存延迟。

3. **VF微架构编程**: A5 的 Dequant 不使用 `AscendDequant` API，而是直接使用 `AscendC::MicroAPI` 的 `RegTensor`、`Cast`、`Mul`、`Add`、`DataCopy` 等微架构指令。这允许：
   - `DIST_BRC_B32`: per-token scale的标量广播加载 (一条指令完成)
   - `Interleave`: bf16→float的高效解交织
   - 更精细的寄存器分配和指令调度

4. **ASWT Scheduler**: `GroupedMatmulAswtWithTailSplitScheduler` 自动处理 M/N 方向的尾部不规则分块，不需要像 A3 那样手动 hardcode 对角策略。

5. **无确定性分支**: A5 W8A8 没有任何确定性相关代码。`SetAtomicAdd` 直接写回 `yGm`，不需要中间暂存和二次分发。这意味着 A5 上 W8A8 的确定性是"天生非确定"的 — 如果需要确定性，必须在更高层次（如数据预处理）保证。

---

## 附录A: 关键常量速查

| 常量 | 值 | 含义 |
|------|----|------|
| `MAX_SINGLE_MNS` | 128 × 256 = 32768 | 最大单 tile M×N 元素数 |
| `HALF_DB_MAX_SINGLE_MNS` | 32 × 256 = 8192 | 输出 ping-pong 半边大小 |
| `BLOCKS_BYTES` | 256 | 标量缓冲区大小 (logits/scale/bias) |
| `BLOCK_ELEMENTS_FP32S` | 8 | float32 的 VF 宽度 |
| `MAX_OUTPUT_M_UBS` | 32 | 每次 logit+route 处理的最大 M 行 |
| `SYNC_AIC_AIV_MODES` | 4 | 跨核同步 mode |
| `AIC_SYNC_AIV_FLAGS` | 4 (+20) | AIC→AIV 事件 |
| `AIV_SYNC_AIC_FLAGS` | 6 (+22) | AIV→AIC 事件 |
| `FLAG_ID_MAXS` | 16 | 双重缓冲的偏移量 |
| `WEIGHT_TILE_K_SMALL` | 16 | NZ格式K方向tile大小 |
| `WEIGHT_TILE_N_LARGE` | 32 | NZ格式N方向tile大小 |
| `WEIGHT_TILE_CAPACITY` | 512 | NZ格式tile容量 |

## 附录B: 涉及的核心文件清单

| 文件 | 行数 | 用途 |
|------|------|------|
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 183 | A5 内核入口 + APT 模板分派 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 92 | CGMCT 组件组装 + 调度包装 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 49 | TilingData POD 结构体 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h` | 35 | APT TilingKey 声明 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 566 | QuantTiling 完整实现 |
| `op_host/grouped_matmul_finalize_routing_tiling.cpp` | 119 | 架构路由 |
| `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` | 368 | CGMCT Kernel 核心类 |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | 563 | VFDequant + Logit + Route |
