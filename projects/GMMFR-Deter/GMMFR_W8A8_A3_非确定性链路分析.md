# GMMFR W8A8 int8 A3 非确定性路径 —— 从 Tiling 到 Kernel 完整链路解析

> **芯片**: Ascend 910B/910C (A3, arch22, `__CCE_AICORE__ == 220`)
> **数据类型**: activation INT8 ND × weight INT8 NZ → output FLOAT32
> **确定性**: 关闭 (`deterministicFlag = 0`)

---

## 1. 文件结构概览

GMMFR (GroupedMatMul Finalize Routing) 算子代码分布在以下层次:

```
gmm/grouped_matmul_finalize_routing/
├── op_host/                                    # 主机端: 算子定义 + Tiling + 形状推断
│   ├── grouped_matmul_finalize_routing_def.cpp        # OP 注册 (ascend910b / ascend910_93)
│   ├── grouped_matmul_finalize_routing_infershape.cpp # 形状/数据类型校验
│   ├── grouped_matmul_finalize_routing_tiling.cpp     # 架构调度 (A3→模板0, A5→模板1)
│   ├── grouped_matmul_finalize_routing_tiling.h       # TilingData / CompileInfo 结构体
│   ├── grouped_matmul_finalize_routing_base_tiling.cpp # ★ W8A8 Tiling 核心逻辑 (549行)
│   ├── grouped_matmul_finalize_routing_base_tiling.h   # ★ Base tiling 基类声明
│   └── config/
│       ├── ascend910b/grouped_matmul_finalize_routing_binary.json  # A3 二进制配置
│       └── ascend910_93/grouped_matmul_finalize_routing_binary.json
│
├── op_kernel/                                  # 设备端: AICORE Kernel
│   ├── grouped_matmul_finalize_routing.cpp            # ★ A3 内核入口 + TilingKey 分派 (171行)
│   ├── grouped_matmul_finalize_routing.h              # ★ QuantGroupMatmul<P> 完整实现 (688行)
│   ├── grouped_matmul_finalize_routing_utils.h        # ★ MNBlockIdxCompute + 数据结构 (204行)
│   └── arch35/                                        # A5 路径 (不涉及)
│
├── op_api/                                     # aclnn OP-API 包装层
└── op_graph/                                   # 图模式降级
```

**架构路由** (`op_host/grouped_matmul_finalize_routing_tiling.cpp:48`):
```cpp
if (compileInfoPtr->npuArch == NpuArch::DAV_3510) {
    // A5 → QuantTiling 模板#1
} else {
    // A2/A3 → BaseTiling 模板#0
}
```
A3 内核编译条件: `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220`

---

## 2. Tiling 层完整解析

### 2.1 总体调用流程

```
DoOpTiling()                                         [base_tiling.cpp:491]
  │
  ├─ 1. ParseInputAndAttr()                          [base_tiling.cpp:177]
  │     ├─ GetInputShape() → 解析 x[M×K] ND, w[E×N×K] NZ → 得出 m_, n_, k_
  │     ├─ ParseAttr() → batch_, residualScale_, tuningConfig_
  │     ├─ 获取硬件参数: ubSize, l1Size, l0CSize (通过 PlatformAscendC)
  │     ├─ blockDim_ = GetCoreNumAic()               ← AIC 核心数
  │     └─ groupNum_ = weight.shape[0]                ← E (MoE专家数)
  │
  ├─ 2. W8A8TilingProcess()                          [base_tiling.cpp:350]
  │     ├─ 选择 baseM/baseN/baseK (固定 128/256/128 或特殊shape交换)
  │     ├─ 计算 vBaseM = UBCALSIZE / baseN
  │     ├─ 设置 MatMul 类型: A=int8 ND, B=int8 NZ, C=int32
  │     ├─ mm_.SetFixSplit(baseM, baseN, baseK)       ← 固定分块策略
  │     ├─ mm_.GetTiling(tilingData_.matmulTiling)    ← Cube tiling 参数计算
  │     ├─ 编码 TilingKey: 基础 100...01 + rowIndex偏移 + scale偏移
  │     └─ workspaceSize_ = userWS + systemWS
  │
  ├─ 3. DeterministicTilingProcess()                 [base_tiling.cpp:413]
  │     └─ 当 deterministic=0: → 直接 return (不做任何分配)
  │
  ├─ 4. OtherSettingTilingProcess()                  [base_tiling.cpp:401]
  │     ├─ sharedInput 为空 → tilingKey_ += 10000
  │     └─ groupListType=0 → tilingKey_ += 1000
  │
  ├─ 5. FillTilingData()                             [base_tiling.cpp:427]
  │     └─ 将所有参数写入 GroupMatmulFRTilingData 结构体
  │
  └─ 6. PrintTilingData()                            [base_tiling.cpp:469]
```

### 2.2 ParseInputAndAttr —— 输入解析

```cpp
// base_tiling.cpp:177-217
ge::graphStatus ParseInputAndAttr() {
    // 1. 解析输入形状
    GetInputShape();   // m_ = x.shape[0], k_ = x.shape[1], n_ = w.shape[2]
                       // x: ND格式, shape=[M, K]
                       // w: NZ格式, shape=[E, N/16, K/16, 16, 16] → n_ = origin[2]

    // 2. 获取硬件信息
    auto ascendcPlatform = PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(UB, ubSize);
    ascendcPlatform.GetCoreMemSize(L1, l1Size);
    ascendcPlatform.GetCoreMemSize(L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);  // Cube MatMul 需要知道缓冲区大小
    blockDim_ = ascendcPlatform.GetCoreNumAic();

    // 3. 解析属性 + tuningConfig
    ParseAttr();
    // batch_ = attrs[5]
    // residualScale_ = attrs[1]
    // deterministic=0 时读取 tuningConfig (用户微调参数)
    //   tuningConfig_ > 0 → 人工指定 avg_m
    //   tuningConfig_ = 0 → 自动计算 avg_m = m_ / groupNum_

    // 4. 获取 group 数量
    groupNum_ = context_->GetInputShape(1)->GetStorageShape()[0];  // weight[E,...]
}
```

### 2.3 W8A8TilingProcess —— 核心 Tiling 参数计算

```cpp
// base_tiling.cpp:350-399
ge::graphStatus W8A8TilingProcess() {
    // ====== Step 1: workspace 计算 ======
    // MatMul workspace: parallNum(4) × baseM(128) × baseN(256) × sizeof(int32_t) × blockDim_
    size_t userWorkspaceSize = CV_PARALL_NUM * 256 * 128 * sizeof(int32_t) * blockDim_;
    //                         = 4 * 256 * 128 * 4 * blockDim_ = 512KB × coreNum
    size_t systemWorkspaceSize = RPC_WORKSIZE * MB_SIZE;  // 20MB (RPC固定开销)
    ubRestBytes_ = A8W8_UBRESTBYTES;  // 101376 (TBuf可用空间)

    // ====== Step 2: baseM/baseN 选择 ======
    uint32_t baseM = BEST_BASE_M;  // 128
    uint32_t baseN = BEST_BASE_N;  // 256

    // ★ 特殊 Shape 分支
    if ((n_ == 7168 || n_ == 7680) && k_ == 2048) {
        uint32_t avg_m = (tuningConfig_ != 0) ? tuningConfig_
                        : ((groupNum_ != 0) ? (m_ / groupNum_) : 1);
        // avg_m ∈ (128, 256] → 交换 baseM ⇄ baseN
        baseM = (avg_m > 128 && avg_m <= 256) ? BEST_BASE_N : BEST_BASE_M;  // 256 : 128
        baseN = (avg_m > 128 && avg_m <= 256) ? BEST_BASE_M : BEST_BASE_N;  // 128 : 256
    }

    // ====== Step 3: vBaseM 计算 ======
    vBaseM_ = UBCALSIZE / baseN;  // 4096 / baseN
    // baseN=256 → vBaseM=16,  baseN=128 → vBaseM=32

    // ====== Step 4: MatMul Tiling ======
    mm_.SetAType(GM, ND, DT_INT8, false);   // A: GM存储, ND格式, int8
    mm_.SetBType(GM, NZ, DT_INT8, false);   // B: GM存储, NZ分形, int8
    mm_.SetCType(GM, ND, DT_INT32);         // C: int32累加结果
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(baseM, n_, k_);            // 逻辑形状
    mm_.SetOrgShape(baseM, n_, k_);
    mm_.SetFixSplit(baseM, baseN, BEST_BASE_K);  // FixSplit(baseM, baseN, 128)
    mm_.GetTiling(tilingData_.matmulTiling);     // 实际Cube tiling参数

    // ====== Step 5: TilingKey 编码 ======
    tilingKey_ = 10000000000000000001UL;  // 基线 key
    if (rowIndexDtype == ge::DT_INT32)  tilingKey_ += ROW_INDEX_FACTOR;   // +10
    if (scaleDtype == ge::DT_BF16)      tilingKey_ += SCALE_FACTOR;       // +100

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;
    // 非确定性: ~20.5MB (2核×512KB + 20MB), ~21MB (4核×1MB + 20MB)
}
```

### 2.4 baseM/baseN 选择决策树

```
W8A8TilingProcess: baseM/baseN 选择
│
├─ n_ == 7168 或 7680 且 k_ == 2048 ?
│   │
│   ├─ YES → 计算 avg_m
│   │   │
│   │   ├─ tuningConfig_ > 0 → avg_m = tuningConfig_
│   │   └─ tuningConfig_ = 0 → avg_m = m_ / groupNum_ (至少为1)
│   │   │
│   │   ├─ 128 < avg_m ≤ 256 → baseM=256, baseN=128 ★交换
│   │   └─ avg_m ≤ 128 或 avg_m > 256 → baseM=128, baseN=256 (不变)
│   │
│   └─ NO → baseM=128, baseN=256 (默认)
│
└─ vBaseM = UBCALSIZE / baseN
   ├─ baseN=256 → vBaseM=16  (默认)
   └─ baseN=128 → vBaseM=32  (交换后，每次AIV处理行数翻倍)
```

### 2.5 FillTilingData —— 关键参数汇总

```cpp
// base_tiling.cpp:427-456
void FillTilingData() {
    // === Cube MatMul 参数 ===
    tilingData_.matmulTiling.set_dbL0C(1);    // L0C 单缓冲
    tilingData_.matmulTiling.set_stepKa(4);   // L1左矩阵每次搬运4×baseK
    tilingData_.matmulTiling.set_stepKb(4);   // L1右矩阵每次搬运4×baseK
    tilingData_.matmulTiling.set_depthA1(8);  // L1左矩阵双缓冲深度=2×stepKa
    tilingData_.matmulTiling.set_depthB1(8);  // L1右矩阵双缓冲深度=2×stepKb
    tilingData_.matmulTiling.set_stepM(1);    // M方向步长(每tile处理1份)
    tilingData_.matmulTiling.set_stepN(1);    // N方向步长

    // === 算子Tiling参数 ===
    tilingData_.set_coreNum(blockDim_);       // AIC核心数(2/4/8)
    tilingData_.set_groupNum(groupNum_);      // MoE专家数
    tilingData_.set_totalInGroup(m_);         // 总M行数(跨所有group)
    tilingData_.set_batch(batch_);            // 输出batch行数
    tilingData_.set_k(k_);                    // K维度长度
    tilingData_.set_n(n_);                    // N维度长度
    tilingData_.set_vBaseM(vBaseM_);          // 16或32
    tilingData_.set_ubCalSize(4096);          // 向量计算单元=16×256元素
    tilingData_.set_parallNum(4);             // Cube slot并行数
    tilingData_.set_ubRestBytes(101376);      // TBuf大小(约99KB)
    tilingData_.set_deterministicFlag(0);     // ★非确定性
    tilingData_.set_deterWorkspaceSize(0);    // ★无确定性workspace
}
```

### 2.6 TilingData 结构体 (内存布局)

```cpp
// grouped_matmul_finalize_routing_tiling.h:40-61  (A3 版本)
BEGIN_TILING_DATA_DEF(GroupMatmulFRTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);  // Cube tiling参数(可变大小)
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);          // offset 0
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);         // offset 4
  TILING_DATA_FIELD_DEF(uint32_t, totalInGroup);     // offset 8
  TILING_DATA_FIELD_DEF(uint32_t, batch);            // offset 12
  TILING_DATA_FIELD_DEF(uint32_t, k);                // offset 16
  TILING_DATA_FIELD_DEF(uint32_t, n);                // offset 20
  TILING_DATA_FIELD_DEF(uint32_t, vBaseM);           // offset 24
  TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);        // offset 28
  TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);      // offset 32
  TILING_DATA_FIELD_DEF(uint32_t, parallNum);        // offset 36
  TILING_DATA_FIELD_DEF(uint32_t, sharedInputOffset);// offset 40
  TILING_DATA_FIELD_DEF(uint32_t, sharedInputLen);   // offset 44
  TILING_DATA_FIELD_DEF(float, residualScale);       // offset 48
  TILING_DATA_FIELD_DEF(uint32_t, quantGroupNum);    // offset 52
  TILING_DATA_FIELD_DEF(uint32_t, withOffset);       // offset 56
  TILING_DATA_FIELD_DEF(uint32_t, hasPertokenScale); // offset 60
  TILING_DATA_FIELD_DEF(uint32_t, hasBias);          // offset 64
  TILING_DATA_FIELD_DEF(uint32_t, deterministicFlag);// offset 68
  TILING_DATA_FIELD_DEF(uint32_t, deterWorkspaceSize);// offset 72
END_TILING_DATA_DEF;
```

### 2.7 TilingKey 完整编码表 (22个W8A8变体中的16个)

TilingKey 编码规则:
- 基线: `10000000000000000001UL` (int64 rowIndex + float scale + sharedInput存在 + groupListType=cumsum)
- RowIndex: int32 → +10
- Scale: bf16 → +100
- groupListType: index(0) → +1000
- sharedInput: null → +10000

```
完整Key                    rowIndex    scale     groupListType    sharedInput
────────────────────────────────────────────────────────────────────────────
10000000000000000001UL     int64       float     cumsum(1)        存在
10000000000000000011UL     int32       float     cumsum(1)        存在
10000000000000000101UL     int64       bf16      cumsum(1)        存在
10000000000000000111UL     int32       bf16      cumsum(1)        存在
10000000000000001001UL     int64       float     index(0)         存在
10000000000000001011UL     int32       float     index(0)         存在
10000000000000001101UL     int64       bf16      index(0)         存在
10000000000000001111UL     int32       bf16      index(0)         存在
10000000000000010001UL     int64       float     cumsum(1)        null
10000000000000010011UL     int32       float     cumsum(1)        null
10000000000000010101UL     int64       bf16      cumsum(1)        null
10000000000000010111UL     int32       bf16      cumsum(1)        null
10000000000000011001UL     int64       float     index(0)         null
10000000000000011011UL     int32       float     index(0)         null
10000000000000011101UL     int64       bf16      index(0)         null
10000000000000011111UL     int32       bf16      index(0)         null
```

### 2.8 确定性关闭时的行为

```cpp
// base_tiling.cpp:413-425
void DeterministicTilingProcess() {
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;
        return;  // ★ 立即返回，不做任何额外分配
    }
    // 以下仅在确定性=1时执行:
    deterministicFlag_ = 1;
    auto ascendcPlatform = PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(L2, l2_size);
    // 根据L2大小分配64MB或96MB确定性工作区
    deterWorkspaceSize_ = l2_size > 96MB ? 96MB : 64MB;
    workspaceSize_ += deterWorkspaceSize_;
}
```

---

## 3. Kernel 层完整解析

### 3.1 内核入口 —— TilingKey 分派

```cpp
// grouped_matmul_finalize_routing.cpp:110-171
extern "C" __global__ __aicore__ void grouped_matmul_finalize_routing(
    GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR pertoken_scale, GM_ADDR group_list, GM_ADDR share_input,
    GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, GM_ADDR y,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // 22个 if-else 分支分派到不同的模板实例
    if (TILING_KEY_IS(10000000000000000001UL)) {
        // int64_t rowIndex, float scale, cumsum groupList, sharedInput存在
        GMMFR_A8W8_IMPL(int64_t, float, false, false);
    } else if (TILING_KEY_IS(10000000000000000011UL)) {
        GMMFR_A8W8_IMPL(int32_t, float, false, false);
    }
    // ... 共22个分支 (16个W8A8 + 6个W8A4)
}
```

分派宏展开:
```cpp
#define GMMFR_A8W8_IMPL(RowIndexDtype, scaleType, groupListType, sharedInputIsNone)
    TPipe pipe;          // Pipeline管理器
    MT mm;                // MatMulImpl<int8 ND, int8 NZ, int32 ND>
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);         // AIC使用subBlockIdx=0
        mm.Init(&tilingData.matmulTiling, &pipe);
    }
    // 组装模板参数
    using param = Param<true,           // combine=true (W8A8始终走combine)
                        RowIndexDtype,  // int64_t 或 int32_t
                        GroupMatmulFRTilingData,
                        scaleType,      // float 或 bfloat16_t
                        groupListType,  // false(cumsum) 或 true(index)
                        sharedInputIsNone>;  // false(存在) 或 true(null)
    QuantGroupMatmul<param> op(mm);
    op.Init(initParams, &tilingData, &pipe);
    op.Process();
```

### 3.2 多核并行模型

```
KERNEL_TYPE_MIX_AIC_1_2: 每个物理核包含 1个AIC + 2个AIV

    ┌─────────────── Core N ───────────────┐
    │                                       │
    │  AIC (Cube, subBlockIdx=0)            │
    │    ├─ MatMul: int8×int8→int32        │
    │    ├─ CrossCoreSetFlag→AIV           │
    │    └─ CrossCoreWaitFlag←AIV          │
    │                                       │
    │  AIV 0 (Vector, subBlockIdx=0)        │
    │    ├─ M方向上半部分处理              │
    │    ├─ Dequant + PerToken + Bias      │
    │    └─ AtomicAdd → yGm               │
    │                                       │
    │  AIV 1 (Vector, subBlockIdx=1)        │
    │    ├─ M方向下半部分处理              │
    │    ├─ Dequant + PerToken + Bias      │
    │    └─ AtomicAdd → yGm               │
    │                                       │
    └───────────────────────────────────────┘

多核调度: curBlock ∈ [coreIdx, coreIdx+coreNum, coreIdx+2*coreNum, ...]
每个core轮转处理不同tile → 负载均衡
```

### 3.3 Init —— 全局缓冲区绑定

```cpp
// grouped_matmul_finalize_routing.h:147-176
void Init(initParams, tilingData, pipe) {
    // 绑定所有GM缓冲区
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(initParams.x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(initParams.weight));
    mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(initParams.workspace));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ SCALE_TYPE*>(initParams.scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initParams.pertoken_scale));
    groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(initParams.group_tokens));
    logitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initParams.logits));
    tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ ROW_INDEX_DTYPE*>(initParams.token_ranks));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT*>(initParams.y));

    // 如果是确定性，额外绑定mmQuantOutGm (指向workspace后半段)
    if (tiling->deterministicFlag == 1) {
        mmQuantOutGm.SetGlobalBuffer(...workspace + MatMul区偏移...);
    }

    subBlockIdx = GetSubBlockIdx();  // AIV: 0或1
    coreIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= GetTaskRation();  // AIV: coreIdx = coreIdx/2 (因为1AIC:2AIV)
    }
    InitUbBuffer();  // ★UB内存分配
}
```

### 3.4 InitUbBuffer —— UB 内存布局详解

```cpp
// grouped_matmul_finalize_routing.h:179-212
void InitUbBuffer() {
    if ASCEND_IS_AIC { return; }  // AIC不需要UB缓冲区

    // === TQue 双缓冲队列 (ping-pong, 通过硬件DMA自动管理) ===
    // scaleInQueue: 加载 weight scale (per-channel)
    pipe->InitBuffer(scaleInQueue, 2, baseN * sizeof(float));
    //   大小: 256 × 4 = 1KB (float) 或 256 × 2 = 512B (bf16)

    // biasInQueue: 加载 bias 值
    pipe->InitBuffer(biasInQueue, 2, baseN * sizeof(bfloat16_t));
    //   大小: 256 × 2 = 512B

    // queBind: 确定性专属暂存 (非确定性下不分配，省12KB)
    if (tiling->deterministicFlag == 1) {
        pipe->InitBuffer(queBind, 2, DETER_UB_SIZE);  // 12KB × 2
    }

    // perTokenScaleInQueue: 加载 per-token scale + logits (combine=1)
    if constexpr (P::combine) {
        // combine路径: scale和logits合并到同一buffer
        uint32_t perTokenScalebufferNum = (hasPertokenScale != 0) ? 2 : 1;
        pipe->InitBuffer(perTokenScaleInQueue, 2,
            Ceil(baseM/2 * sizeof(float) * perTokenScalebufferNum, 32) * 32);
        //   默认: Ceil(128/2 × 4 × 2, 32) × 32 = Ceil(512, 32) × 32 = 512B × 2 ping-pong
    }

    // vecInQueue: MatMul结果输入 (int32)
    pipe->InitBuffer(vecInQueue, 2, ubCalSize * sizeof(int32_t));
    //   大小: 4096 × 4 = 16KB × 2 ping-pong

    // vecOutQueue: 后处理结果输出 (float)
    pipe->InitBuffer(vecOutQueue, 2, ubCalSize * sizeof(DTYPE_OUT));
    //   大小: 4096 × 4 = 16KB × 2 ping-pong

    // === TBuf 静态分区 (共101376字节，固定偏移布局) ===
    pipe->InitBuffer(tmpBuff, ubRestBytes);  // 101376 bytes
    uint32_t ubCalSizeFloat = ubCalSize * sizeof(float);  // 4096 × 4 = 16384

    // 偏移0:        dequantMiddleResult  [16×256 float] = 16KB
    dequantMiddleResult = tmpBuff.GetWithOffset<float>(ubCalSize, 0);

    // 偏移16KB:     pertokenBrcbLocal    [16×256 float] = 16KB
    pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(ubCalSize, ubCalSizeFloat);

    // 偏移32KB:     mulsResultLocal      [16×256 float] = 16KB (复用为bf16 scale的目标)
    mulsResultLocal = tmpBuff.GetWithOffset<float>(ubCalSize, 2 * ubCalSizeFloat);

    // 偏移48KB:     biasCalcLocal        [16×256 float] = 16KB (bias广播+logit广播)
    biasCalcLocal = tmpBuff.GetWithOffset<float>(ubCalSize, 3 * ubCalSizeFloat);

    // 偏移64KB:     sharedTmpLocal       [2×16×256 float] = 32KB (AscendDequant临时空间)
    sharedTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 4 * ubCalSizeFloat);

    // 总计: 16KB+16KB+16KB+16KB+32KB = 96KB < 101376 bytes ✓
}
```

**UB 内存布局总图**:

```
UB Total: ~192KB (TQue + TBuf)
┌──────────────────────────────────────────────────────────────────────┐
│ TQue 双缓冲区域 (通过DMA硬件管理, ping-pong自动切换)                 │
│                                                                      │
│  scaleInQueue[0]: 1KB    scaleInQueue[1]: 1KB    (weight scale)     │
│  biasInQueue[0]:  512B   biasInQueue[1]:  512B   (bias)             │
│  perTokenScaleInQueue[0]: 512B  [1]: 512B  (pertoken scale+logit)  │
│  vecInQueue[0]:   16KB   vecInQueue[1]:   16KB  (MatMul int32结果)  │
│  vecOutQueue[0]:  16KB   vecOutQueue[1]:  16KB  (后处理float输出)   │
│  queBind[0]: 12KB  queBind[1]: 12KB  ← 确定性专属，非确定性下无    │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│ TBuf 静态偏移区域 (101376 bytes, 手动管理)                          │
│                                                                      │
│  Offset 0:     dequantMiddleResult  16KB  ← AscendDequant输出(float)│
│  Offset 16KB:  pertokenBrcbLocal    16KB  ← perToken scale广播结果  │
│  Offset 32KB:  mulsResultLocal      16KB  ← 乘法中间结果 + bf16转换│
│  Offset 48KB:  biasCalcLocal        16KB  ← bias广播 + logit中间    │
│  Offset 64KB:  sharedTmpLocal       32KB  ← AscendDequant API临时   │
│                                       96KB total                     │
│  Offset 96KB:  (剩余 ~3KB padding)                                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.5 Process —— 主循环详解

```cpp
// grouped_matmul_finalize_routing.h:290-335
void Process() {
    // ===== 阶段0: AIV预初始化 =====
    if ASCEND_IS_AIV {
        PreProcess();  // 输出归零 + sharedInput处理
        SyncAll();     // ★ 所有core在此同步，确保PreProcess完成
    }

    // ===== 阶段1: 初始化MNConfig和SyncConfig =====
    MNConfig mnConfig;
    mnConfig.baseM = tiling->matmulTiling.baseM;    // 128 或 256
    mnConfig.baseN = tiling->matmulTiling.baseN;    // 256 或 128
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);

    SyncConfig syncConfig;  // 确定性专属，非确定性下为空

    // ===== 阶段2: 逐组处理 =====
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        // 2.1 读取该组的M大小
        uint32_t m = static_cast<uint32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) { continue; }  // 空组跳过

        // groupListType=1(index): m是绝对值，需减去前组的累积偏移
        if constexpr (P::groupListType) {
            m -= mnConfig.offsetM;
        }

        mnConfig.m = m;
        mnConfig.blockDimM = Ceil(m, mnConfig.singleM);

        // 2.2 计算本组的block范围
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;
        uint32_t thresholdMDimN = thresholdBlockNum * mnConfig.blockDimN;  // 8×blockDimN

        // 2.3 处理本core负责的所有block
        while (curBlock < curCount) {
            // ★ 核心: 确定当前block的(mIdx, nIdx)
            MNBlockIdxCompute(mnConfig, curBlock, preCount, thresholdMDimN, tiling->deterministicFlag);

            // ★ AIC: MatMul计算
            MMCompute(groupIdx, mnConfig);

            // ★ 确定性同步 (非确定性下为no-op)
            VectorSync(mnConfig, syncConfig);

            // ★ AIV: 后处理
            VectorCompute(groupIdx, mnConfig, syncConfig);

            // 步进到下一个本core负责的block
            curBlock += tiling->coreNum;
        }

        // 2.4 更新累积状态
        preCount = curCount % tiling->coreNum;  // 跨组对齐
        mnConfig.offsetM += m;                   // 全局M偏移累积
    }

    // ===== 阶段3: 确定性尾部处理 (非确定性下为no-op) =====
    if (tiling->deterministicFlag == 1) {
        syncConfig.curM = mnConfig.offsetM;
        FRDeterministic(syncConfig);
    }
}
```

**Process 主循环流程图**:

```
                        ┌─────────────┐
                        │  AIV Only   │
                        │ PreProcess  │
                        │(归零+shared)│
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  SyncAll()  │ ← 所有core屏障同步
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         Group 0          Group 1         Group N-1
              │                │                │
    ┌─────────▼─────────┐     │                │
    │ m=groupTokens[0]  │     │                │
    │ blockDimM=ceil()  │     │                │
    │ curCount=BDM×BDN  │     │                │
    └─────────┬─────────┘     │                │
              │                │                │
    ┌─────────▼──────────────────────────────┐
    │  for curBlock in [coreIdx, +coreNum]:  │
    │                                        │
    │  ┌──────────────────────────────┐     │
    │  │  MNBlockIdxCompute()         │     │
    │  │  → (mIdx, nIdx)             │     │
    │  └──────────────┬───────────────┘     │
    │                 │                      │
    │  ┌──────────────▼───────────────┐     │
    │  │  MMCompute()    [AIC Only]   │     │
    │  │  int8×int8 → int32          │     │
    │  │  CrossCoreSetFlag→AIV       │     │
    │  └──────────────┬───────────────┘     │
    │                 │                      │
    │  ┌──────────────▼───────────────┐     │
    │  │  VectorSync()  [非确定性:空] │     │
    │  └──────────────┬───────────────┘     │
    │                 │                      │
    │  ┌──────────────▼───────────────┐     │
    │  │  VectorCompute() [AIV Only]  │     │
    │  │  Dequant + PerToken +       │     │
    │  │  Bias + AtomicAdd→yGm      │     │
    │  │  CrossCoreSetFlag→AIC       │     │
    │  └──────────────────────────────┘     │
    └───────────────────────────────────────┘
```

### 3.6 MNBlockIdxCompute —— 两种调度策略

```cpp
// grouped_matmul_finalize_routing_utils.h:140-161
void MNBlockIdxCompute(mnConfig, curBlock, count, thresholdMDimN, deterministicFlag) {
    // ===== 策略选择 =====
    if (mnConfig.blockDimM <= thresholdDimM     // ≤ 5: 小shape→行优先
        || thresholdDimM == 1                   // 强制行优先
        || deterministicFlag == 1) {            // 确定性→行优先
        // 行优先策略: 按 (mIdx, nIdx) = (block/BDN, block%BDN) 顺序排列
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    }
    else {
        // 对角块策略: 8×8对角线分配，减少L2缓存冲突
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(BDM*BDN, thresholdMDimN)
            ? BDM % thresholdBlockNum      // 尾部块
            : thresholdBlockNum;            // 常规块 (8或7)

        uint32_t curThresholdMThresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN = relativeBlock % thresholdMDimN
                >= AlignDown(curThresholdM * BDN, curThresholdMThresholdN)
            ? BDN % thresholdBlockNum       // 尾部N
            : thresholdBlockNum;             // 常规N (8或7)

        uint32_t localRelativeBlock = relativeBlock % thresholdMDimN % curThresholdMThresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM
                      + relativeBlock / thresholdMDimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock
                      + localRelativeBlock / LCM(curThresholdM, curThresholdN)) % curThresholdN
                      + relativeBlock % thresholdMDimN / curThresholdMThresholdN * thresholdBlockNum;
    }
}
```

**策略选择决策图**:

```
MNBlockIdxCompute 策略选择
│
├─ blockDimM ≤ 5 ?
│   ├─ YES → 行优先策略
│   │   mIdx = block / blockDimN
│   │   nIdx = block % blockDimN
│   │   特点: 顺序访问, 适合小M (M方向tile少)
│   │
│   └─ NO → 对角块策略
│       curThresholdM = ceil(blockDimM, 8)  常规块=8, 尾部块=blockDimM%8
│       curThresholdN = ceil(blockDimN, 8)  常规块=8, 尾部块=blockDimN%8
│       每个8×8对角块内: (m,n) 沿对角线分布
│       特点: 分散访问, 减少L2缓存冲突, 适合大M
│
└─ deterministicFlag == 1 ?
    └─ 强制行优先 (确定性需要保序)
```

---

## 4. 计算流水线详解

### 4.1 MMCompute —— AIC MatMul 完整流程

```cpp
// grouped_matmul_finalize_routing.h:337-374
void MMCompute(groupIdx, mnConfig) {
    // ===== Step 1: 计算尾部tile的实际大小 =====
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {  // 最后一个N tile
        curSingleN = tiling->n - tailN;               // 可能小于singleN
    }

    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {  // 最后一个M tile
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;  // 可能小于singleM
    }

    // ===== Step 2: 计算GM偏移 =====
    mnConfig.curBlockM = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM + curSingleM;
    // x偏移: 全局M偏移 × K
    uint64_t xOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
    // weight偏移: group × N×K + N偏移 × K
    uint64_t weightOffset = groupIdx * tiling->n * tiling->k + tailN * tiling->k;

    // workspace slot偏移: singleM×singleN × (coreIdx + (cubeCount%4) × coreNum)
    mnConfig.workSpaceOffset =
        mnConfig.singleN * mnConfig.singleM * (coreIdx + (cubeCount % tiling->parallNum) * tiling->coreNum);

    // ===== Step 3: AIC执行 =====
    if ASCEND_IS_AIC {
        // 3a. 流水同步: 前4个cube无需等待, 第5个起等AIV消费
        if (cubeCount >= tiling->parallNum) {  // parallNum=4
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);  // 等待AIV释放slot
        }

        // 3b. 设置MatMul参数
        mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
        mm.SetSingleShape(curSingleM, curSingleN, tiling->k);
        mm.SetTensorA(xGm[xOffset]);  // activation: INT8 ND [curSingleM × K]

        auto weightSlice = weightGm[weightOffset];
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(CACHE_MODE_DISABLE);  // 单M tile: 禁用L2缓存
        }
        mm.SetTensorB(weightSlice);  // weight: INT8 NZ [N × K]

        // 3c. K方向迭代
        uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
        while (mm.Iterate()) {
            // Iterate(): 每次处理 baseK=128 的K切片
            //   L1 ← A[curSingleM × 128] + B[128 × curSingleN] (int8)
            //   L0A ← 部分A, L0B ← 部分B
            //   Cube: int8×int8 → int32, 累加到L0C
            //   L0C → workspaceGM[worskspaceOffset] (addMode=true)
            mm.GetTensorC(mmOutGm[worskspaceOffset], 0, true);
            // addMode=true: workspaceGM中int32累加(不是覆盖)
            // K方向的每个frag都累加到同一位置 → 最终得到完整的 M×N int32

            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // mode=2: group内同步
            worskspaceOffset += (mnConfig.baseM * mnConfig.baseN);  // +128×256
            // 注意: worskspaceOffset递增是因为K方向frag在workspace中顺序存储
            // 但最终每个K frag的累加目标都是workSpaceOffset(起始值)
            // 这是Cube API的内部行为 — Iterate内部更新了L0C地址
        }
    }
    cubeCount++;
}
```

**MMCompute 数据流**:

```
  xGm[xOffset]                      weightGm[weightOffset]
  INT8 ND  [M×K]                    INT8 NZ  [K×N]
      │                                  │
      │  DMA → L1                        │  DMA → L1
      ▼                                  ▼
  L1 A Buffer [M×stepKa×baseK]       L1 B Buffer [stepKb×baseK×N]
      │                                  │
      │  DMA → L0A (每次baseK)           │  DMA → L0B (每次baseK)
      ▼                                  ▼
  ┌─────────────────────────────────────────┐
  │         Cube (脉动阵列)                 │
  │  L0A [M×128] × L0B [128×N]             │
  │  → L0C [M×N] int32  (累加模式)         │
  └──────────────────┬──────────────────────┘
                     │ GetTensorC(addMode=true)
                     ▼
  workspaceGM[workSpaceOffset .. +baseM×baseN]
  INT32 ND  [curSingleM × curSingleN]
```

**Workspace Slot 轮转机制**:

```
parallNum=4, coreNum=N, slotSize=singleM×singleN×sizeof(int32_t)

cubeCount   Core i 占用slot      Core i workspace偏移
   0        slot 0              i × slotSize
   1        slot 1              i × slotSize + 1 × coreNum × slotSize
   2        slot 2              i × slotSize + 2 × coreNum × slotSize
   3        slot 3              i × slotSize + 3 × coreNum × slotSize
   4        slot 0 (复用)        i × slotSize + 0 × coreNum × slotSize
   ...

AIC写入slot k时 AIV消费slot k-1 (如果存在)
→ 实现Cube计算和Vector后处理的流水并行
```

### 4.2 VectorCompute —— AIV 后处理完整流程

```cpp
// grouped_matmul_finalize_routing.h:420-469
void VectorCompute(groupIdx, mnConfig, syncConfig) {
    if ASCEND_IS_AIC { return; }

    // ===== Step 1: 计算tile边界 =====
    uint32_t curCubeSingleN = (mnConfig.nIdx == mnConfig.blockDimN - 1)
        ? (tiling->n - mnConfig.nIdx * mnConfig.singleN) : mnConfig.singleN;
    uint32_t curCubeSingleM = (mnConfig.mIdx == mnConfig.blockDimM - 1)
        ? (mnConfig.m - mnConfig.mIdx * mnConfig.singleM) : mnConfig.singleM;

    uint64_t mGlobalOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM;
    uint64_t outOffset = mGlobalOffset * tiling->n + mnConfig.nIdx * mnConfig.singleN;

    uint32_t vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, 8) * 8);
    // baseN=256 → vecBaseM = 4096 / (32×8) = 16
    // baseN=128 → vecBaseM = 4096 / (16×8) = 32
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;

    // ===== Step 2: N方向遍历 =====
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        uint32_t curVecBaseN = mnConfig.baseN;
        if (offsetN + mnConfig.baseN >= curCubeSingleN) {
            curVecBaseN = curCubeSingleN - offsetN;  // 尾部N
        }
        uint32_t alignBaseN = Ceil(curVecBaseN, 8) * 8;

        // 2a. 加载 weight scale (per-channel)
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);

        // 2b. 加载 bias (可选)
        if (hasBias) {
            DataCopyBias(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        }

        // 2c. 计算workspace偏移
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;

        // 2d. AIV子核M方向分工
        VectorOffsetParams coreOffsetM;
        GetOffset(coreOffsetM, curCubeSingleM);
        // subBlockIdx=0: offsetMStart=0,         offsetMEnd=singleCoreM
        // subBlockIdx=1: offsetMStart=singleCoreM, offsetMEnd=curCubeSingleM

        // 2e. 加载 per-token scale + logits (combine路径)
        DataCopyPerTokenScale(mnConfig, coreOffsetM.singleCoreM, coreOffsetM.perTokenOffsetM);
        perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();

        // 2f. ★同步: 等待AIC完成MatMul
        CrossCoreWaitFlag(SYNC_AIC_TO_AIV);

        // ===== Step 3: M方向向量处理 =====
        for (uint32_t offsetM = coreOffsetM.offsetMStart;
             offsetM < coreOffsetM.offsetMEnd;
             offsetM += vecBaseM) {

            uint32_t curVecBaseM = vecBaseM;
            if (offsetM + vecBaseM >= coreOffsetM.offsetMEnd) {
                curVecBaseM = coreOffsetM.offsetMEnd - offsetM;  // 尾部M
            }

            // 3a. DMA搬入MatMul结果 (int32) 到vecInQueue
            DataCopyMMOut(mmOutOffset, curVecBaseM, curVecBaseN, offsetM);

            // 3b. 计算向量参数
            VectorAtomicParams vecAParams{
                curVecBaseM, curVecBaseN, alignBaseN,
                offsetM, mGlobalOffset,
                mnConfig.nIdx * mnConfig.singleN + offsetN,     // yGmOffset0
                outOffset + offsetM * tiling->n + offsetN       // yGmOffset1
            };

            // 3c. 反量化 + per-token缩放 + bias
            ComputeDequantAndActivate(mnConfig, vecAParams, coreOffsetM);

            // 3d. 路由写回
            VectorAtomicProcess(vecAParams, syncConfig);
        }

        // 3e. 释放UB资源
        perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
        if constexpr (std::is_same_v<SCALE_TYPE, float>) {
            scaleInQueue.FreeTensor(scaleInUb);  // float scale直接释放
        }
    }

    // ===== Step 4: ★通知AIC: 本tile消费完毕 =====
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);
}
```

### 4.3 子函数详解

#### DataCopyScale —— Weight Scale 加载

```cpp
// grouped_matmul_finalize_routing.h:537-554
void DataCopyScale(curBaseN, alignBaseN, scaleOffset) {
    // 1. DMA: scaleGm → scaleInQueue → scaleInUb
    DataCopyExtParams scaleParams{1, curBaseN * sizeof(SCALE_TYPE), 1, 1, 0};
    LocalTensor<SCALE_TYPE> scaleLocal = scaleInQueue.AllocTensor<SCALE_TYPE>();
    DataCopyPad(scaleLocal, scaleGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);

    scaleInUb = scaleInQueue.DeQue<SCALE_TYPE>();
    scaleInUb.SetSize(alignBaseN);  // 对齐到8的倍数

    // 2. 如果scale是bf16: 转换为float并缓存到mulsResultLocal
    if constexpr (std::is_same_v<SCALE_TYPE, bfloat16_t>) {
        Cast(mulsResultLocal[alignBaseN],  // 目标: mulsResultLocal偏移alignBaseN后
             scaleInUb,                      // 源: bf16 scale
             CAST_NONE, alignBaseN);
        scaleInQueue.FreeTensor(scaleInUb);  // 释放bf16源
    }
    // 如果是float: scaleInUb直接被AscendDequant消费, 暂不释放
}
```

#### ComputeDequantAndActivate —— 反量化核心

```cpp
// grouped_matmul_finalize_routing.h:487-525
void ComputeDequantAndActivate(mnConfig, vecAParams, coreOffsetM) {
    // Step 1: PerToken Scale 广播
    // 将per-token scale [M, 1] 广播到 [curVecBaseM, alignBaseN]
    PerTokenScaleBrcb(mnConfig, vecAParams.curVecBaseM, vecAParams.alignBaseN,
                      vecAParams.offsetM, coreOffsetM);
    // → pertokenBrcbLocal[curVecBaseM × alignBaseN]: per-token scale 重复N列
    // → biasCalcLocal[curVecBaseM × alignBaseN]:   per-token logit 重复N列 (若有bias)

    // Step 2: 取出MatMul结果
    LocalTensor<int32_t> mmOutInUb = vecInQueue.DeQue<int32_t>();

    // Step 3: 选择scale缓冲区
    LocalTensor<float> scaleBuf;
    if constexpr (std::is_same_v<SCALE_TYPE, bfloat16_t>) {
        scaleBuf = mulsResultLocal[vecAParams.alignBaseN];  // bf16已转为float
    } else {
        scaleBuf = scaleInUb;  // float直接使用
    }

    // Step 4: ★ AscendDequant — int32 → float + per-channel scale
    AscendDequant(dequantMiddleResult,   // 输出: float [curVecBaseM × alignBaseN]
                  mmOutInUb,             // 输入: int32 [curVecBaseM × alignBaseN]
                  scaleBuf,              // per-channel scale [alignBaseN]
                  sharedTmpLocal,        // 临时空间
                  {curVecBaseM, alignBaseN, curVecBaseN});  // 维度
    // AscendDequant内部:
    //   Cast<int32→float>(mmOutInUb) × scaleBuf → dequantMiddleResult
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);

    // Step 5: Per-token 缩放 + Bias
    uint32_t computeSize = vecAParams.curVecBaseM * vecAParams.alignBaseN;
    if constexpr (P::combine) {
        LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
        if (hasBias) {
            // dequantMiddleResult = dequant × pertoken_brcb
            Mul(dequantMiddleResult, dequantMiddleResult, pertokenBrcbLocal, computeSize);
            PipeBarrier<PIPE_V>();
            // yLocalInUb = dequantMiddleResult + biasCalcLocal
            Add(yLocalInUb, dequantMiddleResult, biasCalcLocal, computeSize);
            PipeBarrier<PIPE_V>();
        } else {
            // yLocalInUb = dequant × pertoken_brcb
            Mul(yLocalInUb, dequantMiddleResult, pertokenBrcbLocal, computeSize);
        }
        vecOutQueue.EnQue(yLocalInUb);  // 入队，供VectorAtomicProcess消费
    }
}
```

**反量化计算公式**:

```
对于每个元素 (i, j):
  dequant[i][j] = Cast<float>(mmOut[i][j]) × weight_scale[j]

对于 combine 路径:
  y[i][j] = dequant[i][j] × per_token_scale[i] × logit[i]
            + bias[j] (如果有)

per_token_scale 的处理:
  perTokenScaleInUb 布局: [logit_0, logit_1, ..., logit_{M-1}, scale_0, scale_1, ..., scale_{M-1}]
  当 hasPertokenScale=1: DataCopyPerTokenScale 中执行 Mul(scale, logit, scale, curBaseM)
  结果: scale[i] = scale[i] × logit[i]  (合并为最终per-token缩放因子)
```

#### PerTokenScaleBrcb —— Per-Token Scale 广播

```cpp
// grouped_matmul_finalize_routing.h:575-591
void PerTokenScaleBrcb(mnConfig, curBaseM, alignBaseN, offsetM, coreOffsetM) {
    // perTokenScaleInUb 布局: [logit或logit×scale] + [per_token_scale或logit×scale]
    // combine路径: perTokenScaleInUb[0..M-1] = logit×scale (合并后的)
    //              perTokenScaleInUb[M..2M-1] = logit (logit保持)
    // 实际上 scale和logit已通过Mul合并，perTokenScaleInUb[0..M-1]就是最终因子

    uint32_t alignBaseM = (hasPertokenScale != 0)
        ? (Ceil(coreOffsetM.singleCoreM, 8) * 8) : 0;

    // per-token scale 广播: [curBaseM, 1] → [curBaseM, alignBaseN]
    const uint32_t broadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(
        pertokenBrcbLocal,                              // 目标 [M×N]
        perTokenScaleInUb[offsetM - coreOffsetM.offsetMStart + alignBaseM],  // 源 [M×1]
        broadCastDst, broadCastSrc, sharedTmpLocal);    // (M,N) ← (M,1)

    // 如果有bias: 还需要广播logit到biasCalcLocal
    if (hasBias) {
        // logit广播: [curBaseM, 1] → [curBaseM, alignBaseN]
        BroadCast<float, 2, 1>(
            biasCalcLocal,
            perTokenScaleInUb[offsetM - coreOffsetM.offsetMStart],  // logit在开头
            broadCastDst, broadCastSrc, sharedTmpLocal);

        // biasCalcLocal[i][j] = biasCalcLocal[i][j] (logit) × mulsResultLocal[j] (bias)
        for (int i = 0; i < curBaseM; i++) {
            Mul(biasCalcLocal[alignBaseN * i],     // 第i行logit广播
                biasCalcLocal[alignBaseN * i],
                mulsResultLocal,                    // bias值 (在DataCopyBias中准备好)
                alignBaseN);
        }
    }
}
```

#### GetOffset —— AIV 子核 M 方向分工

```cpp
// grouped_matmul_finalize_routing.h:405-418
void GetOffset(offset, curCubeSingleM) {
    offset.singleCoreM = curCubeSingleM / 2;  // M分两半, 每个AIV处理一半
    if (subBlockIdx == 0) {
        offset.perTokenOffsetM = 0;
        offset.offsetMStart = 0;
        offset.offsetMEnd = offset.singleCoreM;        // [0, singleCoreM)
    } else {
        offset.perTokenOffsetM = offset.singleCoreM;
        offset.offsetMStart = offset.singleCoreM;       // [singleCoreM, curCubeSingleM)
        offset.offsetMEnd = curCubeSingleM;
        offset.singleCoreM = curCubeSingleM - offset.singleCoreM;  // 后半大小
    }
}
// 示例: curCubeSingleM=192
//   subBlockIdx=0: singleCoreM=96, [0, 96)
//   subBlockIdx=1: singleCoreM=96, [96, 192)
```

### 4.4 VectorAtomicProcess —— 非确定性写回 (关键)

```cpp
// grouped_matmul_finalize_routing.h:377-402
void VectorAtomicProcess(vecAParams, syncConfig) {
    LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();

    if constexpr (P::combine) {  // W8A8始终走combine=true
        if (tiling->deterministicFlag == 1) {
            // ★确定性路径: 先写mmQuantOutGm暂存, 后续FRDeterministic保序分发
            DataCopy2DDimParams dimParams{curVecBaseM, curVecBaseN, alignBaseN};
            DataCopyPad2D(mmQuantOutGm[yGmOffset1 - (lowBoundM - windowSize) * n],
                          yLocal, dimParams, n);
            vecOutQueue.FreeTensor(yLocal);
            return;
        }

        // ★非确定性路径:
        SetAtomicAdd<float>();  // ← MTE引擎开启原子加模式
        DataCopyExtParams paramsOut{
            1,                                              // blockCount: 每次拷贝1行
            curVecBaseN * sizeof(float),                    // blockLen: N个float字节数
            1, 1, 0};
        for (uint32_t i = 0; i < vecAParams.curVecBaseM; i++) {
            // 通过路由表查询: 输入第i行 → 输出哪一行
            auto outRow = static_cast<uint64_t>(
                tokenRanksGm.GetValue(vecAParams.mGlobalOffset + vecAParams.offsetM + i));
            // tokenRanksGm: [M] int64/int32, 存储路由映射表
            //   例如 tokenRanksGm[5] = 3 → 输入行5的输出应当路由到输出行3

            // 原子加到目标位置: yGm[outRow × N + N偏移]
            DataCopyPad(yGm[outRow * tiling->n + vecAParams.yGmOffset0],
                        yLocal[i * vecAParams.alignBaseN],  // UB中第i行的起始地址
                        paramsOut);
        }
        SetAtomicNone();  // 关闭原子加模式

    } else {
        // combine=false: 直接按顺序写回 (非GMMFR W8A8路径)
        DataCopy2DDimParams dimParams{curVecBaseM, curVecBaseN, alignBaseN};
        DataCopyPad2D(yGm[yGmOffset1], yLocal, dimParams, n);
    }
    vecOutQueue.FreeTensor(yLocal);
}
```

**非确定性写回的核心机制**:

```
tokenRanksGm (路由表):
  [0]: 2  (输入行0 → 输出行2)
  [1]: 0  (输入行1 → 输出行0)
  [2]: 2  (输入行2 → 输出行2)  ← 行0和行2路由到同一目标!
  ...

MoE场景: 多个expert的输出token可能路由到同一个输出位置
  → 需要AtomicAdd保证正确性
  → 但不同core的AtomicAdd到达顺序不确定 → 非确定性!

写回过程 (以curVecBaseM=16为例):
  for i in 0..15:
    outRow = tokenRanksGm[mGlobalOffset + offsetM + i]
    AtomicAdd(yGm[outRow * N + nOffset], yLocal[i * alignBaseN], N个float)
```

### 4.5 完整同步时序

```
Timeline (单个tile的AIC-AIV交互):

AIC (Cube Core)                              AIV (Vector Core)
│                                                   │
│ cube 0: 启动MatMul                                │
│   K iter 0: int8×int8→int32                       │
│   GetTensorC(ws[0], addMode)                      │
│   CrossCoreSetFlag(SYNC_AIC_TO_AIV=5) ──────────→ │
│   K iter 1: ...                                   │ CrossCoreWaitFlag(5)
│   ...                                             │ DataCopyScale()
│   K iter N-1: 完成                                │ DataCopyBias()
│                                                   │ DataCopyPerTokenScale()
│                                                   │ Dequant + PerToken + Bias
│                                                   │ VectorAtomicProcess()
│                                                   │ ← CrossCoreSetFlag(SYNC_AIV_TO_AIC=3)
│                                                   │
│ cube 1: 启动                                      │
│ cube 2: 启动                                      │
│ cube 3: 启动  (parallNum=4, slot满)               │
│                                                   │
│ cube 4:                                           │
│ CrossCoreWaitFlag(SYNC_AIV_TO_AIC=3) ←────────── │ (等待slot 0被AIV消费)
│ MatMul → workspace slot 0                         │
│ ...                                               │

事件值:
  SYNC_AIC_TO_AIV = 5  (AIC通知AIV: Cube结果就绪)
  SYNC_AIV_TO_AIC = 3  (AIV通知AIC: 已消费, slot可复用)
  CrossCoreSetFlag/WaitFlag mode=2: group内同步
```

---

## 5. 不同 Shape 分支的模拟分析

### 5.1 场景一: 通用小Shape (默认路径)

**输入**: groupNum=4, m=256 (每group 64行), n=1024, k=4096, scale=float, coreNum=2

```
Tiling参数:
  baseM=128, baseN=256, baseK=128
  vBaseM=16, singleM=128, singleN=256
  blockDimM=Ceil(64,128)=1, blockDimN=Ceil(1024,256)=4
  workspace: 512KB×2core = 1MB (MatMul) + 20MB (RPC)

Group 0 (m=64):
  MNBlockIdxCompute: 行优先 (blockDimM=1≤5)
  Core 0: tiles=(0,0),(0,2)   Core 1: tiles=(0,1),(0,3)

  MMCompute tile (0,0): curSingleM=64, curSingleN=256
    xOffset = 0*4096 = 0
    weightOffset = 0*1024*4096 + 0*256*4096 = 0
    wsSlot = 128*256*(0+0*2) = 0  (Slot 0)
    K迭代: ceil(4096/128)=32次, 每次128×256 int8×int8→int32
    每次CrossCoreSetFlag→AIV

  VectorCompute tile (0,0):
    curCubeSingleM=64, curCubeSingleN=256
    mGlobalOffset=0, outOffset=0
    vecBaseM = min(4096/(32×8)=16, 64) = 16
    GetOffset: subBlockIdx=0→[0,32), subBlockIdx=1→[32,64)

    N遍历: offsetN=0, curVecBaseN=256
    M遍历 (subBlockIdx=0): offsetM=0(rows 0..15), offsetM=16(rows 16..31)
      每次VectorAtomicProcess: 16行×16次tokenRanksGm查询
        例: outRow = tokenRanksGm[0+i]
            AtomicAdd(yGm[outRow*1024+0], yLocal[i*256], 256 floats)

  Group 1..3: 同理推进, offsetM=64,128,192
```

### 5.2 场景二: DeepSeek MoE 特殊Shape —— baseM/baseN交换

**输入**: groupNum=4, m=768 (avg=192), n=7168, k=2048, coreNum=4

```
Tiling参数 (命中特殊分支):
  avg_m = 768/4 = 192, 128<192≤256 → 交换!
  baseM=256, baseN=128, baseK=128
  vBaseM=4096/(16×8)=32, singleM=256, singleN=128

对比默认路径:
  ┌──────────────┬────────────┬──────────────┐
  │   参数        │  默认      │  特殊交换    │
  ├──────────────┼────────────┼──────────────┤
  │  baseM       │  128       │  256         │
  │  baseN       │  256       │  128         │
  │  vBaseM      │  16        │  32          │
  │  blockDimN   │  28        │  56          │
  │  slot大小    │  128KB     │  128KB       │
  │  K迭代次数   │  16        │  16          │
  └──────────────┴────────────┴──────────────┘

Group 0 (m=192, blockDimM=Ceil(192,256)=1):
  56个N tile, 行优先调度

  Core 0: tiles=(0,0),(0,4),(0,8),...,(0,52) → 14个tile
  Core 1: tiles=(0,1),(0,5),(0,9),...,(0,53) → 14个tile
  Core 2: tiles=(0,2),(0,6),(0,10),...,(0,54) → 14个tile
  Core 3: tiles=(0,3),(0,7),(0,11),...,(0,55) → 14个tile

  VectorCompute tile (0,0):
    curCubeSingleM=192, curCubeSingleN=128
    vecBaseM = min(32, 192) = 32
    GetOffset: subBlockIdx=0→[0,96), subBlockIdx=1→[96,192)
    M遍历 (subBlockIdx=0): offsetM=0(32行), 32(32行), 64(32行) — 3次

  VectorAtomicProcess (offsetM=0, 32行):
    outRow = tokenRanksGm[0+0+i]  for i∈[0,31]
    AtomicAdd(yGm[outRow*7168+0], ..., 128 floats)

  对比vBaseM=16时: 需要6次M遍历, 现在是3次
  → AIV处理效率提升, 但N方向tile翻倍
  → 总开销权衡: 更大的M分块减少M循环次数, 代价是更多N tile的scale/bias加载
```

### 5.3 场景三: 大Shape —— 对角块调度策略

**输入**: groupNum=1, m=1280, n=3840, k=8192, coreNum=4

```
Tiling参数:
  baseM=128, baseN=256, baseK=128
  blockDimM=Ceil(1280,128)=10 (>5 → 对角策略)
  blockDimN=Ceil(3840,256)=15
  总block: 10×15=150, 每core: ~38个block

MNBlockIdxCompute: 对角策略
  8×8对角块: block 0..63 → 常规块
  尾部块: block 64..89 → 2×7+7×5 混合块
  剩余: block 90..149 → 各core轮转

  对角块分配 (Core 0):
    rB=0:  (0,0)  →  M[0..127],    N[0..255]
    rB=4:  (4,4)  →  M[512..639],  N[1024..1279]
    rB=8:  (0,1)  →  M[0..127],    N[256..511]
    rB=12: (4,5)  →  M[512..639],  N[1280..1535]
    ...

  VectorCompute tile (4,4):
    mGlobalOffset = 4*128 = 512
    outOffset = 512*3840 + 4*256 = 1967104
    curCubeSingleM=128 (非尾部), curCubeSingleN=256 (非尾部)

    scaleOffset = 0*3840 + 4*256 = 1024  (group0, N偏移)
    N遍历: offsetN=0, curVecBaseN=256
    M遍历 (subBlockIdx=0, 4次×16行):
      offsetM=0:  yGmOffset=1967104+0*3840+0=1967104
      offsetM=16: yGmOffset=1967104+16*3840=2028544
      offsetM=32: yGmOffset=1967104+32*3840=2089984
      offsetM=48: yGmOffset=1967104+48*3840=2151424

    VectorAtomicProcess (offsetM=0, 16行):
      for i in 0..15:
        outRow = tokenRanksGm[512+0+i]
        AtomicAdd(yGm[outRow*3840+1024], yLocal[i*256], 256 floats)
```

### 5.4 三种场景对比总结

```
                    场景一            场景二              场景三
                    (通用小Shape)     (DS MoE特殊)        (大Shape对角)
─────────────────────────────────────────────────────────────────────
n, k                1024, 4096        7168, 2048          3840, 8192
m, groupNum         256, 4           768, 4              1280, 1
avg_m               64               192                 1280
─────────────────────────────────────────────────────────────────────
Tiling:
  baseM             128              256 (交换!)         128
  baseN             256              128 (交换!)         256
  vBaseM            16               32                  16
  N-tiles/group     4                56                  15
─────────────────────────────────────────────────────────────────────
Kernel:
  blockDimM         1                1                   10
  blockDimN         4                56                  15
  MNBlock策略       行优先           行优先              对角块
  K迭代次数         32               16                  64
  AIV M循环/次      2(set of 16)     3(set of 32)        4(set of 16)
  slot大小           128KB            128KB               128KB
  workspace总量      ~21MB            ~22MB               取决于coreNum
─────────────────────────────────────────────────────────────────────
  AIC→AIV通知次数   32×1×4=128       16×1×56=896         64×150/4=2400
  AIV→AIC释放次数   4                56                  150/4=38
```

---

## 6. 非确定性 vs 确定性对比

| 维度 | 非确定性 (deterministicFlag=0) | 确定性 (deterministicFlag=1) |
|------|-------------------------------|------------------------------|
| **Workspace** | 仅MatMul缓冲区 (512KB×coreNum) | +64MB/96MB 确定性暂存区 |
| **UB** | 无 queBind (省12KB×2) | queBind = 12KB×2 (暂存) |
| **写回目标** | 直接 `yGm[outRow]` via AtomicAdd | 先写 `mmQuantOutGm`, 再 `FRDeterministic` 重新分发 |
| **写回顺序** | 不确定 (atomics竞争) | 确定 (按行顺序barrier后分发) |
| **MNBlockIdxCompute** | 可用对角策略 | 行优先 (强制顺序) |
| **额外同步** | 无 | `VectorSync()` 管理滑动窗口 |
| **FRDeterministic** | 无 | 每window大小barrier, 重新按行分发 |
| **性能** | 更高 (无额外同步和拷贝) | 较低 (额外sync + 二次分发) |

---

## 7. 完整数据流总图

```
═══════════════════════════════════════════════════════════════════════════
                           GM (Global Memory / HBM)
═══════════════════════════════════════════════════════════════════════════
  │              │              │                │              │
  │ x[int8 ND]   │ w[int8 NZ]   │ scale[float]   │ pertoken     │ bias[bf16]
  │ [M×K]        │ [E×N×K]      │ [E×N]          │ _scale[float]│ [E×N]
  │              │              │                │ [M]+logit[M] │ (可选)
  │              │              │                │              │
  │  tokenRanks  │ groupTokens  │ workspaceGM    │ share_input  │ y[float ND]
  │  [M] int64   │ [E] int64    │ [int32 slots]  │ [bf16 ND]    │ [batch×N]
  │              │              │                │ (可选)       │ ← 最终输出
  └──────┬───────┴──────┬───────┴───────┬────────┴──────┬───────┴────┬─────
         │              │               │               │            │
═════════╪══════════════╪═══════════════╪═══════════════╪════════════╪═════
         │              │               │               │            │
    ┌────▼──┐      ┌───▼───┐     ┌────▼───┐     ┌────▼───┐  ┌─────▼─────┐
    │ L1 A  │      │ L1 B  │     │  (跳过) │     │  (跳过) │  │  (跳过)   │
    │ Buffer│      │ Buffer│     │         │     │         │  │           │
    └───┬───┘      └───┬───┘     └─────────┘     └─────────┘  └───────────┘
        │              │
   ┌────▼──┐     ┌───▼────┐
   │ L0A   │     │ L0B    │              ╔══════════════════════╗
   │[M×128]│     │[128×N] │              ║   AIC (Cube Core)    ║
   └───┬───┘     └───┬────┘              ╚══════════════════════╝
       └──────┬──────┘
              │
         ┌────▼────┐
         │  Cube   │  int8×int8 → int32
         │ 脉动阵列│  L0C [M×N]
         └────┬────┘
              │ GetTensorC(addMode=true)
              ▼
    workspaceGM[slot]  INT32 ND  [curSingleM × curSingleN]
              │
══════════════╪═══════════════════════════════════════════════════════════
              │               ╔══════════════════════╗
              │               ║   AIV (Vector Core)   ║
              │               ╚══════════════════════╝
              │
    ┌─────────▼──────────┬─────────────────┬──────────────────┐
    │ DataCopyMMOut      │ DataCopyScale   │ DataCopyPerToken │
    │ (int32→vecInQueue) │ (float→scaleUb) │ (float→ptScaleUb)│
    │                    │ + bf16→float    │ + logit合并      │
    └─────────┬──────────┴────────┬────────┴────────┬─────────┘
              │                   │                 │
    ┌─────────▼───────────────────▼─────────────────▼─────────┐
    │              AscendDequant()                             │
    │  int32 → float  (Cast)                                  │
    │  float × weight_scale[j]  (per-channel)                 │
    │  → dequantMiddleResult [M×N] float                      │
    └──────────────────────────┬──────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────┐
    │              PerTokenScaleBrcb()                          │
    │  pertoken_scale[i] 广播到 [M×N]  → pertokenBrcbLocal   │
    │  logit[i] 广播到 [M×N] (若有bias) → biasCalcLocal      │
    └──────────────────────────┬──────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────┐
    │              Combine (W8A8=combine=true)                 │
    │  dequant × pertokenBrcb  (per-token缩放)               │
    │  + biasCalcLocal  (加bias, 可选)                        │
    │  → vecOutQueue [M×N] float                              │
    └──────────────────────────┬──────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────┐
    │          VectorAtomicProcess()   ★非确定性核心★         │
    │                                                         │
    │  SetAtomicAdd<float>()                                  │
    │  for i in 0..curVecBaseM-1:                             │
    │    outRow = tokenRanksGm.GetValue(mGlobal+i)            │
    │    AtomicAdd(                                           │
    │      yGm[outRow × N + nOffset],                         │
    │      yLocal[i × alignN],                                │
    │      curVecBaseN floats)                                │
    │  SetAtomicNone()                                        │
    └─────────────────────────────────────────────────────────┘
```

---

## 附录A: 关键常量速查

| 常量 | 值 | 含义 | 位置 |
|------|----|------|------|
| `BEST_BASE_M` | 128 | Cube默认M分块 | base_tiling.cpp:22 |
| `BEST_BASE_N` | 256 | Cube默认N分块 | base_tiling.cpp:24 |
| `BEST_BASE_K` | 128 | Cube K分块 | base_tiling.cpp:23 |
| `UBCALSIZE` | 4096 | 向量计算单元=16×256 | base_tiling.cpp:48 |
| `CV_PARALL_NUM` | 4 | Cube slot并行数 | base_tiling.cpp:62 |
| `A8W8_UBRESTBYTES` | 101376 | TBuf大小(约99KB) | base_tiling.cpp:51 |
| `SYNC_AIV_TO_AIC` | 3 | AIV→AIC事件ID | utils.h:28 |
| `SYNC_AIC_TO_AIV` | 5 | AIC→AIV事件ID | utils.h:29 |
| `DETER_UB_SIZE` | 12KB | 确定性专属queBind大小 | utils.h:30 |
| `thresholdBlockNum` | 8 | 对角策略基础块数 | utils.h:19 |
| `thresholdDimM` | 5 | 大小shape区分阈值 | utils.h:22 |
| `ROW_INDEX_FACTOR` | 10 | TilingKey: int32 rowIndex | base_tiling.cpp:64 |
| `SCALE_FACTOR` | 100 | TilingKey: bf16 scale | base_tiling.cpp:65 |
| `SHARED_INPUT_FACTOR` | 10000 | TilingKey: sharedInput null | base_tiling.cpp:66 |
| `GROUP_LIST_TYPE_FACTOR` | 1000 | TilingKey: groupListType=0 | base_tiling.cpp:68 |

## 附录B: 涉及的核心文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `op_host/.../grouped_matmul_finalize_routing_base_tiling.cpp` | 549 | W8A8 Tiling, FillTilingData |
| `op_host/.../grouped_matmul_finalize_routing_tiling.h` | 65 | TilingData 结构体定义 |
| `op_host/.../grouped_matmul_finalize_routing_tiling.cpp` | 119 | 架构路由 + Tiling 注册 |
| `op_kernel/.../grouped_matmul_finalize_routing.cpp` | 171 | Kernel 入口 + TilingKey 分派 |
| `op_kernel/.../grouped_matmul_finalize_routing.h` | 688 | QuantGroupMatmul 完整实现 |
| `op_kernel/.../grouped_matmul_finalize_routing_utils.h` | 204 | 数据结构 + MNBlockIdxCompute |
