# K_SHIFT — K轴错峰优化

> 主仓: **ops-nn** (`mat_mul_v3`)  |  辅仓: omni-ops (`ai_infra_matmul`，衍生版本)

---

## 1. 硬件原理

### 1.1 问题: 多核齐步造成 L2 bank 冲突

Ascend NPU 多核共享 L2 Cache。L2 内部按地址 hash 到多个 **bank** 组，
每组每周期只服务一个请求。多核从 K 方向搬运 A/B 矩阵数据时，若全部从相同 K 偏移开始:

```
  时间 →    t0    t1    t2    t3    t4
  Core0    [L2]  [L2]  [L2]  [L2]
  Core1    [L2]  [L2]  [L2]  [L2]    ← 同时命中同一 bank 组
  Core2    [L2]  [L2]  [L2]  [L2]    ← L2 仲裁排队，有效带宽暴跌
  Core3    [L2]  [L2]  [L2]  [L2]
```

### 1.2 解决: 各核从不同 K 偏移开始搬运

`GetMDLConfig(enKShift=true, enPeakStagger=true)` 使 Cube 硬件为每个核
计算不同的 K 起始偏移，错开 L2 访问窗口:

```
  Core i 的 K 起始偏移 = i × ceil(total_K / usedCoreNum)
  每个核仍然遍历完整 K 范围，环形迭代 (到末尾后回到 0)

  时间 →    t0    t1    t2    t3    t4    t5    t6    t7
  Core0    [L2]  [L2]  [L2]  [L2]
  Core1          [L2]  [L2]  [L2]  [L2]                    ← 错峰!
  Core2                [L2]  [L2]  [L2]  [L2]                 L2 请求在时间轴
  Core3                      [L2]  [L2]  [L2]  [L2]           均匀分布，冲突骤减
```

**效果**: L2 有效带宽利用率提升 15-30%。

### 1.3 关键认知

K_SHIFT **不改变计算语义** — 每核仍然累加完整 K 范围的所有 tile，数学结果等价。
但由于 **FP 加法非结合** `(a+b)+c ≠ a+(b+c)`，不同核的 FP 累加结果可能有 ≤2 ULP 差异。
这是第 8 节 batch 一致性问题的根因。

---

## 2. 完整调用链路

```
┌─ Python ───────────────────────────────────────────────────────────┐
│ aclnnMatMulV3(self, mat2, out, cubeMathType)  // 标准 ACLNN 入口    │
│ torch_npu.npu_quant_matmul(x1, x2, scale, ...)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─ ACLNN API (Host) ─────────────────────────────────────────────────┐
│ MatMulV3 / BatchMatMulV3 / GemmV3  (算子族，主仓 ops-nn)             │
│   └─ AicoreOp 图编译 → 触发 Tiling 回调                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─ Tiling 阶段 (Host CPU) ───────────────────────────────────────────┐
│                                                                     │
│  MatmulV3BaseTiling::DoOpTiling()                                   │
│  │                                                                  │
│  ├─ GetMoreArgs()    ← shape/dtype/format/transpose/nd2nz          │
│  │   GetFormat() → aFormat/bFormat/outFormat                       │
│  │   GetDtype()  → aType/bType/cType/biasType, isHf32              │
│  │   GetShape()  → mValue/kValue/nValue, isATrans/isBTrans         │
│  │                                                                  │
│  ├─ InitTilingData()  MatmulTiling API 初始化                       │
│  ├─ SetRunInfo() / SetParamsV310()                                 │
│  │                                                                  │
│  ├─★ GetTilingFromRepo()?  ← AOE 调优知识库 (按 shape 签名查)      │
│  │  │  (matmul_v3_base_tiling.cpp:731 / GetTilingFromRepo:958)     │
│  │  │                                                                │
│  │  ├─YES (repo 命中)──────────────────────────────────────────────┐│
│  │  │   TranslateAoeTiling(tuningTiling)  // "Get tiling from repo"││
│  │  │     载入整套预调 tiling (aoeTiling->*):                       ││
│  │  │       stepKa/stepKb, depthA1/B1, baseM/N/K, usedCoreNum,    ││
│  │  │       singleCoreM/N/K, iterateOrder, dbL0A/B/C ...          ││
│  │  │     + aoeTiling->tilingEnable                                ││
│  │  │         → CheckAoeTilingEnable(:1096) 解析十进制位:          ││
│  │  │             个位→SPLITCOREMODE 十位→LOADMODE(FULLLOAD)      ││
│  │  │             千位→FIXOPTI    万位→SPECIALOPT ★              ││
│  │  │         → tilingEnableSpecialOpti = 万位 (K_SHIFT? : BASE)  ││
│  │  │           ★ 这是 tilingEnableSpecialOpti 的【唯一】写入点   ││
│  │  │             (matmul_v3_base_tiling.cpp:181, 全仓仅此一处)   ││
│  │  │   DoNd2NzVectorTiling() / SetNd2NzInfo() / bias & MC_SPLIT_K││
│  │  │   return  ★ 直接返回, 跳过下面 SelectNZTiling/DoBasicTiling ││
│  │  │             /OptimizeBasicKernelStepK (启发式不跑)          ││
│  │  │──────────────────────────────────────────────────────────────┘│
│  │  │                                                                │
│  │  └─NO (repo 未命中) → 运行时启发式 tiling:                       │
│  │      ├─ SelectNZTiling()   NZ format → 限制 TilingCalcSelect    │
│  │      ├─ DoBasicTiling()    计算 baseM/N/K, singleCoreM/N/K      │
│  │      │   ├─ ResetBase()          L0C 决定 baseM=128/256         │
│  │      │   ├─ SetBaseBlockTiling() 按 transpose 调整 baseM/baseN  │
│  │      │   ├─ CalL1Tiling()         depthA1/B1, stepKa/Kb(初值=8) │
│  │      │   ├─ DoSelectTiling()      BL1/AL1/L2Cache 路径选择      │
│  │      │   └─ DoNd2NzVectorTiling() nd2nz 切分参数                │
│  │      ├─ ★ OptimizeBasicKernelStepK()  ← stepK 调整 (非模板选择) │
│  │      │     满足前置门+形状区间: stepKa/stepKb 8→4, depthA1/B1  │
│  │      │     减半。★ 不写 tilingEnableSpecialOpti (启发式不开K_SHIFT)│
│  │      └─ SetNd2NzInfo()     baseAN/AD/BN/BD                      │
│  │                                                                  │
│  │  ★ 关键: repo 命中才有 K_SHIFT 模板 (万位=1); 启发式路径 SPECIALOPT│
│  │    恒为 BASE。stepK=4 两条路径都可能 (repo 直给 / 启发式命中区间)。│
│  │                                                                  │
│  DoLibApiTiling()                                                   │
│  ├─ SetRunInfo()     写回 tilingData (usedCoreNum, singleCoreM等)  │
│  ├─ SetNd2NzInfo()                                                  │
│  └─ ★ DoTilingKey()   GET_TPL_TILING_KEY(                          │
│         LOADMODE, SPLITCOREMODE, FIXOPTI, MIXND2NZ,                 │
│         SPECIALOPT,  ← tilingEnable_.tilingEnableSpecialOpti        │
│         FP32ADDMM)                                                  │
│                                                                     │
│  PostTiling()                                                       │
│  ├─ memcpy_s 写回 RawTilingData                                    │
│  ├─ SetBlockDim(usedCoreNum)                                       │
│  └─ SetScheduleMode(0)  (BASE 无核间同步)                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ TilingKey (含 SPECIALOPT bit)
                               ▼
┌─ 编译期模板选择 (ASCENDC_TPL_SEL) ──────────────────────────────────┐
│                                                                     │
│  mat_mul_v3_tiling_key.h (主仓 ops-nn)                               │
│                                                                     │
│  ASCENDC_TPL_ARGS_DECL 定义 6 维模板参数:                            │
│    LOADMODE(4bit) × SPLITCOREMODE(8bit) × FIXOPTI(4bit)             │
│    × MIXND2NZ(4bit) × SPECIALOPT(4bit) × FP32ADDMM(4bit)           │
│                                                                     │
│  #else 默认分支 (ND×ND / ND×NZ):                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 1. MIX_AIC_1_2, BASE, BASE, BASE, TRUE,        K_NOT_SHIFT   │   │
│  │ 2. AIC_ONLY,    BASE, BASE, BASE, FALSE,       K_NOT_SHIFT   │   │
│  │ 3. AIC_ONLY,    BASE, BASE, BASE, FALSE,       K_SHIFT    ★ │   │
│  │ 4. AIC_ONLY,    BASE, AL1,  BASE, FALSE,       K_NOT_SHIFT   │   │
│  │ 5. MIX_AIC_1_2, BASE, BL1,  BASE, TRUE,        K_NOT_SHIFT   │   │
│  │ 6. AIC_ONLY,    BASE, BL1,  BASE, FALSE,       K_NOT_SHIFT   │   │
│  │ 7. MIX_AIC_1_2, BASE, BL1,  BASE, PARALLEL,    K_NOT_SHIFT   │   │
│  │ 8. MIX_AIC_1_2, BASE, BL1,  ALIGNOUT, TRUE,    K_NOT_SHIFT   │   │
│  │ 9. MIX_AIC_1_2, BASE, BL1,  ALIGNOUT, FALSE,   K_NOT_SHIFT   │   │
│  │10. MIX_AIC_1_2, BASE, BL1,  VEC_NZ2ND, FALSE,  K_NOT_SHIFT   │   │
│  │11-15. SC/SC_GM2L1/NKM  ...  (各 MIXND2NZ 变体)               │   │
│  │16. MIX_AIC_1_0, BASE, MC,   BASE, FALSE,       K_NOT_SHIFT   │   │
│  │17-20. DET  ...  (MIXND2NZ/FIXOPTI 变体)                       │   │
│  │21. MIX_AIC_1_2, BASE, BASE, BASE, FALSE, K_NOT_SHIFT, FP32EN │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  #elif NZ×NZ 分支 (仅 ops-nn):                                      │
│    1. AIC_ONLY, BASE, BASE, BASE, FALSE, K_SHIFT/K_NOT_SHIFT ★     │
│    2. MIX_AIC_1_2, BASE, BASE, VEC_NZ2ND, FALSE, K_NOT_SHIFT       │
│                                                                     │
│  GemmV3 (仅 ops-nn): gemm_v3.cpp 中同样含 K_SHIFT 分支              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─ Kernel 执行 (Device, AI Core) ─────────────────────────────────────┐
│                                                                     │
│  template <int LOADMODE, int SPLITCOREMODE, int FIXOPTI,            │
│            int MIXND2NZ, int SPECIALOPT, int FP32ADDMM>             │
│  __global__ void mat_mul_v3(GM_ADDR aGM, bGM, biasGM,               │
│                              offsetWGM, cGM, workspaceGM, tilingGM) │
│  {                                                                  │
│      GET_TILING_DATA(tilingData, tilingGM);                         │
│                                                                     │
│      // ★ 第 1 优先级: K_SHIFT                                      │
│      if constexpr (BASE && BASE && BASE && FALSE && K_SHIFT) {      │
│          MMV3_IMPL_CLASS(MatmulBaseKernel, format_x1,               │
│                          MatmulBaseBlock, MM_CFG_K_SHIFT);          │
│                                                                     │
│      // 第 2 优先级: 未对齐场景                                      │
│      } else if constexpr (BASE && BASE && BASE && TRUE) {           │
│          MMV3_IMPL_CLASS(MatmulBaseUnAlignedKernel, ...);           │
│                                                                     │
│      // 第 3 优先级: Single Core Split K                            │
│      } else if constexpr (BASE && SC && BASE && FALSE) {            │
│          MMV3_IMPL_CLASS(MatMulSingleCoreSplitKKernel, ...,         │
│                          MM_CFG_PRELOAD_MK);                        │
│                                                                     │
│      // ... NKM, GM2L1, MC, DET 各自独立 kernel 类 ...              │
│                                                                     │
│      // fallback                                                    │
│      } else {                                                       │
│          MMV3_IMPL_CLASS(MatmulBaseKernel, ..., MM_CFG_NO_PRELOAD); │
│      }                                                              │
│  }                                                                  │
│                                                                     │
│  MMV3_IMPL_CLASS 内部:                                              │
│    → 根据 tilingData.matmulRunInfo.transA/transB 展开 4 分支         │
│    → MatmulBaseKernel.Init(aGM,bGM,cGM,...,&tilingData,&pipe)       │
│      → BaseBlock.Init()  读取 tilingData 初始化 block 参数          │
│      → BaseBlock.InitBlockIndex()  分配 M×N basic block 到 core     │
│    → MatmulBaseKernel.Process()                                     │
│      → for each L2 tile (mTile, nTile):                            │
│          for each round:                                            │
│            BaseBlock.UpdateBasicIndex()                             │
│            BaseBlock.UpdateBlockParams()                            │
│            BaseBlock.CalcGMOffset()  // 计算 A/B/C GM 地址偏移 (与K方向无关)│
│            MatmulImpl<...,MM_CFG_K_SHIFT>.IterateAllBlocks()        │
│              ├─ MTE2: 搬运 A/B tile 从 GM→L1→L0A/L0B               │
│              │   K 起始偏移 = core_id × ceil(K/singleCoreK) / nCores│
│              ├─ Cube MMAD: L0A × L0B → 累加到 L0C                  │
│              └─ MTE3: L0C → GM (写回 C 矩阵)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. stepK 调整与 K_SHIFT 模板选择（两个独立决策）

> **机制更正（对照 ops-nn 源码核验）**：早期版本把 `OptimizeBasicKernelStepK` 当作"K_SHIFT 触发器"是**不准确的**。源码中该函数**只调整 stepKa/stepKb 与 depthA1/depthB1，从不写 `tilingEnableSpecialOpti`**。K_SHIFT 模板（`MM_CFG_K_SHIFT`）是否启用，由 tilingKey 的 SPECIALOPT 位决定，而该位**唯一**写入点是 AOE 调优路径 `CheckAoeTilingEnable`（`matmul_v3_base_tiling.cpp:181`，从 `aoeTilingEnable` 万位解析）。二者是**独立决策**：AOE 决定模板，`OptimizeBasicKernelStepK` 决定 L1 的 stepK 粒度。下文分开陈述。

### 3.1 stepK 调整：`OptimizeBasicKernelStepK()`

`matmul_v3_base_tiling.cpp:807`，`DoOpTiling()` 末尾调用。命中条件后只改 stepK/depth，**不改 tilingKey、不改模板选择**。

```
                     OptimizeBasicKernelStepK()
                                │
       ┌────────────────────────┼──────────────────────────┐
       │ 前置门 (10 条，ALL 必须满足)                         │
       ├───────────────────────────────────────────────────┤
       │ ① FullLoad=BASE && SplitCore=BASE && FixOpti=BASE │
       │ ② GetMixNd2nzType() == NO_ND2NZ  (ops-nn有此行)   │
       │ ③ baseM×baseN == 32768  (128×256, L0C 最优)      │
       │ ④ baseK == 64                                     │
       │ ⑤ M,N,K 各自 %256==0, ≥768                        │
       │ ⑥ K 不是 2 的幂次方 (!IsPowerOfTwo)                │
       │ ⑦ M×N > 128 × 256 × aicNum  (超出单核单轮)       │
       │ ⑧ aicNum == 24                                    │
       │ ⑨ M%16384≠0 && N%16384≠0  (非mata)               │
       │ ⑩ aType,bType,cType ∈ {FP16, BF16}                │
       └──────────┬────────────────────────────────────────┘
                  │ 全部 YES
       ┌──────────▼────────────────────────────────────────┐
       │ 形状区间 (任一满足即可)                               │
       ├───────────────────────────────────────────────────┤
       │ [标准]   M,N,K ≥768, 全 256 对齐, K !power2       │
       │ [mid-M]  M∈[10368,18000], N,K∈[1280,5120]        │
       │ [mid-K]  M,N∈[1280,5120], K∈[13788,19304]        │
       │ [big-M]  M∈[320000,380000], N∈[960,3500],         │
       │          K∈[1280,3328]                             │
       └──────────┬────────────────────────────────────────┘
                  │ 命中
                  ▼
       ┌─────────────────────────────────┐
       │ 唯一作用 (源码确认):             │
       │  stepKa/stepKb: 8 → 4           │
       │  depthA1/depthB1: /8*4 减半      │
       │ ★ 不写 tilingEnableSpecialOpti  │
       │ ★ 不改 tilingKey 模板位         │
       └─────────────────────────────────┘
```

注：函数体开头还会用 `GET_TPL_TILING_KEY(..., SPECIALOPT=BASE, ...)` 重算一次 `tilingKey_`（仅用于 plog 打印 fit/doesn't-fit），随后被 `DoTilingKey()` 覆盖，不影响最终 key。

> **易混点澄清（stepK=4 ≠ K轴错峰）**：`OptimizeBasicKernelStepK` 的 stepK 8→4 只是 L1 K-窗口减半 / partial-sum 边界增多，**不是错峰**。真正的"K 轴错峰"是 K_SHIFT 模板（`MM_CFG_K_SHIFT` 的 `enPeakStagger/enKShift=true`，各核不同 K 起点环形搬运）。**知识库未命中(repo MISS)时，启发式只给 stepK=4、不给错峰** —— `tilingEnableSpecialOpti` 恒 BASE → SPECIALOPT=0 → kernel 走 `MM_CFG_NO_PRELOAD`（enPeakStagger/enKShift 均 false），各核 K 顺序一致、是确定性的。错峰**只在 repo 命中且万位=1** 时才有（见 §3.2）。二者在 repo 命中时常常一起出现，故易被误认为"stepK=4 就是错峰"。

### 3.2 K_SHIFT 模板选择：`CheckAoeTilingEnable`（AOE 离线调优）

K_SHIFT kernel 模板（`MM_CFG_K_SHIFT`，enKShift/enPeakStagger=true）是否编译进图，**仅**由 tilingKey 的 SPECIALOPT 位是否为 `K_SHIFT(1)` 决定。`tilingEnableSpecialOpti` 在 ops-nn 中**只有一处赋值**：

```cpp
// matmul_v3_base_tiling.cpp:176-181  (CheckAoeTilingEnable 内)
uint32_t tilingSpecOpti = (aoeTilingEnable / 10000U) % 10U; // aoe 万位
if (tilingSpecOpti > ENABLE_K_SHIFT) { ... return false; }
tilingEnable_.tilingEnableSpecialOpti = static_cast<TilingEnableSpecialOpti>(tilingSpecOpti);
```

`aoeTilingEnable` 是 AOE（Ascend Optimization Engine）**离线调优**产物的外部配置，调优器对某 shape 判定 K_SHIFT 最优时把万位置 1 烘焙进 compile info。即"按 shape 触发 K_SHIFT"的真实路径是 **AOE 离线调优 → 万位=1**，不是运行时 `OptimizeBasicKernelStepK`。运行时的 stepK=4（§3.1）是与之配套的 L1 粒度调整，二者解耦：可单有其一。

**关于"特定 shape 才进入"（知识库/tiling 仓库）**：`aoeTilingEnable`（连同 stepKa/stepKb、depthA1/B1、baseM/N/K、usedCoreNum 等整套 tiling）并非运行时算出，而是从 **AOE 调优 tiling 仓库**按 shape 签名取出的预调结果。ops-nn 的 `TranslateAoeTiling`（`matmul_v3_base_tiling.cpp:1057`，日志打印 "Get tiling from repo"）按输入签名（m/n/k/trans/dtype/batch/对齐标志等，见 `:1046-1054`）查 repo，命中则**整套**载入（含 `aoeTiling->tilingEnable`、`aoeTiling->stepKa/stepKb`），再由 `CheckAoeTilingEnable`（`:1096`）解析万位。因此：

- **repo 命中**：tilingEnable 万位=1 的 shape 才进 K_SHIFT 模板；stepK 也直接来自 repo。
- **repo 未命中**：走运行时启发式 tiling，`OptimizeBasicKernelStepK` 按其硬编码 shape 区间（§3.1）决定是否把 stepK 调成 4，但 `tilingEnableSpecialOpti` 仍为 BASE（启发式**不**开 K_SHIFT 模板）。

一句话：**K_SHIFT 模板只对"进过 AOE 知识库且被标 1"的特定 shape 生效**；`OptimizeBasicKernelStepK` 的硬编码区间是 repo 缺失时的 stepK 启发式近似，与模板开关无关。

**唯一性结论（全 ops-nn matmul 树核验）**：`tilingEnableSpecialOpti` 在整个仓里**只有一处赋值**——`matmul_v3_base_tiling.cpp:181`（`CheckAoeTilingEnable` 内）。无环境变量、无编译宏、无运行时启发式会把它置为 `ENABLE_K_SHIFT`；`OptimizeBasicKernelStepK` 也不写它。故 **AOE 知识库是 K_SHIFT 模板的唯一启用通路**，文档无需再列其它"调用到"的路径。GemmV3 复用 `MatmulV3BaseTiling`（`gemmv3_tiling.cpp:49` `REGISTER_TILING_TEMPLATE`），其 K_SHIFT 走同一通路，非独立入口。（另有 `transpose_batch_mat_mul` 这个**兄弟算子**带 `CLOSE_MATMUL_K_SHIFT` 环境变量/`GetDeterministic()` 标志，但那是另一个算子且只用于**关闭** K_SHIFT，不在本文档范围。UT `test_matmul_v3_tiling.cpp:1627` 直接构造 K_SHIFT key 属测试桩，非运行时通路。）

kernel 侧模板匹配（`mat_mul_v3.cpp`）要求 `MIXND2NZ==FALSE && SPECIALOPT==K_SHIFT` 才走 `MM_CFG_K_SHIFT`：

```cpp
// ops-nn mat_mul_v3.cpp  (ND 默认分支与 NZ×NZ 分支各一处)
if constexpr (... && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE &&
              SPECIALOPT == MAT_MUL_V3_K_SHIFT) {
    MMV3_IMPL_CLASS(MatmulBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_K_SHIFT);
}
```

### 代码 (ops-nn，`OptimizeBasicKernelStepK` 真实逻辑)

```cpp
void MatmulV3BaseTiling::OptimizeBasicKernelStepK()   // matmul_v3_base_tiling.cpp:807
{
    constexpr uint64_t baseMNCheck = 32768;  // 128×256
    constexpr uint64_t baseKCheck = 64;
    constexpr uint64_t alignCheck = 256;
    constexpr uint64_t MNCheck = 768;
    constexpr uint64_t mataCheck = 16384;
    constexpr uint64_t aicNumCheck = 24;

    bool baseMNKFlag = runInfo_.baseM * runInfo_.baseN == baseMNCheck && runInfo_.baseK == baseKCheck;
    bool alignFlag = m%256==0 && n%256==0 && k%256==0 && m>=768 && n>=768 && !IsPowerOfTwo(k);
    bool middleMShapeFlag = m>=10368 && m<=18000 && n>=1280 && n<=5120 && k>=1280 && k<=5120;
    bool middleKShapeFlag = m>=1280 && m<=5120 && n>=1280 && n<=5120 && k>=13788 && k<=19304;
    bool BigMShapeFlag = m>=320000 && m<=380000 && n>=960 && n<=3500 && k>=1280 && k<=3328;
    bool globalMNFlag = m*n > baseMNCheck * compileInfo_.aicNum;
    bool aicNumCheckFlag = compileInfo_.aicNum == aicNumCheck;
    bool notMataFlag = m%mataCheck!=0 && n%mataCheck!=0;
    bool dtypeFlag = (aType==FP16||aType==BF16) && (bType同) && (cType同);

    // 仅用于日志打印 fit/doesn't-fit；tilingKey 随后被 DoTilingKey() 覆盖
    tilingKey_ = GET_TPL_TILING_KEY(FullLoad, SplitCore, FixOpti, GetMixNd2nzType(),
                                    TilingEnableSpecialOpti::BASE, TilingEnableFp32Addmm::FALSE);
    if (FullLoad==BASE && SplitCore==BASE && FixOpti==BASE
        && GetMixNd2nzType()==NO_ND2NZ       // ★ ops-nn 多了这行 (omni 无)
        && baseMNKFlag && notMataFlag && dtypeFlag
        && (alignFlag || middleM || middleK || bigM)
        && globalMNFlag && aicNumCheckFlag)
    {
        constexpr uint64_t oriStepKValue = 8;
        constexpr uint64_t optStepKValue = 4;
        if (stepKa == oriStepKValue) { depthA1 = depthA1/stepKa*optStepKValue; stepKa = optStepKValue; }
        if (stepKb == oriStepKValue) { depthB1 = depthB1/stepKb*optStepKValue; stepKb = optStepKValue; }
        // ★ 注意: 原代码此处【没有】tilingEnableSpecialOpti = ENABLE_K_SHIFT 这一行
        //         (旧版文档此处为杜撰, 已删)。模板选择只由 AOE 万位决定, 见 §3.2
    }
}
```

### 3.3 错峰生效时的配套参数调整：原因与作用

> 本节回答：**当 K 轴错峰（K_SHIFT 模板）生效时，为什么 stepK/depth 必须跟着调？调了什么、起了什么作用？**
> 前提：错峰只在 repo 命中且 `tilingEnable` 万位=1 时生效（§3.2）。本节描述"错峰已开"之后的配套 L1 参数。

#### 3.3.1 错峰改变了什么（起因）

先厘清"每个核处理什么"——分两层：

- **M/N 输出分工**：不同核算 C 的不同 M行×N列 block（core0→rows 0-127，core1→128-255…），**永远不同**（不管错峰与否）。
- **K 遍历**：每核都遍历**完整 K**（0→K-1）累加自己的输出块。各核读的 B 矩阵 K 列段是**共享**的（同 N-block 的核共用 `B[:,nblock]`），A 的 K 列段 K 范围也相同。

所以"各核读同一套 K 列"恰恰是 **BASE 的问题**：所有核 lockstep 遍历 K → 同一时刻都在读 K=[0,63] → 命中**同一组 L2 bank** → bank 仲裁排队 → 有效带宽暴跌（§1.1）。

错峰（`MM_CFG_K_SHIFT`：`enKShift=true` + `enPeakStagger=true`）**不改各核算什么**，只改"各核何时读哪段 K"——让 K 起点错开：

```
  K_start(core_i) = i × ceil(K / usedCoreNum)        ← enKShift: 各核 K 起点错开
  每核仍遍历完整 K, 到末尾回 0 环形迭代 (数学等价, 只是顺序旋转)
  enPeakStagger: 让多核的 GM→L1(MTE2) 搬运在时间轴上错峰 → 命中不同 L2 bank
```

关键变化：**任一时刻，不同核处在 K 的不同段**（核 0 在 K=[0,256)、核 6 在 K=[192,448)…），从而命中不同 L2 bank。这是带宽收益的来源。

#### 3.3.2 调了哪些参数（源码确认）

先点明 stepKa/depthA1 的精确定义（源码 `CalL1Tiling` `:638-662`，`DB_SIZE=2` 双缓冲）：

- `depthA1` = L1 A-buffer **总共**装的 baseK 个数 = `L1/2 ÷ baseM ÷ baseK ÷ dtype`。
- `stepKa = depthA1 / DB_SIZE = depthA1 / 2` = **单个双缓冲槽**装的 baseK 个数（L1 分 2 槽，Cube 算槽0 时 MTE2 载槽1）。
- 即"一次载入"分两层：**单槽** = `stepKa` 个 baseK；**L1 总量** = `depthA1 = stepKa×2` 个 baseK。

错峰生效时（repo 命中），AOE 调优产物的整套 tiling 里，L1 相关参数取**小窗口**配置（`OptimizeBasicKernelStepK` `:862-864` 在 repo 缺失时对同样 shape 给一致启发式）：

| 参数 | 不错峰(BASE) | 错峰(K_SHIFT) | 含义 |
|------|-------------|---------------|------|
| `stepKa` / `stepKb` | 8 | **4** | **单缓冲槽**的 baseK 个数（`depthA1/2`）|
| `depthA1` / `depthB1` | 16(=8×2) | **8(=4×2)** | L1 总 baseK 数（`stepKa×DB_SIZE`，同步减半）|
| `baseM×baseN` | — | 32768(128×256) | L0C 最优分块（错峰前置门，不变只校验）|
| `baseK` | — | 64 | 单 K tile 宽（不变只校验）|

即：错峰把**单槽 baseK 数 8→4**、**L1 总 K 数据量减半**（单槽 `128×512`→`128×256`；L1 总量 `128×1024`→`128×512`），关系 `depthA1 = stepKa × DB_SIZE` 始终保持。

#### 3.3.3 为什么必须调（原因链）

> 依据 Ascend Cube 硬件模型（L2 bank/L1 容量/MTE2 流水，见 §1、§8.3）+ 源码前置门推导。**仓库源码无注释解释 L1 微架构原因，本节为从错峰目标反推的最合理解释。**

两条主因（直接→预算）：

```
①【直接】错峰靠"各核 K 段在时间上错开"分散 L2 请求。stepK 越小(=4) → L1→L0A 的 K 推进越细
   → MTE2 预取的 K-chunk 越小、越频繁 → 错峰能分散的请求粒度越细、在途请求越多
   → L2 bank 命中越均匀 (错峰带宽收益的必要条件)。stepK=8 chunk 太粗, 错峰分散不开。

②【预算】L1 容量固定, 须同时容纳 A 的 K-窗口 + B 的 K-窗口 + MTE2 双缓冲预取流水。
   depthA1/B1 减半使 L1 K-窗口与 stepK=4 匹配, 并给 MTE2/Cube 重叠流水留出预算。
```

一句话：**错峰要靠细粒度的 K 预取才能把 L2 请求分散开（主因），stepK 8→4 + depth 减半既是"分散粒度"所需，也匹配 L1/流水的容量预算。** 这正是 `OptimizeBasicKernelStepK` 前置门③④（`baseM×baseN==32768 && baseK==64`，L0C 最优块）与 shape 区间存在的意义：只在 L0C/L1 配置刚好适配错峰小窗口的 shape 上才启用。

> 说明：双缓冲流水（MTE2 搬运 与 Cube 计算 重叠，隐藏 GM→L1 延迟）是**所有高性能 matmul 的标配**（`doMTE2Preload` 等），与错峰无关；错峰（`enPeakStagger`）是在双缓冲基础上把**多核的 MTE2 预取时机**在时间轴上错开。

#### 3.3.4 作用（性能 vs 精度，双刃剑）

| 维度 | 作用 | 机制 |
|------|------|------|
| **性能（动机）** | L2 有效带宽 +15~30% | 各核 K 段错开 → 命中不同 L2 bank → 仲裁排队减少（§1.2）|
| **精度/一致性（代价）** | 累加顺序敏感度↑ → batch 间 ≤2 ULP 差异 | stepK=4 → L0C partial-sum 回写边界从 2 个增至 3 个（K=12 tile 时），每边界一次 FP 舍入；叠加错峰的 per-core 不同 K 顺序 → 不同累加序（§8.3/§8.5）|

注：partial-sum 边界增多本身（stepK=4）在**不错峰**时各核边界位置相同、仍确定（§3.1 易混点）；非确定性是"边界增多 × 各核 K 顺序不同"两者叠加的结果。`batch_invariant=True`（PR 8a8172a）通过强制 `SPECIALOPT=BASE` 关掉错峰，从根上消除 per-core K 顺序差异，恢复确定性（此时 stepK 仍可能=4，但各核顺序一致 → 一致）。

#### 3.3.5 参数与错峰的关系总览

```
                   repo 命中 (万位=1)
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
   SPECIALOPT=K_SHIFT            stepK=4, depthA1/B1 减半
   (enPeakStagger/enKShift)      (L1 小窗口, 腾空间给错峰流水)
           │                           │
           └──────────┬────────────────┘
                      ▼
         kernel = MatmulBaseKernel(MM_CFG_K_SHIFT)
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
  性能: L2 错峰 +15~30%        精度: per-core K 序不同
                                  + partial-sum 边界多
                                  → batch 间 ≤2 ULP 差异
```

---

### 与 omni-ops 的差异

omni-ops `OptimizeBasicKernelStepK()`（`ai_infra_matmul_base_tiling.cpp`，与 ops-nn 几乎逐行一致）缺少第 ② 条 `GetMixNd2nzType()==NO_ND2NZ` 检查，且 `DoTilingKey()` 中 MIXND2NZ 硬编码为 `NO_ND2NZ`（ops-nn 是 `GetMixNd2nzType()` 动态获取）。

**更关键的差异（模板选择）**：omni-ops `CheckAoeTilingEnable` 仅在 `ai_infra_matmul_base_tiling.h:108` 有声明，**全仓无定义、无调用**；因此 `tilingEnableSpecialOpti` 在 omni 中恒为默认 `BASE`，**从不被置为 `ENABLE_K_SHIFT`**。后果：omni kernel `ai_infra_matmul.cpp:195/204` 的 `SPECIALOPT==K_SHIFT → MM_CFG_K_SHIFT` 分支是**不可达死代码**；即 omni 当前根本不会启用 K_SHIFT 模板（详见 §7）。

---

## 4. TilingKey 位域编码

```
GET_TPL_TILING_KEY(LOADMODE, SPLITCOREMODE, FIXOPTI, MIXND2NZ, SPECIALOPT, FP32ADDMM)
                      4bit        8bit       4bit     4bit       4bit        4bit
                      [31:28]    [27:20]    [19:16]  [15:12]    [11:8]      [7:4]
   (字段位宽经 mat_mul_v3_tiling_key.h 的 ASCENDC_TPL_*_BW 宏核验: 4/8/4/4/4/4)

K_SHIFT 生效 key (ND 默认分支, kernel 要求 MIXND2NZ==FALSE):
  LOADMODE=0 | SPLITCORE=0 | FIXOPTI=0 | MIXND2NZ=FALSE(1) | SPECIALOPT=K_SHIFT(1) | FP32ADDMM=0
  → MIXND2NZ=1 落 [15:12]=0x1000, SPECIALOPT=1 落 [11:8]=0x0100, 合计 low-byte 段 0x1100
   (宏值: MIXND2NZ_TRUE=0/FALSE=1/TRUE_PARALLEL=2; K_NOT_SHIFT=0/K_SHIFT=1 —— 见 tiling_key.h)
```

### DoTilingKey 对比

```cpp
// ops-nn (matmul_v3_base_tiling.cpp:2795)
void MatmulV3BaseTiling::DoTilingKey() {
    uint64_t mix = static_cast<uint64_t>(GetMixNd2nzType());  // 0/1/2 动态
    tilingKey_ = GET_TPL_TILING_KEY(FullLoad, SplitCore, FixOpti,
                                    mix,                       // ★
                                    tilingEnableSpecialOpti, FP32Addmm);
}

// omni-ops (ai_infra_matmul_base_tiling.cpp:1973)
void AiInfraMatmulBaseTiling::DoTilingKey() {
    uint64_t mix = static_cast<uint64_t>(MixNd2NzType::NO_ND2NZ);  // 恒为 1
    tilingKey_ = GET_TPL_TILING_KEY(FullLoad, SplitCore, FixOpti,
                                    mix,                            // ★
                                    tilingEnableSpecialOpti, FP32Addmm);
}
```

**影响**: omni-ops 的 `MIXND2NZ_TRUE(0)` 和 `MIXND2NZ_TRUE_PARALLEL(2)` 对应的编译模板
永久不可达。

---

## 5. MM_CFG_K_SHIFT 配置

```cpp
// ops-nn: mat_mul_v3_common.h:74-76
// omni-ops: ai_infra_matmul_common.h:73  (参数完全一致, 逐字相同)

// enable K shift
constexpr MatmulConfig MM_CFG_K_SHIFT =
    GetMDLConfig(false, false, 0, false, false, false, true, true, true, false,
                 false, true);

// 对照同文件其它常量反推形参位置 (GetMDLConfig 形参名在 CANN SDK 头, 不在本仓):
//   MM_CFG_VEC_ND2NZ = GetMDLConfig(false, false, 0, true)   // 注释 "set isVecND2Nz"
//     → 第 4 形参 = isVecND2Nz (★ 旧文档误标在第 1 形参, 已纠正)
//   MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, true)  // "set enUnitFlag"
//     → 第 7 形参 = enUnitFlag
//   MM_CFG_K_SHIFT 在第 7/8/12 形参为 true:
//     [7] enUnitFlag=true, [8] enPeakStagger=true (核间搬运错峰),
//     [12] enKShift=true   (K轴偏移)  —— 源文件顶部注释 "enable K shift" 即指此

// 对比默认配置
constexpr MatmulConfig MM_CFG_NO_PRELOAD =
    GetMDLConfig(false, false, 0, false, false, false, true);
    // 仅 enUnitFlag=true, 无 enPeakStagger, 无 enKShift
```

---

## 6. Kernel 模板分发树

ops-nn `mat_mul_v3.cpp` — 20+ 分支，omni-ops 仅 6（差异标注 ▼）:

```
TilingKey 匹配顺序:

K_SHIFT     → MatmulBaseKernel                     (MM_CFG_K_SHIFT)
未对齐       → MatmulBaseUnAlignedKernel             (MM_CFG_NO_PRELOAD)   ▼ omni: 合并到 else
SC_SPLIT_K  → MatMulSingleCoreSplitKKernel          (MM_CFG_PRELOAD_MK)   ▼ omni: 合并到 else
NKM_SPLIT_K → MatMulSingleCoreSplitKKernel          (MM_CFG_PRELOAD_NK)   ▼ omni: 合并到 else
GM2L1       → MatMulSingleCoreSplitKKernelGmToL1    (MM_CFG_PRELOAD_MK)   ▼ omni: 合并到 else
MC_SPLIT_K  → MatMulMultiCoreSplitK                 (BASE)                ▼ omni: 合并到 else
DET_SPLIT_K → MatMulKernelDeterministicSplitK       (BASE)
  DET+TRUE   → MatMulUnAlignedKernelDetSplitK       (BASE)
  DET+VEC    → MatMulKernelDeterministicSplitK       (VEC_NZ2ND)
AL1_FULLLOAD→ MatmulBaseKernelAL1FullLoad           (MM_CFG_MDL)
BL1_FULLLOAD→ MatmulBaseKernelBL1FullLoad           (MM_CFG_NO_PRELOAD)
BL1+ALIGNOUT→ MatmulBaseUnalignedNKernel            (fixpipe)
BL1+PARALLEL→ MatmulCvpBaseKernel                   (parallel nd2nz)
BL1+ATONZ   → MatmulBaseAToNZWithBL1FixpipeKernel   (A→NZ fixpipe)

fallback     → MatmulBaseKernel                     (MM_CFG_NO_PRELOAD)
```

> 辅仓 omni-ops: SC/NKM/GM2L1/MC/未对齐 全部走 `else → MatmulBaseKernel`，
> 损失了针对各 SplitCore 模式的专属优化。

---

## 7. ops-nn vs omni-ops 差异总结

> 核心结论先行：**omni-ops 当前根本不会启用 K_SHIFT**。`tilingEnableSpecialOpti` 在 omni 中没有任何赋值路径（恒为 `BASE`），因此 kernel 里所有 `SPECIALOPT==K_SHIFT → MM_CFG_K_SHIFT` 分支都是**不可达死代码**。下表为逐项对照。

| 对照项 | ops-nn | omni-ops | 影响 |
|--------|--------|----------|------|
| AOE 调优仓库入口 `TranslateAoeTiling`（按 shape 取离线调优 tiling） | 有（`matmul_v3_base_tiling.cpp:1057`，注释 "Get tiling from repo"） | **无** | omni 无"知识库"按 shape 取预调 tiling |
| `CheckAoeTilingEnable` 万位=SPECIALOPT → 置 `ENABLE_K_SHIFT` | 有定义有调用（`:147`/`:1096`） | 仅 `.h:108` 声明，**无定义无调用** | omni `tilingEnableSpecialOpti` 永为 BASE，K_SHIFT 模板不可达 |
| kernel `SPECIALOPT==K_SHIFT → MM_CFG_K_SHIFT` 分支 | 可达（AOE 万位=1 时） | **死代码**（`ai_infra_matmul.cpp:195/204`） | omni 实际从不走 K_SHIFT |
| NZ×NZ format K_SHIFT 模板 | 有 | 有分支但不可达 | WeightNZ+WeightNZ 无错峰 |
| GemmV3 K_SHIFT 分支 | 有（`gemm_v3.cpp:303`） | omni 无独立 GemmV3 | FP32 GEMM 无错峰 |
| `GetMixNd2nzType()` 动态 → DoTilingKey | 动态 0/1/2 | 硬编码 `NO_ND2NZ`(=1) | omni 的 MIXND2NZ_TRUE(0)/PARALLEL(2) 模板永不可达 |
| `OptimizeBasicKernelStepK` 中 `GetMixNd2nzType()==NO_ND2NZ` 门 | 有 | **无**（条件更宽松） | omni 更多 shape 会触发 stepK=4 |
| SC/DET/MC/NKM/GM2L1 独立 kernel 类 | 有 | 合并到 `else` | omni Split-K 无针对性调度 |
| arch35/A5 高级 tiling | —— | omni-ops 本身即 arch35 衍生 | omni 面向 A5(Ascend950) |

**对 batch 一致性的含义**：由于 omni 从不启用 K_SHIFT 模板（无 per-core K 偏移），omni 的 `AiInfraMatmul` 在 K_SHIFT 维度上**当前是确定性的**。`OptimizeBasicKernelStepK` 的 stepK=4 在 omni 仍会发生，但单有 stepK=4（无 K_SHIFT 模板）时各核 K 迭代顺序一致，不引入跨核差异（详见 §8）。PR `8a8172a` 的 `batch_invariant` 覆盖（`DoTilingKey` 强制 `SpecialOpti=BASE`）在当前 omni 是**防御性 no-op**，为将来若引入 AOE/K_SHIFT 路径时预留开关。

---

## 8. K_SHIFT 与 batch 一致性 — 逐层图解

### 8.1 极简例子: M=6, K=12, N=6, 3核

`singleCoreM=2, singleCoreN=3`。每个核的 A 切片是 **2行×全12列**，固定不变。

```
A 矩阵 (6×12):  每个核读自己那2行, 全部K列              B 矩阵 (12×6): 按N列分
  K= 0  1  2  3  4  5  6  7  8  9 10 11               K= 0..5  6..11
M0 [████████████████████████████]  ← Core0处理         0 [██████|██████]
M1 [████████████████████████████]                      1 [██████|██████]
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                 ...
M2 [████████████████████████████]  ← Core1处理        11 [██████|██████]
M3 [████████████████████████████]                         N0-2列  N3-5列
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                  ↑ Core0/2共享  ↑ Core1共享
M4 [████████████████████████████]  ← Core2处理
M5 [████████████████████████████]
```

```
C 矩阵 (6×6): 切成 3×2=6 块, round=ceil(6/3)=2 轮

    N=0..2  N=3..5
M0 ┌───────┬───────┐
M1 │ C[0]  │ C[1]  │       第1轮:              第2轮:
  ─├───────┼───────┤        Core0 → C[0]        Core0 → C[3]
M2 │ C[2]  │ C[3]  │        Core1 → C[1]        Core1 → C[4]
M3 │       │       │        Core2 → C[2]        Core2 → C[5]
  ─├───────┼───────┤
M4 │ C[4]  │ C[5]  │   每个核的 A 切片在两轮中不变:
M5 │       │       │     Core0 始终读 A[0:2, 0:12]
  ─└───────┴───────┘     Core1 始终读 A[2:4, 0:12]
                         Core2 始终读 A[4:6, 0:12]

  变化的是 B 切片和 C 目标位置:
    第1轮 Core1 读 B[:, 3:6] → 写 C[2:4, 3:6]
    第2轮 Core1 读 B[:, 0:3] → 写 C[4:6, 0:3]
```

> **核心**: A 切片在 M/K 两维都固定; 轮次只改变 B 切片(N列)和 C 目标块。

### 8.2 真实场景: M=768, K=768, N=768, 24核

Tiling: `singleCoreM=128, singleCoreN=256, singleCoreK=768`

```
C (768×768) = 6行块 × 3列块 = 18 blocks, round=1

       N:  0..255   256..511   512..767
  M:    ┌─────────┬─────────┬─────────┐
   0    │ Core  0 │ Core  6 │ Core 12 │   rows    0..127
  128   │ Core  1 │ Core  7 │ Core 13 │   rows  128..255
  256   │ Core  2 │ Core  8 │ Core 14 │   rows  256..383
  384   │ Core  3 │ Core  9 │ Core 15 │   rows  384..511
  512   │ Core  4 │ Core 10 │ Core 16 │   rows  512..639
  640   │ Core  5 │ Core 11 │ Core 17 │   rows  640..767
        └─────────┴─────────┴─────────┘
  Core 18-23 idle
```

```
A (768×768): 按行分, 每个核固定读 128行 × 全768列

  K=0 ─────────────────────────── 767
  ┌─────────────────────────────────┐  rows   0..127 → Core  0,6,12
  ├─────────────────────────────────┤  rows 128..255 → Core  1,7,13
  ├─────────────────────────────────┤  rows 256..383 → Core  2,8,14
  ├─────────────────────────────────┤  rows 384..511 → Core  3,9,15
  ├─────────────────────────────────┤  rows 512..639 → Core  4,10,16
  ├─────────────────────────────────┤  rows 640..767 → Core  5,11,17
  └─────────────────────────────────┘

  同一段行被多个核共享 (不同 N 列块): Core 0,6,12 都读 rows 0..127

B (768×768): 按列分, 多个核共享同一列块

  K=0 ─────────────────────────── 767
  ┌──────────┬──────────┬──────────┐
  │ N 0..255 │N 256..511│N 512..767│  ← 三列, 各 768行×256列
  └──────────┴──────────┴──────────┘
    ↑ Core0-5  ↑ Core6-11 ↑ Core12-17
```

```
单核工作示例 — Core 1 (rows 128..255):

  offsetA = 1 × 128 × 768 = 98304  → &A[128, 0]
  offsetB = 0 × 256 = 0            → &B[0, 0]
  offsetC = 0 + 1×128×768 = 98304  → &C[128, 0]

  mm_.SetTensorA(A[98304])   ← 128行×768列
  mm_.SetTensorB(B[0])       ← 768行×256列
  mm_.Iterate()              ← K方向分12个tile(baseK=64), 全量迭代
```

### 8.3 K_SHIFT 改变: K tile 的迭代顺序

K_SHIFT 不改变 A/B/C 的分配。改变的是 `mm_.Iterate()` 内部 K tile 的迭代**起点**。

```
K=768, baseK=64 → 12 个 tile:  [t0][t1][t2][t3][t4][t5][t6][t7][t8][t9][t10][t11]
                                K:0-63                              K:704-767

无 K_SHIFT — 所有核相同:
  tile 顺序: t0 → t1 → t2 → ... → t11
  Core0 和 Core6 读 tile 顺序完全一样

有 K_SHIFT — 每核从不同 tile 开始 (环形):
  K_start(core) = core_id × ceil(768/24) = core_id × 32 字节

  Core  0: K_start=0   → [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]
  Core  1: K_start=32  → [ ─t0─后半, t1..t10, t11, ─t0─前半 ]  ← 第一个tile跨边界
  Core  6: K_start=192 → [t3, t4, t5, t6, t7, t8, t9, t10, t11, t0, t1, t2]
  Core 12: K_start=384 → [t6, t7, t8, t9, t10, t11, t0, t1, t2, t3, t4, t5]

关键: 各核在 t=0 时刻从不同 K 偏移读 L2:
  Core 0: K=[0..63]     → GM 地址低段
  Core 6: K=[192..255]  → GM 地址中段  } 命中不同 L2 bank → 带宽提升
  Core12: K=[384..447]  → GM 地址高段
```

```
stepK 的变化 (K_SHIFT 同时改了):
                BASE (stepKa=8)              K_SHIFT (stepKa=4)
  L1窗口:   [A: 128×512] 一次预取         [A: 128×256] 一次预取
  L1→L0A:   分 8 次, 每次 128×64          分 4 次, 每次 128×64
  partial:   tile 0-7 批 + tile 8-11 批    tile 0-3 + 4-7 + 8-11
             ↑ 2 次 L0C 累加边界           ↑ 3 次 L0C 累加边界 → 更多 FP 舍入
```

### 8.4 M 改变: 同一行是否换核

```                                                第2轮:
M=768 (18 blocks, round=1):              M=1536 (36 blocks, round=2):

            N0  N1  N2                    第1轮 N0  N1  N2        第2轮 N0  N1  N2
            ┌───┬───┬───┐              ┌───┬───┬───┐          ┌───┬───┬───┐
         M0 │C0 │C6 │C12│           M0 │C0 │C12│C24│       M6 │C6 │C18│C30│
         M1 │C1 │C7 │C13│           M1 │C1 │C13│C25│       M7 │C7 │C19│C31│
         M2 │C2 │C8 │C14│           M2 │C2 │C14│C26│       M8 │C8 │C20│C32│
         M3 │C3 │C9 │C15│           M3 │C3 │C15│C27│       ...
         M4 │C4 │C10│C16│           M4 │C4 │C16│C28│          └───┴───┴───┘
         M5 │C5 │C11│C17│           M5 │C5 │C17│C29│
            └───┴───┴───┘              └───┴───┴───┘

  Core1 在 M=768:  C1 block, A[128:256,:], K_start=32
  Core1 在 M=1536: C1 block(第1轮), C7 block(第2轮), 同一段 A 行, K_start 不变
```

**Core 1 在两轮中都处理 rows 128..255, K_start 始终 32 → batch 一致。**
但如果 M 增长导致 block 总数不能被 round 整除，`InitBlockIndex` 重分配可能
把同一行交给不同 core → 不同 K_start → 不同 K 迭代顺序 → FP 偏差。

### 8.5 batch 不一致的根因

```
必要条件 (全部同时满足):
  ① M 改变 → totalTileCnt 变 → InitBlockIndex 重新分配
  ② 同一段 M 行被分到 core_id 不同的核
  ③ 不同核的 K_start 不同 → K tile 迭代顺序不同
  ④ FP 加法非结合 → 不同累加顺序 = 不同结果
```

### 8.6 因素汇总

| 因素 | 机制 | 影响 |
|------|------|------|
| A 切片固定 | 每核 M 行范围不变, 全 K | 同核同偏移 → batch 一致 |
| K_SHIFT 偏移 | K_start = core_id × K/nCores | 不同核 = 不同 K 迭代顺序 |
| InitBlockIndex | 按 usedCoreNum/totalTileCnt 分配 | M 变→分配变→行可能换核 |
| FP 非结合 | (a+b)+c ≠ a+(b+c) | 不同迭代顺序→不同 FP 结果 |
| stepK 8→4 | L1 深度减半→partial sum 增多 | 放大累加顺序差异 |

### 8.7 预期与判据

| 场景 | 预期偏差 | 判据 |
|------|---------|------|
| BASE (无 K_SHIFT) | 逐字节一致 | MD5 全等 |
| K_SHIFT, 同核同偏移 | 逐字节一致 | MD5 全等 |
| K_SHIFT, 行换核 | ≤ 2 ULP (FP16) / ≤ 1 ULP (FP32) | max\|Δ\| + cos ≥ 0.999999 |

---

## 9. 测试

```python
# K_SHIFT 典型触发 shape
M, K, N = 768, 768, 768      # 标准: 全256对齐, K=768 非2幂
M, K, N = 12800, 2560, 2560   # middle-m

# batch 一致性: 需用宽松判据 (见 8.7)
```

> 脚本: `test_matmul_golden.py` — 40 golden + 13 batch (含 K_SHIFT 分支)
