# Attention 类算子通用范式

## 算子全景

53 个 Attention 算子目录，覆盖 FlashAttention、SparseAttention、MLA、IncreFlashAttention、BlockSparseAttention 等。

## 代码组织架构

```
attention/
├── ${op_name}/op_kernel/
│   ├── ${op_name}.cpp           # 核函数入口
│   ├── ${op_name}.h             # Kernel 类
│   ├── arch22/                  # A2A3 架构代码
│   │   └── *_template_tiling_key.h  # A2A3 TilingKey (18 字段, 64-bit)
│   └── arch35/                  # A5 架构代码
│       ├── kernel/              # 核函数 (*_regbase.h)
│       ├── service/             # 微核服务 (mm1/2_processor, vec1/2_processor)
│       ├── matmul_modules/      # MatMul 模块
│       └── vector_api/          # Vector API 层
└── common/op_kernel/arch35/     # A5 共享设施 (50+ files)
```

## FlashAttention 分核策略

### A2A3 四种拆分体系

| Kernel 类 | 拆分母 | 拆分逻辑 |
|-----------|--------|----------|
| `S1s2Bn2gs1` | S1 + S2 + D | s1/s2/d BaseSize/BaseTailSize/OuterSize |
| `S1Bn2gs1` | 仅 S1 | S2 整体循环内处理 |
| `Bn2gs1s2B` | Batch | b BaseSize/BaseTailSize, `IterateBatch` |
| `S1s2Bn2gs1SameAB` | S1 + AIC-AIV 配对 | S1 + vecCoreOffset (subBlockIdx) |

TilingKey 中 UB0/UB1/Block 控制拆分维度：
- UB0=S1, UB1=S2 → S1+S2 拆分
- UB0=S1, UB1=D → 仅 S1
- Block=B → Batch 拆分

### A5 三种新分核模式

在 `CalcRealCoreIdx()` 中实现：

1. **顺序分核**：s1 块依次分发 → core1/2/3 → 轮回，提高 L2 复用
2. **对称分核**：N 上半顺序 + N 下半对称
3. **正倒序循环分核（TND）**：正序分发 → 倒序回发，varlenCycleCoreNums = coreNum × 2

## 流水同步事件链

### A2A3 三缓冲

```
taskId 0:       IterateBmm1
taskId 1:       WaitBmm1Result → IterateBmm1 → ProcessVec1 → SetFlag(MTE3_MTE2) → WaitFlag → IterateBmm2
taskId 2+:      ... → IterateBmm2 → WaitBmm2Result → ProcessVec2
```

### A5 四阶流水线

```
taskId 0:       IterateBmm1
taskId 1:       IterateBmm1 → ProcessVec1
taskId 2:       IterateBmm1 → ProcessVec1 → IterateBmm2
taskId 3+:      IterateBmm1 → ProcessVec1 → IterateBmm2 → ProcessVec2
```

### ProcessVec1 内部事件链

```
WaitBmm1Result → GetBmm1Result → SetFlag(MTE2_V) → WaitFlag(MTE2_V) →
Muls(scale) → CopyInAttenMask → ComputeAttenMask →
[PSE 条件同步] → SetFlag(V_MTE2)/WaitFlag(V_MTE2) →
SoftMaxCompute → Cast + DataCopy / NdToNz →
SetFlag(V_MTE3) → WaitFlag(V_MTE3)
```

### Same-AB 跨核同步

```cpp
constexpr uint64_t SYNC_C1_V1_FLAG[3] = {4, 5, 6};  // Cube→Vector
constexpr uint64_t SYNC_V1_C2_FLAG[3] = {7, 8, 9};  // Vector→Cube
constexpr uint64_t SYNC_C2_V2_FLAG[3] = {1, 2, 3};  // Cube→Vector
```

AIC: IterateBmm1 → SetFlag(C1_V1) → AIV: ProcessVec1 → SetFlag(V1_C2) → AIC: IterateBmm2 → SetFlag(C2_V2) → AIV: ProcessVec2

## TilingKey 详细位域

### A2A3 (64-bit, 18 字段)

```
bits 5-2:   ImplMode (4)
bits 9-6:   Layout (4): BSND/SBND/BNSD/TND
bits 13-10: S1TemplateType (4 enum)
bits 17-14: S2TemplateType (4 enum)
bits 21-18: DTemplateType (4 enum)
bits 29-26: BigDoubleBuffer (2): NONE/BIG_BUFFER/BIG_DOUBLE_BUFFER
bits 35-32: PseMode (4)
bits 47-44: AntiquantMode (4)
bits 51-48: HasQuant (4)
bits 63-60: OutDtype (4)
```

### A5 (62-bit, 12 字段)

```
bits 17-8:  S1TemplateType (10) — ★ 直接存 size 值 0/16/64/128/256
bits 27-18: S2TemplateType (10) — ★ 直接存 size 值 0/16/32/64/128/256/512
bits 39-28: DTemplateType (12) — ★ 0/16/32/48/64/80/96/128/160/192/256/768
bits 51-40: DvTemplateType (12) — ★ NEW: Dv 独立于 D
bit 56:     HasAtten (1) — 合并了 A2A3 的 HasAttenMask
bit 57:     HasDrop (1)
bit 58:     HasRope (1)
bit 61:     Regbase (1) — ★ NEW
```

A5 删除了 UB0/UB1/Block 维度映射、BigDoubleBuffer、Bmm1Format/Bmm2Source、Sparse mode 等 10+ 个字段。

## Softmax 特殊机制

- **softmaxReduceSize**：当 s1BaseSize > 256 时 reduceSize=1（否则 8），使用 `SOFTMAX_REDUCE_CFG` + `softmaxTempBuf` Brcb
- **AA_INVALID_LINE_HIGH_PRECISION** (ImplMode=2)：SoftMaxCheckRes (uint16_t bitmap) 追踪无效行，AdjustSoftMaxRes 精度修正
- **Sink 操作**：增量 Softmax 的初始 max/sum 通过 S_V 同步从 Scalar 读取

## IFA（IncreFlashAttention）A5 关键变化

- Entry: `incre_flash_attention_entry_regbase.h` — 全面 Regbase 化
- 新增模板参数：`quantMode`, `KvLayoutType` (PA_BBH/PA_BNBD/PA_NZ), `isFd`, `enableKVPrefix`, `enableS1OutSplit`
- TSCM 双缓冲：`TSCM<QuePosition::VECIN, 1, 0x4>` 替代 TQue 用于轻量 AIC-AIV 数据传递
- TilingKey Config 字段 (10 bits)：将 S1/S2/D/DV 编码为单一枚举值 (0-18)

## GM Workspace 布局

```
mm1Res[2]    = 2 * mmNRatioOffset     (bmm1 ping-pong)
stage1Res[2] = 1x or 2x mmNRatioOffset
mm2Res[2]    = 2 * mm2Offset           (bmm2 ping-pong)
vec2Res[2]   = 2 * mm2Offset           (vec2 ping-pong)
```

## UB 缓冲分配（S1S2 kernel）

```
maskTBufPing 9KB | maskTBufPong 16KB | pseTBuf 16KB
stage1PingBuf 32KB | stage2TBuf 32KB | commonTBuf 32KB
softmaxSumBuf[2] 2KB×2 | softmaxExpBuf[2] 2KB×2 | softmaxMaxBuf single
softmaxTempBuf 16KB (reduceSize==1 时) | stage1PongBuf 34KB
```

## 来源
- Agent 深度分析 `attention/flash_attention_score` arch22/arch35 全部 16 个文件
- Agent 分析 `attention/incre_flash_attention` 全面 Regbase 转化
- Agent 分析 `attention/block_sparse_attention` arch22 vs arch35 同步差异
