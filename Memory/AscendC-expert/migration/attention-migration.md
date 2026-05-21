# Attention A2A3 → A5 迁移实例

## FlashAttentionScore TilingKey 简化

### A2A3: 18 字段, 64-bit
```
KernelType(1), VectorAllReduce(1), ImplMode(4), Layout(4),
S1TemplateType(4 enum), S2TemplateType(4 enum), DTemplateType(4 enum),
IsGroupCnt(4), AccumLevel(4), BigDoubleBuffer(2),
PseMode(4), BiasMode(4), DeqMode(4), AntiquantMode(4),
HasQuant(4), HasRope(4), HasDelta(4), OutDtype(4)
```

### A5: 12 字段, 62-bit
```
KernelTypeKey(1), ImplMode(2), Layout(4),
S1TemplateType(10) — ★ 直接存 size 值 0/16/64/128/256
S2TemplateType(10) — ★ 直接存 size 值 0/16/32/64/128/256/512
DTemplateType(12) — ★ 直接存 size 值 0/16/32/48/64/80/96/128/160/192/256/768
DvTemplateType(12) — ★ NEW: Dv 独立于 D
PseMode(4), HasAtten(1), HasDrop(1), HasRope(1), OutDtype(2), Regbase(1)
```

### 删除的字段
- UB0/UB1/Block 维度选择（不再手动映射 buffer slot）
- Bmm1Format / Bmm2Source / BigDoubleBuffer / EnableL1Reuse
- Sparse mode / MatmulPolicyType
- DataType（替换为 OutDtype）
- HasAttenMask（合并入 HasAtten）

## 分核策略变化

| 方面 | A2A3 | A5 |
|------|------|-----|
| 分配方式 | offset-based (formerNum/tailNum) | CalcRealCoreIdx (times/offsetCoreIdx) |
| 策略 | 单一种类 S1/S2 等距拆分 | 3 种模式：顺序/对称/正倒序循环 |
| L2 复用 | 不考虑 | 顺序分核提高 L2 复用 |
| TND | 不支持 | 正倒序循环分核 |

实际代码变化：
```cpp
// A2A3: 手动计算每个核的 offset
singleCoreSeqSize = ...; eachCoreSeqSize[i] = ...;

// A5: 新增 splitCoreMode 控制分发模式
CalcRealCoreIdx(relativePos, times, offsetCoreIdx, isPartialCalc);
```

## Buffer 容量变化

```
        A2A3       A5        变化
总AI核: 24     →   36        (+50%)
L0C:    128KB  →   256KB     (2x!)
UB:     192KB  →   248KB     (+29%)
BIAS:   1KB    →   4KB       (4x)
L2:     192MB  →   128MB     (-33%)
L1:     —      →   512KB     (新增)
L0A/B:  64KB   →   64KB      (不变)
```

## 同步机制变化

### A2A3: 本地 SetFlag/WaitFlag
```cpp
SetFlag<HardEvent::MTE2_V>(0);
WaitFlag<HardEvent::MTE2_V>(0);
// eventID 范围 0-2
```

### A5: 跨核 CrossCoreSetFlag/WaitFlag
```cpp
CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
// eventID 可达 22
InitSyncFlags<4, 4, 4>();  // 三种模式独立配置
```

## CV 融合通路

A5 新增 Cube↔Vector 直连：
- `L0C2UB`（Fixpipe L0C→UB）：无需写 GM
- `UB2L1`（DataCopy ND→NZ）：UB 直接搬入 L1 供下一轮 Cube

A2A3 必须 L0C → GM → UB → L1，多两跳。

## IncreFlashAttention 全面 Regbase 化

A5 IFA 迁移要点：
- 新增 `incre_flash_attention_entry_regbase.h` entry
- TilingData 改用 `IncreFlashAttentionTilingDataRegbase`
- 模块化拆分：mm1/2_processor, vec1/2_processor
- 新增参数：`quantMode`, `KvLayoutType` (PA_BBH/PA_BNBD/PA_NZ), `isFd`, `enableKVPrefix`, `enableS1OutSplit`
