# A5 分核策略

## 核心变化：无需错位

A5 硬件新增**同地址请求并行处理**特性，不再需要 A2A3 的错位分核：
```
A5: 核0: 数据[0:N], 核1: 数据[N:2N]  // 直接均分
```

## FlashAttention 新分核模式

A5 `FlashAttentionScoreKernelTrain` 实现三种分核模式：

### 顺序分核
```
s1基本块1 → core1, s1基本块2 → core2, s1基本块3 → core3,
s1基本块4 → core1, s1基本块5 → core2, ...
```
提高 L2 上数据复用。

### 对称分核
```
上半部分 N: 顺序分核
下半部分 N: 与上半部分对称分核
```
用于 BN2 方向存在对称性的场景。

### 正倒序循环分核（TND）
```
先正序: s1块1→core1, s1块2→core2, s1块3→core3
再倒序: s1块4→core3, s1块5→core2, s1块6→core1
```
TND（variable-length）场景专用，varlenCycleCoreNums = coreNum * 2。

### 分核实现

```cpp
// 计算实际分核索引
__aicore__ inline int64_t CalcRealCoreIdx(
    int64_t relativePos, int64_t times, int64_t offsetCoreIdx, bool isPartialCalc);

// TND 专用
__aicore__ inline int64_t CalcRealCoreIdxVarlen(
    int64_t calcLoops, int64_t calcLoopsRemain, int64_t cycleCoreNums);
```

## GMM ASWT 对角线分组（cgmct 框架）

A5 `quant_grouped_matmul_inplace_add` 使用 `GroupedMatmulAswtWithTailSplitScheduler`：

ASWT = Adaptive Split with Tail。核心思想：
- 按 group 边界进行对角线分组
- 窗口内 M/N 块分配使用 `BlockMmad` 模块化调度
- 尾块特殊处理（TailSplit）

与 A2A3 对角线分核的区别：
- A2A3: `MNBlockIdxCompute` 手动计算 block → (mIdx, nIdx) 映射
- A5: cgmct 调度器自动管理 block 分配和 Group 边界处理

## A5 GMM Weight Quant Tiling（kernel 级）

A5 的 `grouped_matmul` weight quant 模式在 tiling 中选择 baseM/baseN/baseK：

```
从 L0C 容量出发：
  maxSingleM = L0C_SIZE / (baseN * sizeof(float))  // 输出用 float
  maxSingleN = L0C_SIZE / (baseM * sizeof(float))
  maxSingleK = L0C 受限的 K 方向最大切片
  
从 UB 容量出发：
  ubBudget = UB_SIZE - scale_buf - workspace
  tileM = ubBudget / (n * sizeof(half) * DOUBLE_BUFFER)
```

不是从固定模板出发，而是从硬件容量反推最优 tile 大小。

## 分核通用原则

| 维度 | A2A3 | A5 |
|------|------|-----|
| 同地址冲突 | 需要错位规避 | 硬件支持并行，直接均分 |
| 分核策略 | offset_i = f(coreIdx) | 规则滑动窗口 |
| L2 复用 | 不考虑 | 顺序分核提高 L2 复用 |
| 调度框架 | 手动 MNBlockIdxCompute | cgmct Scheduler 自动调度 |
| Tiling 粒度 | 从枚举模板选择 | 从硬件容量反推 |
