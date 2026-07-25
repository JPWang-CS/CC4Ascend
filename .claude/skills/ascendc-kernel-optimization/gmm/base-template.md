# GMM 类 base 模板（深度版）

以 `grouped_matmul` A2A3 为基准。内容 trace 自 `gmm/grouped_matmul/op_kernel/grouped_matmul.h`。

## 数学本质（优化出发点）

GMM = 一组矩阵乘（每 group 一个 MatMul，M/N/K 因 group 而异）。与普通 MatMul 的区别：**多 group 边界** + **group 间 M 不均**。

核心矛盾：多核并行算不同 group 的输出块时，A2A3 硬件**不支持同地址并行**——若相邻核同时写相近的 GM 输出区域，硬件串行化导致利用率骤降。base 模板的核心优化是**对角线分核**，让相邻核命中不同输出区域。

## 对角线分核（base 核心，已 trace grouped_matmul.h:27-113）

### 阈值（已 trace，含注释）
```cpp
constexpr uint32_t thresholdDimM = 1;  // :27 无特殊策略（纯行优先）
constexpr uint32_t thresholdDimM = 5;  // :29 "5 is obtained by tests" 对角线启用阈值
// thresholdBlockNum = 8（对角线窗口边长，经验值）
```

### 算法（MNBlockIdxCompute，已 trace :90-113）

**简单路径**（blockDimM ≤ 5）：行优先线性映射
```
mIdx = (curBlock - count) / blockDimN
nIdx = (curBlock - count) % blockDimN
```
相邻 block 的 (mIdx, nIdx) 在 N 方向连续 → 相邻核写相邻输出列 → A2A3 上会冲突。仅在 blockDimM 小时用（冲突概率低）。

**对角线窗口路径**（blockDimM > 5）：
```
relativeBlock = curBlock - count
curThresholdM = 窗口 M 边长（含尾部 blockDimM % 8 处理）
curThresholdN = 窗口 N 边长（同理）
localRelativeBlock = relativeBlock % thresholdM_dimN % (curThresholdM*curThresholdN)
mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * 8
nIdx = (localRelativeBlock + localRelativeBlock / LCM(curThresholdM, curThresholdN)) % curThresholdN + ...
```
相邻 block 的 (mIdx, nIdx) 沿**对角线**推进而非行优先：curBlock→(m,n), curBlock+1→(m+1,n+1) 方向。相邻核命中的输出块在 M 和 N 方向都错开 → 不写同一 GM 区域 → 无同地址冲突。

### 为什么是 thresholdDimM=5
注释明示"obtained by tests"。M 维块数 ≤5 时输出区域窄，相邻核冲突概率低，行优先的简单性更划算；>5 时输出区域宽，冲突成为瓶颈，对角线的错位收益超过其计算开销。

## TilingData 结构（已 trace grouped_matmul.cpp:123-220）
```
GMMTilingData:
  gmmBaseParams    # group 数 / 各 group M/N/K / 布局
  mmTilingData     # 单 group matmul 的 baseM/baseN/baseK
  gmmArray         # 多 group 偏移地址数组
```

## 类层次（已 trace :114/:323/:377）
```
GMMProcess<ComputeType>               # :114 单 group 处理（MatMul + 后处理）
  └── GMMGroupMSparseProcess          # :323 group M=0 跳过的稀疏变体
GMMCompute<...>                        # :377 顶层编排（遍历 group + 分核调度）
```

## A5 base（arch35 + gmm_infra，已 trace 结构）

```
op_kernel/arch35/{non_quant, quant_adaptive_sliding_window_templates, weight_quant_basic_block}/
op_kernel/gmm_infra/{arch/gmm_arch.hpp, gemm/, epilogue/, layout/, detail/}
```

### A5 分核：ASWT（已验证文件名）
`arch35/quant_adaptive_sliding_window_templates/` 的 `GroupedMatmulAswtWithTailSplitScheduler`：按 group 边界对角线分组 + 尾块 TailSplit。A5 硬件支持同地址并行，**无需 A2A3 的错位规避**，可直接用更规则的滑动窗口，减少无效偏移与冗余地址变换。

### gmm_infra 模板层（A5 新增）
CUTLASS 式模板库：`gemm/{block,tile}/`（MatMul 模板）、`epilogue/`（后处理模板）、`arch/gmm_arch.hpp`（A2A3/A5 架构抽象）。A5 GMM 通过这层复用，不再 arch22/arch35 硬拆。

## A2A3 vs A5 分核本质差异

| | A2A3 | A5 |
|---|---|---|
| 同地址并行 | 不支持 | 支持 |
| 分核策略 | 对角线错位规避（thresholdDimM=5） | 规则滑动窗口（ASWT） |
| 相邻核关系 | 必须错开输出区 | 可直接相邻 |
| 尾块处理 | thresholdBlockNum 取模 | TailSplit |

迁移要点：A2A3→A5 时对角线约束可放开，先功能等价保留 tile，profiling 确认同地址并行生效后再简化。

## 相关
- [分核详解](split-and-core.md)（2D split / GEMV 阈值）