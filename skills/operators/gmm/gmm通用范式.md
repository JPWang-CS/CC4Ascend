# GMM（分组矩阵乘）类算子通用范式

## 算子列表（11 个全面分析）

| 算子 | 量化 | A5 (arch35) | A2A3 | 分核策略 |
|------|:---:|:---:|:---:|------|
| `grouped_matmul` | a16w8 quant | ✓ | ✓ | A2A3 对角线 / A5 ASWT |
| `grouped_matmul_add` | — | ✓ | ✓ | 对角线 + Atomic Add |
| `grouped_matmul_finalize_routing` | A8W8/A8W4 | ✓ | ✓ | 对角线分组 + token routing |
| `grouped_matmul_swiglu_quant` | A8W8/A8W4 MSD | — | ✓ | 3-stage pipeline + workspace-split |
| `grouped_matmul_swiglu_quant_v2` | A8W4/A4W4 | ✓ | — | A5 MSD / fusion path |
| `quant_grouped_matmul_dequant` | 量化+反量化 | — | ✓ | 2D M/N split + GEMV threshold |
| `quant_grouped_matmul_inplace_add` | 量化+inplace | ✓ | — | cgmct ASWT 对角线分组 |

## 分核策略详解

### 对角线分核（A2A3）

核心实现：`MNBlockIdxCompute(mnConfig, block, count, thresholdM_dimN)`

```
if blockDimM <= thresholdDimM (通常 5):
    简单行优先：mIdx = block / blockDimN, nIdx = block % blockDimN
else:
    窗口对角线分核：
    - 对角线窗口大小 = thresholdDimM × blockDimN
    - 按对角线逐窗口分配 block
```

关键参数：
- `thresholdBlockNum = 8`（grouped_matmul_add）
- `thresholdDimM = 5`
- 目的：避免相邻核访问相同 GM 地址（A2A3 不支持同地址并行）

### ASWT 对角线分组（A5 cgmct）

`GroupedMatmulAswtWithTailSplitScheduler`：
- 按 group 边界进行对角线分组
- 窗口内 M/N 块使用 BlockMmad 模块化调度
- 尾块 TailSplit 特殊处理
- 无需手动计算 MNBlockIdx → M/N 映射

### QuantGMM 2D M/N Split

`quant_grouped_matmul_dequant` 运行时计算最优 2D split：
```cpp
l0CMNFractal = 256;  // 从 L0C 容量
oriBaseMN = floor(sqrt(l0CMNFractal));
// Try 0..3，选 MTE2 流量最小的配置
for (int32_t i=1; i<4; i++) {
    int32_t mte2Now = (fracN << (3-i)) + (fracM << i);
    if (mte2Now < mte2Min) { chosen = i; }
}
MCoreNum = 1 << (3 - chosen);
NCoreNum = 1 << chosen;
```

### GEMV 阈值切换

当 M <= GEMV_THRESHOLD(=8) 或 workspace 较小时：Normal MM → GEMV（逐行 Mmad + Dequant），避免浪费的 padding 计算。

## 量化场景 UB Buffer 分配

### 通用预算公式

```
UB_Budget = 激活 × 2 (double buffer) + Weight + Scale + Workspace
```

### 各算子 Scale 预算

| 算子 | Scale 模式 | UB 预算计算 |
|------|-----------|------------|
| grouped_matmul (A2A3) | PerChannel / PerToken / PerGroup | `baseN * sizeof(float) * 2` (double buffer) |
| grouped_matmul_finalize_routing | PerChannel + PerToken | scale_queue + perToken_queue + dequant_middle + muls_result + bias + shared_tmp |
| grouped_matmul_swiglu_quant | PerChannel + PerToken | `n * 4 * 2 + tokenLen * 4` |
| quant_grouped_matmul_inplace_add (A5) | PERTENSOR + PERCHANNEL | `baseM * sizeof(float)` ping-pong via BlockEpilogueDequant |

### SwiGLU Quant UB 计算（A8W8）

```
tmpBufSize = (n/2) * 4        // SwiGLU reduce workspace
perChannelBuf = n * 4 * 2     // double buffered channel scale
remainingUb = ubSize - tmpBuf - perChannelBuf
maxRowInUb = remainingUb / (n*4 + n/2 + 4) / 2
```

### SwiGLU Quant UB 计算（A8W4 MSD）

```
8.5 * row * n + 4 * alignUp(row, 8) + 6n + 64 <= ubSize
```

## 三阶流水线（A8W4 MSD）

```
for each workspaceSplitLoopIdx:
  // Stage 1: PreProcess (n+1) - weight cast in AIV
  // Stage 2: MidProcess (n) - matmul in AIC  
  // Stage 3: PostProcess (n-1) - dequant + SwiGLU + requant in AIV
```

滑动窗口：迭代 n 的 MidProcess 与 n+1 PreProcess、n-1 PostProcess 重叠。

## Pre-deferred MMCompute 模式

```cpp
while (curBlock < curCount) {
    MnBlockIdxCompute(mnConfig, ...);
    computeOp.MmCompute(lastMnConfig, blockIdx);     // 算上一个
    computeOp.MmComputePrepare(groupIdx, mnConfig);   // 准备当前（软件预取）
    curBlock += coreNum;
}
computeOp.MmCompute(lastMnConfig, blockIdx);  // 最后一个
```

## Atomic Add 机制（grouped_matmul_add）

多核写重叠输出位置时自动原子累加：
```cpp
SetAtomicAdd<float>();
DataCopyPad(yGm[outOffset], tensorIn, copyParams);
SetAtomicNone();
```

## V1 → V2 演进

| 特性 | V1 | V2 |
|------|-----|-----|
| TILING_KEY | 0/1/2 | 0/2/3/4/5 |
| A4W4 支持 | ❌ | ✅ |
| groupListType | cumsum only | cumsum + direct count |
| Fusion path | ❌ | ✅ GroupedMatmulDequantSwigluQuantFusion |
| A5 优化 | — | CUSTOM_CFG_MDL (enableGetTensorC=true) |

## 来源
- Agent 深度分析 `gmm/` 全部 7 个算子目录 11 个 kernel 文件
