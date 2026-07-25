# GMM（分组矩阵乘）类算子范式

真实源：`ops-transformer_AI/gmm/`。共 **7 个算子**。范式内容已对照真实代码验证。

## 算子清单

| 算子 | 说明 |
|---|---|
| `grouped_matmul` | 基础，A2A3 平铺 + A5 arch35 |
| `grouped_matmul_add` | + Atomic Add |
| `grouped_matmul_finalize_routing` | + token routing |
| `grouped_matmul_swiglu_quant` | + SwiGLU + 量化（A8W8/A8W4 MSD） |
| `grouped_matmul_swiglu_quant_v2` | V2（A4W4 / fusion path） |
| `quant_grouped_matmul_dequant` | 量化+反量化，2D M/N split |
| `quant_grouped_matmul_inplace_add` | 量化+inplace |

## 代码组织（GMM 特有，非 arch22/arch35 二分）

GMM **同时有 A2A3 和 A5**，但用三层结构（已验证）：

```
grouped_matmul/op_kernel/
├── arch35/                                  # A5（non_quant / quant_adaptive_sliding_window / weight_quant_basic_block）
├── a16w4_msd/                               # A2A3 A16W4 MSD
├── gmm_infra/                               # 共享模板基础设施（CUTLASS 式）
│   ├── arch/gmm_arch.hpp                    # 架构抽象层
│   ├── gemm/ epilogue/ layout/ detail/
├── grouped_matmul.cpp/.h                    # A2A3 主入口
├── grouped_matmul_apt.cpp                   # A5 入口（apt 后缀）
├── grouped_matmul_antiquant_a8w4.h          # A2A3 平铺量化变体（含 _msd/_pre/_nz）
├── grouped_matmul_a4w4.h / _regular.h       # A2A3 A4W4
└── grouped_matmul_tiling_key.h
```

> ⚠️ 判断 A2A3/A5 不能只看 arch22 子目录。GMM 的 A2A3 是平铺 + `a16w4_msd/` + `gmm_infra/arch/` 抽象。

## 分核策略（已验证 grep）

### A2A3 对角线分核
核心：`MNBlockIdxCompute(mnConfig, block, count, thresholdDimM)`
- `thresholdDimM = 5`（已验证）、`thresholdBlockNum = 8`（已验证）
- blockDimM ≤ thresholdDimM → 简单行优先；否则 → 窗口对角线
- 目的：规避 A2A3 同地址访问冲突

### A5 ASWT 对角线分组
`GroupedMatmulAswtWithTailSplitScheduler`：按 group 边界对角线分组 + 尾块 TailSplit。A5 支持同地址并行，无需错位规避。

### QuantGMM 2D M/N Split（已验证算法）
运行时从 L0C 容量反推最优 split：`l0CMNFractal=256`，遍历选 MTE2 流量最小的 M/N 核数配比。

### GEMV 阈值切换
M ≤ 阈值时 Normal MM → GEMV（逐行 Mmad + Dequant），避免 padding 浪费。

## 量化 UB 预算

通用：`UB = 激活×2(double buffer) + Weight + Scale + Workspace`。各算子 Scale 预算因 PerChannel/PerToken/PerGroup 而异，精确公式见 `*_antiquant_*.h`。

## 三阶流水（A8W4 MSD）

PreProcess(n+1) / MidProcess(n) / PostProcess(n-1) 滑动窗口重叠：weight cast(AIV) → matmul(AIC) → dequant+SwiGLU+requant(AIV)。

## V1 → V2 演进

V2 新增：A4W4 支持、groupListType direct count、GroupedMatmulDequantSwigluQuantFusion、A5 CUSTOM_CFG_MDL。

## 来源
- `ops-transformer_AI/gmm/`（find 全文件 + grep thresholdDimM/MNBlockIdxCompute 验证）