# GMM 分核优化（深度版）

trace 自 `gmm/quant_grouped_matmul_dequant/` 与 `grouped_matmul/`。对角线分核的完整算法见 [base-template](base-template.md)。

## 2D M/N Split（A5，已 trace quant_grouped_matmul_dequant.h:76,301）

### 机制
A5 运行时从 L0C 容量反推最优 M/N 核数配比。入口 `ProcessGroup(..., uint32_t nCoreNum)`（:76），mode 决定 nCoreNum（:301）。

### 算法
```
l0CMNFractal = 256   # L0C 能容纳的 MN 分形数
oriBaseMN = floor(sqrt(l0CMNFractal))
遍历 i=1..3 选 MTE2 流量最小配置:
  mte2Now = (fracN << (3-i)) + (fracM << i)
  取最小者 chosen = i
MCoreNum = 1 << (3 - chosen)
NCoreNum = 1 << chosen
```

### 为什么这样优化
对角线分核（1D）在 M、N 都大时负载不均：M 方向核多则 N 复用差，反之亦然。2D split 让 M 和 N 方向都有核，L0C 容量约束下选 MTE2 搬入流量最小的配比（流量主导性能）。遍历 4 个候选而非解析求解，因 L0C 分形映射非线性，穷举 4 次代价可忽略。

### 适用范围
M、N 都较大的 A5 场景。M 或 N 很小时退化为 1D（只有一方有核）。

### 代价
group 边界处 M/N 配比需重算；多核写不同输出区域需保证不冲突（A5 同地址并行支持，无需额外规避）。

## GEMV 阈值切换（已 trace）

### 阈值
```cpp
constexpr uint32_t GEMV_THRESHOLD = 8;   // quant_grouped_matmul_dequant_base.h:53
constexpr uint32_t TILING_KEY_GEMV = 10000001;  // config.h:123
```

### 为什么 M≤8 切 GEMV
Normal MatMul 路径按 baseM×baseN 分形算，M≤8 时 baseM 方向大量 padding（Cube 单元 16×16 分形，M=8 只用一半），浪费一半算力且多搬无效数据。GEMV 路径逐行 LoadData + Mmad，无 padding。

### 实现（quant_grouped_matmul_dequant_gemv.h:254-295）
```cpp
LoadData2dParams loadData2DA;   // A（激活）逐行
loadData2DA.repeatTimes = realM;
LoadData2dParams loadData2DB;   // B（权重）
LoadData<int8_t>(l0aXGemv, l1XGemv[...], loadData2DA);
// 逐行 Mmad + Dequant
float pertokenScaleDequant[GEMV_THRESHOLD];  // :406
```

### 适用范围
M ≤ 8 的瘦长矩阵（激活少、权重多）。M > 8 时 GEMV 反而因逐行调度开销变慢。

## ASWT 对角线分组（A5）
见 [base-template §A5 分核](base-template.md)。按 group 边界对角线 + TailSplit，A5 同地址并行无需错位。

## 优化选择决策

| 场景 | 选择 | 原因 |
|---|---|---|
| A2A3 + blockDimM>5 | 对角线分核（thresholdDimM=5） | 规避同地址冲突 |
| A2A3 + blockDimM≤5 | 行优先 | 冲突概率低，简单更划算 |
| A5 + M/N 都大 | 2D M/N Split | 负载均衡，MTE2 流量最优 |
| A5 + M≤8 | GEMV（TILING_KEY_GEMV） | 避免 padding 浪费 |
| A5 + 多 group | ASWT | 同地址并行，规则窗口 |