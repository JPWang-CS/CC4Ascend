# A2A3 分核策略

## GMM 对角线分核

A2A3 GMM 算子使用**对角线分核**策略，核心实现在 `MNBlockIdxCompute`。

### 算法逻辑

```
if blockDimM <= thresholdBlockNum (通常是 5):
    简单行优先：mIdx = block / blockDimN, nIdx = block % blockDimN
else:
    窗口对角线分核：
    - 对角线窗口大小 = thresholdDimM × blockDimN（如 5×N）
    - 按对角线逐窗口分配 block，减少同地址冲突
```

关键参数：
- `thresholdBlockNum = 8`（grouped_matmul_add）
- `thresholdDimM = 5`
- 对角线窗口避免相邻核访问相同 GM 地址

### 为什么需要错位

A2A3 硬件不支持同地址请求并行处理：
```
核0: offset=0 → 数据[0:N]
核1: offset=offset_1 → 数据[offset_1:N+offset_1]  // 偏移避免冲突
```

对角线分核本质是**错位分核**的变体：通过窗口化分配，让相邻核处理不重叠的 M 范围，减少 GM 访问冲突。

## Attention 多维分核

### A2A3 FlashAttention 四种拆分体系

1. **S1S2 拆分** (`FlashAttentionScoreS1s2Bn2gs1`)
   - 沿 S1 轴拆分：`s1BaseSize / s1BaseTailSize / s1OuterSize`
   - 沿 S2 轴拆分：`s2BaseSize / s2BaseTailSize / s2OuterSize`
   - 沿 D 轴拆分：`dBaseSize / dBaseTailSize / dOuterSize`
   - S2 轴支持 CAUSAL/BAND/PREFIX sparse mode 动态范围
   - `ComputeAxisIdx()`：`multiCoreInnerIdx` → `boIdx / n2oIdx / goIdx / s1oIdx`

2. **S1 拆分** (`FlashAttentionScoreS1Bn2gs1`)
   - 仅沿 S1 轴拆分，S2 整体在一个循环内
   - 使用 `STemplateType` / `DTemplateType` 编译期常量

3. **B 拆分** (`FlashAttentionScoreBn2gs1s2B`)
   - 沿 Batch 轴拆分：`bBaseSize / bBaseTailSize / bOuterSize`
   - 两个 MatMul 均使用 `BatchMode::BATCH_LESS_THAN_L1`
   - 使用 `IterateBatch` 而非 `IterateAll`

4. **Same-AB 布局** (`FlashAttentionScoreS1s2Bn2gs1SameAB`)
   - 在 S1 轴上通过 `vecCoreOffset` (AIV subBlockIdx) 进一步拆分给 AIV0/AIV1

### 分核配置编码

TilingKey 中 UB0/UB1/Block 三元组控制拆分维度：
- `UB0=S1, UB1=S2, Block=NONE` → 沿 S1 和 S2 拆分
- `UB0=S1, UB1=D, Block=NONE` → 仅沿 S1 拆分
- `UB0=NONE, UB1=NONE, Block=B` → 沿 batch 拆分

## QuantGMM 的 2D Split

`quant_grouped_matmul_dequant` 使用运行时计算的 2D 分核：

```cpp
// 根据 L0C 容量反推
l0CMNFractal = 256;  // 256*1024/2/16/16/4
oriBaseMN = floor(sqrt(l0CMNFractal));

// 尝试 0..3，选择 MTE2 流量最小的 split
MCoreNum = 1 << (3 - chosen);
NCoreNum = 1 << chosen;
```

目标是 minimize MTE2 traffic（搬运量）→ 自动找到 M/N 方向的最佳切分比例。
