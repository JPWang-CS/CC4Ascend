# FFN 类 base 模板（深度版）

以 `ffn` 为基准。trace 自 `ffn/ffn/op_kernel/ffn_base.h`。

## 数学本质（优化出发点）

FFN = 两段 MatMul 夹激活：`MatMul1(x, W1) → Activation → MatMul2(_, W2)`。MoE FFN 还多一层 expert dispatch（每个 token 路由到不同专家的 W1/W2）。

核心矛盾：两段 MatMul + 激活若串行（各自独立 kernel），中间结果（MatMul1 输出）要回 GM 再被 MatMul2 读，**GM 往返浪费带宽**。base 优化是融合 + 量化场景的 MSD 多级流水。

## base 实现：FFNBase（已 trace ffn_base.h:24）

### 两段 MatMul + 专家调度
```cpp
class FFNBase {
    void Init(x, weight1, weight2, ...);        // :85
    void ComputeExpertParallNum(expertI, baseM, ...);  // :474 算每个专家的并行度
    void ComputeZeroN1WithoutBias(coreIdx, expertIdx);  // :486 N1 段无 bias
    void ComputeStepZeroN1WithBias(expertIdx, ...);     // :501 N1 段有 bias
    void ComputeZeroN1WithBias(coreIdx, expertIdx);     // :542
    void ProcessNormal();                                // :595 正常流程
};
```

### 专家负载均衡（ComputeExpertParallNum，:474）
MoE FFN 各 expert token 数不均。该函数按 expertI 和 baseM 算每个专家分到多少核，保证多核负载均衡。不均衡会导致部分核空等。

### N1 段变体（:486/:501/:542）
MatMul1（N1 段）分有 bias / 无 bias、step 与否多个变体，由 tiling 按场景选。bias 存在时 ComputeStep 多一步加 bias。

## MSD 多级流水（ffn_antiquant_msd.h）

### 为什么 MSD
量化 FFN（A8W4/A16W4）需 weight 反量化（Vector）再做 MatMul（Cube）。串行则 Vector 反量化时 Cube 空等。MSD（Multi-stage Dataflow）让反量化(n+1) / matmul(n) / 后处理(n-1) 滑动窗口重叠。

### 三级（类比 gmm swiglu_quant MSD）
```
PreProcess(n+1):  weight cast / 反量化（AIV）
MidProcess(n):    MatMul（AIC）
PostProcess(n-1): activation + requant（AIV）
```
三级重叠让 AIV 和 AIC 都不空等。

### 适用范围
weight 需反量化的量化 FFN。非量化 FFN 不需要 MSD（无反量化阶段）。

### 代价
三级流水需 3 份 workspace 缓冲；expert 边界处流水要 flush，短 expert 收益小。

## GLU 变体（ffn_glu.cpp）
GLU 激活：MatMul1 输出切半，一半 gating × 另一半，再 MatMul2。`ffn_glu.cpp` 是独立入口处理这个切半拼接逻辑。

## base 设计理由总结
1. 两段 MatMul 融合（中间留 UB / 流水，不回 GM）
2. MoE 专家负载均衡（ComputeExpertParallNum）
3. 量化场景 MSD 三级流水掩盖反量化
4. N1 段按 bias/step 选变体

## FFN 系列
- `ffn`：MoE + 普通 FFN
- `swin_attention_ffn` / `swin_transformer_ln_qkv`(+quant)：Swin 专用，Graph only
- `ffn_worker_scheduler` / `ffn_worker_batching`：Attn/FFN 分离架构，AI CPU