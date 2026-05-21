# Activation 激活函数 API

## 官方内置激活函数

`ActivationOperation` API (CANN 8.0+)：

| 激活函数 | 公式 | 典型场景 |
|---------|------|----------|
| **ReLU** | max(0, x) | CNN、通用 |
| **GELU** | $x \cdot \Phi(x)$（tanh 近似） | Transformer、BERT |
| **SiLU/Swish** | $x \cdot \sigma(x)$ | LLaMA、EfficientNet |
| **LeakyReLU** | x>0 ? x : αx | GAN |
| **Sigmoid** | $1/(1+e^{-x})$ | 二分类 |
| **Tanh** | $\tanh(x)$ | RNN/LSTM |

## GELU 实现

GELU 使用 tanh 近似：
$$GELU(x) = \frac{x}{1 + \exp(-1.595769122 \cdot (x + 0.0455399241 \cdot x^3))}$$

Host 侧调用（二段式）：
```cpp
aclnnGeluGetWorkspaceSize(input, output, &workspaceSize, &executor);
aclnnGelu(workspaceAddr, workspaceSize, executor, stream);
```

## 自定义激活函数：多项式拟合

霍纳法则 + 多项式拟合（仅需 5 条 FMA 指令）：

```cpp
__aicore__ inline float horner(float x, const float coeff[6]) {
    float result = coeff[5];
    result = Mad(result, x, coeff[4]);  // result*x + c4
    result = Mad(result, x, coeff[3]);
    result = Mad(result, x, coeff[2]);
    result = Mad(result, x, coeff[1]);
    result = Mad(result, x, coeff[0]);
    return result;
}
```

性能对比（Atlas 300I Pro，SiLU）：
- 组合指令法：12.8 TFLOPS, 4.2μs
- 多项式法：**18.3 TFLOPS, 2.1μs**（吞吐 +43%, 延迟 -50%）

拟合区间 [-4, 4] 内 5 阶多项式可达 1e-6 级别误差。

## 注意事项

| 事项 | 说明 |
|------|------|
| **FP16 中间转 FP32** | 涉及 Exp/Tanh 等超越函数时先 Cast 到 FP32 |
| **融合优于独立调用** | 紧跟 MatMul 的激活应在 GetTensorC 后直接在 UB 处理 |
| **分段策略** | 大正值 / 大负值 / 零点附近 / 核心区域分段处理 |

## 代码仓中的 SwiGLU 实现

```cpp
// GMM SwiGLU: x * sigmoid(x*beta)
SwiGLU<float, false>(workspace, src0, src1, beta, halfTokenLen);
```

SwiGLU 在 grouped_matmul_swiglu_quant 中作为 VecProcess（AIV 侧）的一部分，紧跟在 MatMul dequant 之后，在 quant 之前。
