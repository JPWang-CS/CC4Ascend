# SoftMax 高阶 API

## 数值稳定性原理

标准 Softmax：
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**核心问题**：FP16 最大表示 65504，$e^{12} \approx 162754$ 即溢出。

**Max-Shift 解法**：
$$\text{Softmax}(x_i) = \frac{e^{x_i - M}}{\sum_j e^{x_j - M}},\quad M = \max(x)$$

平移后所有输入 ≤ 0，$e^{\le 0} \in (0,1]$，永不溢出。

## Vector-Only 全向量化实现

```cpp
template <typename T>
__aicore__ inline void ComputeSoftmax(
    LocalTensor<T>& input, LocalTensor<T>& output, int32_t len)
{
    LocalTensor<T> maxLocal, workBuf;
    ReduceMax(maxLocal, input, workBuf, len);

    // ★ 使用 Brcb 广播（全程不经过 Scalar 单元）
    LocalTensor<T> maxBroadcasted;
    Brcb(maxBroadcasted, maxLocal, 1, len);

    LocalTensor<T> tmpLocal;
    Sub(tmpLocal, input, maxBroadcasted, len);

    // 先 Cast 到 FP32 保证精度
    LocalTensor<float> tmpF32, expF32;
    Cast(tmpF32, tmpLocal, RoundMode::CAST_NONE, len);
    Exp(expF32, tmpF32, len);

    // 求和
    LocalTensor<float> sumLocal;
    ReduceSum(sumLocal, expF32, workBuf, len);

    // 广播和
    LocalTensor<float> sumBroadcasted;
    Brcb(sumBroadcasted, sumLocal, 1, len);

    // 归一化
    LocalTensor<float> outF32;
    Div(outF32, expF32, sumBroadcasted, len);

    Cast(output, outF32, RoundMode::CAST_NONE, len);
}
```

## 关键注意事项

| 事项 | 说明 |
|------|------|
| **绝不用 GetValue(0)** | 会强制 CPU/Scalar 等待 Vector，打断流水线。应用 Brcb |
| **FP32 中间累加** | ReduceSum 在 FP16 下有大数吃小数现象（数据量 >1000 时明显） |
| **FP16 Exp 溢出** | 输入 > 12 即溢出。未做 Max-Shift 结果 = NaN |
| **防除零** | 归一化时加 `1e-12f` |
| **Brcb 参数** | `Brcb(dst, src, 1, len)` 第3参数=1（只广播第0个元素） |

## Online Softmax（长序列分块）

当一行数据超出 UB 容量时，分块 + Online 更新全局 max/sum：

$$m_{new} = \max(m_{old}, m_{block})$$
$$l_{new} = l_{old} \cdot e^{m_{old}-m_{new}} + l_{block} \cdot e^{m_{block}-m_{new}}$$

## Masked Softmax（Attention 场景）

- **因果掩码**（Decoder）：直接索引判断 `col > row` → 跳过，无需传 mask 张量
- **布尔掩码**：mask=false 位置映射为 -∞，走标准 Softmax
- **优化**：掩码应用融合到 Max Reduction 阶段，整体仅需遍历两次

## 代码仓中的 SoftMax 变体

Attention 算子中使用的 `SoftmaxFlashV2`：
- 支持 `softmaxReduceSize` 机制：当 s1BaseSize > 256 时 reduceSize=1，否则为 8
- `SOFTMAX_REDUCE_CFG` 非标准 reduce 配置 + softmaxTempBuf Brcb
- `AA_INVALID_LINE_HIGH_PRECISION` (ImplMode=2)：Softmax 后 AdjustSoftMaxRes 精度修正
