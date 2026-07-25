# Elementwise 计算范式

ops-nn activation 类（80 算子，gelu/silu/relu/sigmoid 等纯 Vector 计算）。内容经真实代码 trace。源：`ops-nn/activation/`。

## 范式特征

activation 是**最基础的 elementwise**：每个元素独立计算，无 reduce 无 matmul 无跨元素依赖。
- arch 分布：216 arch35 / 3 arch22（已验证，几乎全 A5）
- 算子：gelu / fast_gelu(+v2) / silu / relu / sigmoid / tanh / swish ...

## base 流水（elementwise 通用）

```
for each tile:
    CopyIn(tile)      # GM → UB
    Compute(tile)     # elementwise op（Add/Mul/激活函数）
    CopyOut(tile)     # UB → GM
```
双缓冲 TQue ping/pong 掩盖 MTE2/V/MTE3。

## A5 Regbase elementwise（趋势）
A5 activation 倾向 MicroAPI 寄存器级：
- `RegTensor` 替代 LocalTensor
- 激活函数用多项式霍纳法则拟合（gelu/silu）
- FP16→FP32 中间精度保证精度

## 常见优化
- **多项式拟合**：gelu 等用霍纳法则近似，减 exp 调用
- **中间精度**：FP16 输入升 FP32 计算再降回，避免精度损失
- **融合**：activation 常与前置 matmul/quant、后置 cast 融合（见 transformer 仓 attention quant、ops-nn dequant_swiglu_quant）
- **inplace 变体**：省输出 GM

## base 设计理由
elementwise 是所有算子的**底层构建块**：
- 无依赖 → 可任意并行（核间按 tile、核内按元素）
- 瓶颈在搬运（MTE2/MTE3）而非计算 → 优化重点是流水掩盖和数据通路

## gelu 实际结构（已 trace activation/gelu/op_kernel/arch35/）

gelu（A5）用两种组织：
- `gelu_struct.h`：TilingKey 模板 `ASCENDC_TPL_ARGS_DECL(Gelu, schMode[dType])`，schMode（MODE_0/1）+ dType（FP16/BF16/FP32）选分支
- `gelu_dag.h`：基于 **atvoss DAG 框架**（`atvoss/util/dag.h` + `atvoss/util/vec.h`），用 DAG 描述计算图，非传统 Kernel 类

> A5 新算子倾向 atvoss DAG 框架（声明式构图）替代手写 Process 三阶段。elementwise 在 DAG 里是一个 vec 节点。