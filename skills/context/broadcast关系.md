# broadcast 关系

## 概述

broadcast（广播）描述算子如何处理不同 shape 的张量运算：较小的张量自动"广播"为较大张量的形状。类似 NumPy broadcasting。

## 广播规则（按优先级）

### 规则1：维度数不一致 → 左侧补1

向最长 shape 的数组看齐，shape 不足的部分在左侧填充 1。

```
a.shape=(2, 3), b.shape=(2, 2, 3)
→ a 被 broadcast 为 (1, 2, 3)
```

### 规则2：某一维度为1 → 拉伸匹配

```
a.shape=(1, 3), b.shape=(3, 1)
→ a → (3, 3), b → (3, 3)
```

### 规则3：无1维度且不匹配 → 报错

## 广播过程示例

```
a.shape=(2, 2, 3)
b.shape=(2, 3)

Step 1 (规则1): b → (1, 2, 3)
Step 2 (规则2): b → (2, 2, 3)
计算 a+b ✓
```

## 特殊限制

当两个输入的数据类型或推导后类型属于以下集合时：
**COMPLEX64, COMPLEX128, DOUBLE, INT16, UINT16, UINT64**

除上述广播规则外，还需要：
- 连续的需广播轴 + 连续的不需广播轴合并后维度 < 6

**反例**：`a.shape=(5,1,5,1,5,1), b.shape=(5,5,5,5,5,5)` → 合并后维度=6 → 报错
**正例**：`a.shape=(5,1,5,5,1,1), b.shape=(5,5,5,5,5,5)` → 合并后维度=4 → 成功

## 来源
- `ops-transformer_AI/docs/zh/context/broadcast关系.md`
