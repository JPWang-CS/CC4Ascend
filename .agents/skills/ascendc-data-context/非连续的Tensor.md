# 非连续的 Tensor

## 概述

大部分算子 API 的输入 `aclTensor` 支持**非连续 Tensor**，通过 `(shape, strides, offset)` 三元组表示。

## 定义

- **strides**：描述 Tensor 维度上相邻两个元素的间隔
  - stride=1 → 该维度连续
  - stride>1 → 该维度非连续（元素间有空隙）
- **offset**：Tensor 首元素相对基地址的偏移

## 示例

shape=(6, 5), strides=(10, 1), offset=22

```
dim0 stride=10: 相邻行间隔 10 个元素（非连续）
dim1 stride=1:  同行内连续
```

shape=(4, 3), strides=(20, 2), offset=22

```
dim0 stride=20: 相邻行间隔 20 个元素（非连续）
dim1 stride=2:  同行内间隔 1 个元素（非连续）
```

## 创建方式

通过 `aclCreateTensor` 接口创建时指定 shape/strides/offset，参见[《算子库接口》](https://hiascend.com/document/redirect/CannCommunityOplist)中"公共接口"。

## 编程注意事项

- aclnn API 默认处理 ND 连续张量
- 非连续 Tensor 在 aclnn 内部会做 stride 感知的访问
- A5 ND DMA 支持随路 ND→NZ 转换，处理非连续数据更高效

## 来源
- `ops-transformer_AI/docs/zh/context/非连续的Tensor.md`
