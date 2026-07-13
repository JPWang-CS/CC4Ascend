---
name: ascendc-data-context
description: 查询 AscendC / aclnn 算子的数据语义与上下文规则。涵盖数据类型（FLOAT16/BF16/INT8/INT4/FP8/MXFP4 等及简写）、数据格式（ND/NCHW/NZ 分形格式）、基础数据结构（aclTensor/aclScalar/aclOpExecutor）、量化介绍（pertensor/perchannel/pertoken/pergroup/perblock/MX 量化）、broadcast 广播规则、类型互推导（Type Promotion）、Tensor-Scalar 互推导、互转换（Type Conversion）、非连续 Tensor（stride/offset）、两段式接口、以及 Attention 的 sparseMode 稀疏掩码模式。当需要确认某种 dtype/format 是否支持、量化粒度怎么选、shape 能否广播、类型如何推导转换、scale shape 怎么算、或 sparseMode 取值含义时调用。（本 skill 讲两段式的概念/数据语义；其 C 语言调用代码流程 Init→…→释放见 ascendc-operator-invocation）
---

# AscendC 数据语义与上下文规则

本 skill 收录算子输入输出的数据类型、格式、量化、广播、类型推导/转换、非连续张量、接口调用约定等"上下文规则"知识。确认数据语义、校验 shape/dtype 合法性、或设计量化 scale 时按下表查阅。

## 明细文档

### 类型与格式
- [数据类型](数据类型.md) — aclTensor 支持的数据类型简写表、推理/训练/量化常用类型、A5 新增 FP8/HIFLOAT8/MXFP8/MXFP4。
- [数据格式](数据格式.md) — ND/NCHW/NHWC/HWCN 等格式、维度约束、NPU 私有分形格式（NZ/FRACTAL_Z）、维度缩写。
- [数据结构](数据结构.md) — aclTensor/aclScalar/aclIntArray/aclOpExecutor 等基础数据结构与创建接口、aclOpExecutor 生命周期。

### 量化
- [量化介绍](量化介绍.md) — 静态/动态量化、量化粒度（T/C/K/G/B 量化）的 scale shape、常见组合量化（K-C、G-B、MX 量化 G-G）、A5 MXFP8/MXFP4。

### 类型推导与转换
- [互推导关系](互推导关系.md) — Tensor-Tensor 类型提升完整规则表（Type Promotion）。
- [TensorScalar 互推导关系](TensorScalar互推导关系.md) — Tensor 与 Scalar 类型不一致时的推导规则（Scalar 向 Tensor 靠拢）。
- [互转换关系](互转换关系.md) — 输出类型与计算类型不一致时的转换规则（Type Conversion）。

### 形状与排布
- [broadcast 关系](broadcast关系.md) — 广播三规则、合并维度 <6 的特殊类型限制。
- [非连续的 Tensor](非连续的Tensor.md) — (shape, strides, offset) 三元组、stride 语义、aclnn 非连续访问、A5 ND DMA。

### 接口与稀疏
- [两段式接口](两段式接口.md) — aclnn GetWorkspaceSize + 执行两段式调用模式、workspace、命名约定、二段不可重复调用。
- [sparse_mode 参数说明](sparse_mode参数说明.md) — Attention sparseMode 0-9 全模式速查（causal/band/prefix/varlen/treeMask）及参数约束。
