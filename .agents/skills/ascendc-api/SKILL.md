---
name: ascendc-api
description: 查询 AscendC 算子开发的 API 用法、函数原型、参数与约束。涵盖数据搬运 DataCopy/DataCopyPad、矩阵结果搬出 Fixpipe、流水线同步 SetFlag/WaitFlag（含 A5 CrossCoreSetFlag 跨核同步）、核心数据结构 GlobalTensor/LocalTensor/TPipe/TQue/TBuf、高阶 API MatMul/SoftMax/Activation 激活函数、以及 Kernel 调试 PRINTF/DumpTensor。当需要写 Kernel 代码、查某个 AscendC API 怎么调、它的参数结构体/模板配置/对齐约束、或排查 API 用法错误时调用。（本 skill 只讲 API 怎么调；用 PRINTF/DumpTensor 定位数值差异、调试卡死或分析性能见 ascendc-debug）
---

# AscendC API 速查

本 skill 收录 AscendC Kernel 开发中最常用 API 的函数原型、参数结构体、关键约束和 A2A3/A5 差异。写 Kernel、查 API 调用方式或排查 API 用法问题时，按下表选对应明细文档阅读。

## 明细文档

### 基础数据结构
- [GlobalTensor / LocalTensor / TPipe / TQue / TBuf](GlobalTensor与LocalTensor与TPipe与TQue与TBuf.md) — 五大核心数据结构的位置、作用、分配释放、TPosition 逻辑位置、三阶段 CopyIn/Compute/CopyOut 范式。

### 基础 API
- [DataCopy 与 DataCopyPad](DataCopy与DataCopyPad.md) — GM↔Local 数据搬运、DataCopyParams 结构体、32B 对齐约束、非对齐 Pad 搬运、A5 ND→NZ 随路转换与 L1→GM 通路删除。
- [Fixpipe](Fixpipe.md) — L0C 结果搬出到 L1/UB/GM，随路量化/ReLU/类型转换，FixpipeParamsV220(A2A3) vs FixpipeParamsC310(A5) 与数据通路差异。
- [SetFlag 与 WaitFlag 同步](SetFlag与WaitFlag同步.md) — 同核内流水线同步、HardEvent 事件类型表、三阶段流水模式、A5 CrossCoreSetFlag/CrossCoreWaitFlag 跨核同步与死锁规避。

### 高阶 API
- [MatMul 高阶 API](MatMul高阶API.md) — 四步使用法、MatmulConfig 模板（Norm/MDL/IBShare/BasicBlock）、双缓冲、数据通路与对齐约束、A5 MatMul V3。
- [SoftMax 高阶 API](SoftMax高阶API.md) — Max-Shift 数值稳定、全向量化实现、Brcb 广播替代 GetValue、Online Softmax 长序列分块、Masked Softmax、SoftmaxFlashV2 变体。
- [Activation 激活函数](Activation激活函数.md) — 内置激活函数表、GELU/SwiGLU、多项式霍纳法则拟合、FP16→FP32 中间精度、与 MatMul 的融合。

### 调试 API
- [PRINTF 与 DumpTensor](PRINTF与DumpTensor.md) — Kernel 内标量打印与 Tensor dump、核号限流、1MB 空间约束、MetricsProf 打点、CPU→NPU→msprof 调试路线。
