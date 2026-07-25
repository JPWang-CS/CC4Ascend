---
name: ascendc-kernel-programming
description: AscendC kernel 编程范式——怎么从空文件写成一个完整 kernel。覆盖 A2A3 Membase（TPipe/TQue/三阶段 CopyIn-Compute-CopyOut）、A5 Regbase（MicroAPI/RegTensor/__simd_vf__）、A5 SIMT（__simt_vf__ 线程级）、高阶 API（Matmul<>/SoftMax<> 模板）、ops-tensor Blaze（CUTLASS 式 kernel/block/tile/epilogue）。每种范式给真实代码骨架（trace 自 ops-transformer_AI examples 与 ops-tensor/include/blaze）。当要新建 kernel、写 kernel 骨架、选编程范式、或查某范式怎么写时调用。（本 skill 讲"怎么组装 API 写 kernel"；单个 API 原型见 ascendc-api；优化偏离见 ascendc-kernel-optimization）
---

# Kernel 编程范式

本 skill 教怎么写 kernel——把 ascendc-api 里的单个 API 组装成完整 kernel。每种范式给真实代码骨架。

## 范式决策树

```
要写的 kernel 是？
├─ elementwise / reduce / 简单向量计算
│   ├─ A2A3 → Membase（TQue 三阶段）
│   └─ A5   → Regbase（MicroAPI RegTensor）或 atvoss DAG
├─ 矩阵乘类（MatMul / GMM / 量化 matmul）
│   ├─ A2A3 → Membase + 高阶 Matmul<> 模板
│   └─ A5   → Blaze（kernel/block/tile/epilogue）或 高阶 Matmul<>
├─ 离散访存（gather/scatter/embedding）
│   └─ A5 → SIMT（__simt_vf__）
├─ 通算融合（MC2 通信+matmul）
│   └─ Blaze + CCU
└─ 不确定 → 先看 membase（最基础），再按芯片/算子类选
```

## 范式清单（按层）

- [Membase（A2A3 经典）](membase.md) — TPipe/TQue/TBuf + CopyIn/Compute/CopyOut 三阶段，add_example 完整骨架
- [Regbase（A5）](regbase.md) — MicroAPI/RegTensor/__simd_vf__/__VEC_SCOPE__，寄存器级编程
- [SIMT（A5）](simt.md) — __simt_vf__ + GetThreadIdx/GetThreadNum，线程级离散访存
- [高阶 API 模板](highlevel-api.md) — Matmul<>/SoftMax<>/Activation，组合式
- [Blaze（ops-tensor）](blaze.md) — CUTLASS 式 kernel/block/tile/epilogue + DispatchPolicy，matmul 类专用
- [量化 elementwise](quant-elementwise.md) — dequant→compute→requant 骨架，Cast/SatMode/buffer 尺寸/饱和

## 范式定位（与相邻 skill 边界）

| skill | 讲什么 |
|---|---|
| ascendc-api | 单个 API 原型/参数（DataCopy、SetFlag...） |
| **本 skill** | **怎么把 API 组装成 kernel（范式骨架）** |
| ascendc-development | 开发流程 + A2A3→A5 迁移方法论 |
| ascendc-kernel-optimization | 写好后怎么优化（base 模板 + 偏离） |

## 真实源

- Membase 骨架：`ops-transformer_AI/examples/add_example/op_kernel/add_example.h`
- Regbase/SIMT：`ops-transformer_AI/attention/*/op_kernel/arch35/`、`posembedding/*/op_kernel/arch35/`
- Blaze：`ops-tensor/include/blaze/`（header-only）+ `include/tensor_api/`（Layout/Shape/Coord）