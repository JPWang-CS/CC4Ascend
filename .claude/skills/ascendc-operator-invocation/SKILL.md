---
name: ascendc-operator-invocation
description: 调用已编译好的 AscendC 算子的三种方式与完整调用流程。涵盖 PyTorch API、aclnn API（C 语言两段式：Init→CreateAclTensor→GetWorkspaceSize→aclrtMalloc→执行→同步→取结果→释放）、GE 图模式（构图调用）、以及 build.sh --run_example 快速调用（eager/graph）。当要写 aclnn 单算子调用测试代码、用 GE 图模式跑算子、把算子接入 PyTorch、或用 build.sh 快速验证算子时调用。注意：本 skill 是"调用算子"，编译构建算子见 ascendc-install。
---

# AscendC 算子调用方式

本 skill 说明如何调用一个已编译部署好的算子。写调用测试或集成时按下表查阅。

## 三种调用方式

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| PyTorch API | Kernel 注册到 PyTorch 框架 | Torch 推理/训练 |
| aclnn API | C 语言两段式接口（前缀 aclnn） | 快速调用、单算子验证 |
| GE 图模式 | 通过算子 IR 构图调用 | 图编译优化、多算子融合 |

快速调用：`bash build.sh --run_example ${op} eager cust [--soc=ascend950]`（图模式用 `graph`）。

## 明细文档

- [算子调用方式](quick_op_invocation.md) — 三种调用方式对比、build.sh 快速调用、aclnn 完整调用流程与 C++ 代码、GE 图模式调用流程、PyTorch 集成。
