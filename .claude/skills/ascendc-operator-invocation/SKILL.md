---
name: ascendc-operator-invocation
description: 调用已编译好的 AscendC 算子的几条主通路与完整调用流程。调用通路矩阵：PyTorch binding / aclnn eager / aclnn graph / GE graph / build.sh --run_example 的 eager/graph 验证入口。当要写 aclnn 单算子调用测试代码、判断某需求覆盖哪些调用通路、用 graph/eager 样例快速验证、把算子接入 PyTorch/TorchAir、或区分 host 层到底是哪条通路出问题时调用。注意：本 skill 讲“怎么调用/哪条通路在起作用”；编译构建看 ascendc-install，编译/安装/checker 失效排查看 ascendc-build-errors（落地后回链）。
---

# AscendC 算子调用通路

本 skill 不再把“调用方式”只粗分成三类，而是显式按**调用通路矩阵**来理解当前工作区：

## 调用通路矩阵

| 通路 | 典型位置 / 入口 | 适用场景 |
|---|---|---|
| **PyTorch binding** | `ops-transformer_AI/torch_extension/`, `op-plugin/` | 面向框架集成、PyTorch / TorchAir 用户侧调用 |
| **aclnn eager** | `test_aclnn_*.cpp`, `build.sh --run_example <op> eager` | 单算子快速验证、Host 侧两段式调用 |
| **aclnn graph** | Host 方案里必须显式判断是否覆盖；在当前仓里常与“图模式”讨论交织出现 | 需要在方案里单独判断支持面，不能默认等同 eager |
| **GE graph** | `test_geir_*.cpp`, `op_graph/`, 图模式构图调用 | 图编译 / 图执行 / GE 侧集成 |
| **验证入口** | `build.sh --run_example <op> eager|graph` | 用现成样例快速跑通 eager/graph |

> 注意：官方文档常把若干图形态笼统写成“图模式”。在当前工作区讨论 host 方案时，**要显式区分 PyTorch binding、aclnn eager、aclnn graph、GE graph**，并在方案里写清楚本次支持哪些通路、哪些 scope out。

## 当前真实来源（本次重构已核对）

- `ops-transformer_AI/docs/zh/invocation/quick_op_invocation.md`
  - 官方调用总览、`build.sh --run_example` eager/graph、aclnn 两段式、GE 图模式
- `ops-transformer_AI/build.sh`
  - `--run_example` 的真实命令分支；eager/graph 两种样例编译/执行入口
- `ops-transformer_AI/torch_extension/README.md`
  - PyTorch JIT builder / C++ wrapper / TorchAir graph mode 开发入口
- `ops-transformer_AI/docs/zh/develop/graph_develop_guide.md`
  - 图模式交付件、`op_graph` 路径、GE 注册侧要求
- `op-plugin/examples/README.md`
  - PyTorch 侧 extension / TORCH_LIBRARY_IMPL / pybind 适配样例
- `ops-nn/docs/QUICKSTART.md`
  - `test_aclnn_*.cpp` 样例与 `build.sh --run_example ... eager ...` 的工程侧验证路径

## 先查哪个明细

- 想快速判断本次需求覆盖哪些通路 → 看 [调用通路与真实来源](quick_op_invocation.md#1-调用通路与真实来源)
- 想直接用现成样例跑 eager/graph → 看 [build.sh 快速验证入口](quick_op_invocation.md#2-buildsh-快速验证入口)
- 想自己写 C/C++ 单算子调用 → 看 [aclnn eager 两段式](quick_op_invocation.md#3-aclnn-eager-两段式调用)
- 想理解图模式交付件和 GE 构图调用 → 看 [GE graph 通路](quick_op_invocation.md#4-ge-graph-通路)
- 想理解 PyTorch / TorchAir 侧接入 → 看 [PyTorch binding / TorchAir graph](quick_op_invocation.md#5-pytorch-binding--torchair-graph)

## 与其他 skill 的边界

- 编译 / 安装 / build.sh 包结构 → `ascendc-install`
- dtype / format / quant / broadcast / transpose 语义 → `ascendc-data-context`
- host 工程链为什么没生效 / checker 报错 / stale package → `ascendc-build-errors`
- graph/eager 路径都通了以后做语义对拍 → `ascendc-golden-testing`

本 skill 只负责：
> **把“当前到底是哪条调用通路在工作、怎么调用、怎么快速验证”讲清楚。**