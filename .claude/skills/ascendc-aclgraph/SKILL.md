---
name: ascendc-aclgraph
description: AscendC 算子入 ACLGraph 的交付件要求与机制。ACLGraph 对标 CUDAGraph 的 Capture&Replay（基于 kernellaunch 捕获执行序列后重放，解决 HostBund 瓶颈）。覆盖：入图两种方式（裸 ACLGraph / torch.compile+npugraph_ex）、meta 接口、host 侧 tiling 更新算子的额外接口（get_max_workspace/.out/_infer_output）、静态 kernel 编译、SuperKernel 融合与限制。当算子要进 ACLGraph、写 meta 接口、做静态 kernel、配 SuperKernel、或排查入图失败时调用。源：华为 wiki WIKI2026040910715297（已落 D:\Desktop\TMP\log.txt）。
---

# 算子入 ACLGraph

本 skill 讲 AscendC 算子入 **ACLGraph** 的机制与交付件要求。真实源：wiki WIKI2026040910715297。

## ACLGraph 是什么

**ACLGraph** 对标 CUDAGraph 的 Capture & Replay：
- **Capture**：基于 kernellaunch 捕获模型执行流程（内存分配、算子执行等），记录进执行图
- **Replay**：重放已捕获的图，避免重复构建/编译/分配
- 目的：解决单算子 HostBund 瓶颈，提升推理性能

> 与 GE Graph 不同：ACLGraph 是 kernellaunch 级 Capture/Replay，不是 GE 的 IR 构图。推理算子入图分 GE Graph 和 ACL Graph 两条路，本 skill 只讲 ACLGraph。

## 明细文件

- [入图方式与 meta 接口](entry-and-meta.md) — 裸 ACLGraph / torch.compile+npugraph_ex 两种方式、meta 接口注册与实现
- [tiling 更新算子额外接口](tiling-update-op.md) — host 侧 tiling 更新算子（FIA 类非 AICPU 下沉）的 get_max_workspace / .out / _infer_output + npugraph_ex 注册
- [静态 kernel 编译](static-kernel.md) — capture 时编死、验证方法（profiling/plog）
- [SuperKernel 融合](superkernel.md) — 融合范围、optimize/debug 选项、不可融原因表、aclnn/SIMT 限制

## 快速定位

| 需求 | 看 |
|---|---|
| 算子怎么进 ACLGraph | entry-and-meta.md |
| 算子 tiling 动态变化（FIA 类） | tiling-update-op.md |
| 静态 kernel 没生效/报错 | static-kernel.md |
| SuperKernel 融合失败/排查 | superkernel.md |

## 交付件总览（推理算子入图）

入 ACLGraph 相关交付件（区别于基础算子交付件）：
- **meta 接口**（torch.compile 路径必需）：device="meta" 的 shape 推导
- **get_max_workspace / .out / _infer_output**（host 侧 tiling 更新算子必需）
- **npugraph_ex tiling 更新注册**（npugraph_ex 完整后端入图时）
- 静态 kernel 编译 + SuperKernel 支持（默认要求，ST 用例须覆盖）

## 边界

- 基础算子交付（op_def/tiling/kernel/aclnn）→ ascendc-development
- 单算子调用通路（eager/aclnn 直调）→ ascendc-operator-invocation
- 本 skill 只讲 ACLGraph 入图这一层