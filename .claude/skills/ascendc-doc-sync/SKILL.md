---
name: ascendc-doc-sync
description: AscendC 算子规格/语义/数据布局变更时的跨仓文档同步排查清单（经验型）。涵盖 ops-nn（proto.h 注册注释 / 一个 dir 多份 aclnn .md / README / op_api·op_host·op_kernel 注释 / examples 数据构造 / fusion_pass）、ops-tensor（kernel 设计文档与源码注释）、op-plugin（torch_npu 对外 doc）、ops-transformer（官方规格 docs/zh）、项目工作目录文档。当算子改了输入/属性/shape/dtype/量化语义/数据布局/转置规则，需要排查"哪些文档要跟着改"、防止只改一处漏掉兄弟仓或同目录多份 aclnn doc 时调用（典型漏改：aclnn .md 同步了、proto.h 注释却漏）。
---

# 算子变更跨仓文档同步清单

本 skill 是经验型清单：算子规格/语义变更后，同一份变更会散射到多个仓的多个独立文档产物。按明细文档逐仓逐产物排查，避免漏改。

> 核心教训：proto.h 注释 ≠ aclnn .md ≠ README ≠ ops-tensor kernel doc，同步其一不等于同步其余；一个 op_dir 可含多份 aclnn doc；ops-tensor/op-plugin 是兄弟仓易漏。

## 明细文档

- [算子文档同步排查清单](operator-doc-sync-checklist.md) — 5 仓 × 产物类型矩阵（路径模式 + 何时改）、对外文档决策策略、项目文档 stale 陷阱、典型漏改模式、QBMM MX scale-batch 漏改案例。
