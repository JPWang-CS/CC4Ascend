---
name: mm-fp32nz-aclgraph-project-scope
description: MM-fp32nz-aclgraph 项目阶段1只验 nn 仓 torch.matmul fp32+NZ 在 ACLGraph 下的真实行为，不动 omni fix 路径
metadata:
  type: project
---

项目 `D:\Desktop\Code\CC4Ascend\projects\MM-fp32nz-aclgraph\` 阶段1 的唯一目标：**验证 nn 仓 `torch.matmul(fp32_a, fp32_b_nz)` 在裸 ACLGraph / npugraph_ex 下的真实行为**——走 aclop ND 回退？走 WeightNz kernel？报错？结果对不对？建立 omni 迁移基线。

**Why:** omni 仓没有 aclop 可回退（nn 靠 aclop 兜底 fp32+NZ），omni 迁移 ACLGraph 时若 nn 自己都没在 ACLGraph 下解 fp32+NZ，omni 必须自创路径。这是 omni 迁移方案选型的前提，必须先有板上真值。

**How to apply:**
- **不要混淆**：[[omni-aiinframatmul-migration]] 记的是 omni 自定义算子 `torch.ops.custom.npu_ai_infra_matmul` 的 fix 链路（Fix 1-4 已 eager 上板通过），那条路径已 work，**不是**本项目阶段1的关注点。
- 阶段1 关注的是 `torch.matmul`（nn op-plugin 通路）在 ACLGraph 下的行为，**已知 eager 走 aclop ND 结果正确**（见 [[mm-debug-fp32-nz-cube2]]），未知的是 ACLGraph capture/replay 下的行为。
- 本机无 NPU、无 nn/op-plugin 源码；脚本由用户上板跑，方案只设计观测点与判据。
- 阶段2（omni 迁移方案）在阶段1有板上结果后才展开。
