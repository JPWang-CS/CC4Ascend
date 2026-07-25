# MM-fp32nz-aclgraph 项目

> matmul fp32 + NZ 权重输入，迁移到 omni 仓，需兼容 ACLGraph 模式。

## 问题链

1. **根因**：主线 nn 仓 `npu_transform_weight`（WeightFormatCast）**不支持 fp32→NZ 转换**
2. **主线后果**：torch 侧 fp32+NZ 走不了 WeightNz → 回退 `aclop`（ND 计算）
3. **omni 迁移困境**：omni **没有 aclop 可调用**，不能像 nn 那样回退
4. **朴素解法的坑**：torch 侧把 fp32 NZ 转 ND 再算 → **ACLGraph 模式出问题**（ReFormat 在 capture/replay 不兼容）

## 约束

- 目标 SoC：A2/A3（arch32）
- B（权重）NZ，外部预打包
- 必须 fp32 原生精度，不可降精度（cube=2/USE_FP16 已否决）
- 需兼容 ACLGraph（推理交付）

## 已查证事实（前期 trace）

- c0 是 dtype 自适应：GetSizeC0 fp32→8/fp16→16/int8→32（matmul_v3_common.h:243）
- nn 仓 aclnn 路径（aclnn_matmul.cpp:98）在 A2(DAV_2201) 放行 fp32+NZ
- op-plugin eager 默认（非 AclnnOnly）fp32+NZ 走 aclop ND（DO_MATMUL_COMPATIBILITY，op_api_common.h:586）
- omni WeightNz host 守卫（aclnn_ai_infra_matmul.cpp:94）无条件拒 fp32+NZ（删了 nn 的 arch 守卫）
- ACLGraph 机制见 skill ascendc-aclgraph

## 工作计划

### Phase 1：golden 验证 nn 仓 ACLGraph 下 fp32+NZ 真实行为
- 写 golden 脚本，板上跑 nn 仓 matmul fp32+NZ 在 ACLGraph（裸/npugraph_ex）下的行为
- 观察：走 aclop ND？NZ kernel？报错？结果对不对？
- fp16+NZ 作对照组（已知 work）
- **目的**：建立 omni 迁移的参照基线

### Phase 2：据 Phase 1 真实现象，分析 omni 迁移方案
- 若 nn ACLGraph 下 fp32+NZ 走 aclop（omni 没法照搬）→ 需新方案
- 若 nn ACLGraph 下 fp32+NZ 报错/挂 → nn 自己都没解，omni 需自创路径

## 文件
- `README.md` — 本文件
- `golden/` — golden 验证脚本（Phase 1）