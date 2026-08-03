---
name: qbmm-a8w4int-nz-cast-timing
description: QBMM a8w4-int NZ 入口走不通的真因是 isA8W4Msd 在 ProcessScaleTensor cast 前校验 UINT64, 非"缺 cast"; ND 入口 cast 在 dtype 检查前故能放行
metadata:
  type: project
---

QBMM-omni a8w4-int (int8×int32-packed-int4 + UINT64 x2Scale + FLOAT pertoken + FLOAT yOffset) **必须走 V5 ND 入口** (`aclnnAiInfraQuantMatmulV5`), 不能走 WeightNz (`aclnnAiInfraQuantMatmulWeightNz` = `aclnn_ai_infra_quant_matmul_v4.cpp`).

**真因 (差异归因, 修正 host-engineer 初判 "缺 INT64→UINT64 cast"):**
- WeightNz 入口**有** INT64→UINT64 cast (`aclnn_v4.cpp:1640` `ProcessScaleTensor`, 调用点 :1833), 但它在 `CheckParams`(:1828, 含 `isA8W4Msd` 校验)**之后**才跑.
- `isA8W4Msd` (`quant_matmul_v4_common.h:144`) **严格要求** `scale->GetDataType()==DT_UINT64`, 校验时 scale 仍是 binding 传来的原始 INT64 → 返 false → yOffset 被 `aclnn_v4.cpp:1989` 拒 (非 EH0012 版本限制, A2A3 MSD 本身支持 yOffset).
- V5 ND 入口 (`aclnn_quant_matmul_v5.cpp:532-534`) INT64→UINT64 cast 在 dtype 检查**之前**, 且 `:277` `!isA8W4Int` 放行 yOffset → a8w4-int 可通.

**binding 路由 (`npu_ai_infra_quant_matmul.cpp:246-252`):** `else` 分支 `(!is_a4w4 && is_nz_format(x2)) ? WeightNz : V5`. weight_nz=False → `is_nz_format(x2)=false` → V5 ND. 路由纯看 format 不看 dtype.

**V5 ND a8w4-int perchannel 约束 (全 trace 确认):** x1Scale(pertoken)=FLOAT[M]1D(:63), x2Scale(scale)=UINT64[N]1D(via INT64 cast), yOffset(offset)=FLOAT[N]1D**必非空**(:528), bias/yScale/x2Offset=null(:524), groupSize=0(:180), K%2==0(:711 `SUPPORTED_K_ALIGN_NUM_INT4=2`), 仅 910B/910_93(DAV_2201, :744).

**golden.py 落地 (2026-07-30):** 新增 `v3_a8w4int_nd` (M=N=128,K=256,out=fp16,enabled=True); `v3_a8w4int_nz` 保持 disabled + 注释更新为 cast 时序真因 (删旧 EH0012 证伪注释). gen_data/call_npu a8w4 函数体不改 (已是 perchannel 布局, 匹配 V5 ND). oracle `golden_ref:511-520` MSD 公式 `mm × x2Scale[N] × x1Scale[M] + yOffset[N]` 与 numpy naive max|Δ|=0 精确匹配.

**首上板怀疑点:** V5 ND 运行时是否真接受 a8w4 yOffset 源码放行 (`:277/:528`) 未上板验证, 待用户服务器验.

**Why:** host-engineer 初判"WeightNz 缺 cast"不准确 — cast 存在但时序错 (CheckParams 之后). 差异归因须读调用点时序不能只看 cast 函数定义.
**How to apply:** 后续遇"a8w4 NZ 走不通"类问题, 先查 isA8W4Msd 校验时序 vs cast 调用时序, 别假设 cast 缺失; a8w4-int 统一走 V5 ND.

相关: [[qbmm-pergroup-int4-kg-semantics]] (同 V4 核, 走 V5 ND).
