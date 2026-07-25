---
name: omni-aiinframatmul-migration
description: omni-ops AiInfraMatmul 是 ops-nn mat_mul_v3 的迁移改名；fp32+NZ 精度问题的排查坐标与已知差异
metadata:
  type: project
---

omni-ops `AiInfraMatmul` 算子 = ops-nn `mat_mul_v3` 的迁移+改名（非量化 fp16/bf16/fp32 matmul，支持 WeightNz）。文件 1:1 对应：base_block/base_kernel/cvp_base_kernel/l1_full_load/nd2nz*/sc_splitk_block/simplifiedkey/base_tiling/tiling_data。**排查 omni 算子 bug 一律以 ops-nn mat_mul_v3 为 ground truth 对拍。**

**Why:** 2026-07 修 fp32(ND self)×fp32(FRACTAL_NZ weight) 在 910B 上结果全错（fp16/bf16 正常）。migration commit `7678e34c "强制走NDND或NDNZ格式"` 重写了 common/matmul_util.cpp 的 SetMmSupportFormat + aclnn；`9493646d "放开拦截"` 只放开了 aclnn fp32+NZ 的 dtype 拦截。

**How to apply（对拍已确认的迁移差异，排查 omni matmul 类算子先看这些）：**
- `common/op_host/op_api/matmul_util.cpp::GetMatMulOp`：ops-nn 有 `CheckSupportInfoFormatNdNzNd` + 在 fall-through 到 `MatMulV3Nd` 前做 `x2=ReFormat(x2,FORMAT_FRACTAL_NZ)`；omni 迁移时**丢了**这一支（只有 NzNzNd）。fp32+NZ 修复即补回此支。ReFormat 把 x2 的 view/storage/original format 全刷成 NZ，保证 op 节点 mat2 输入 format=NZ→选中 NZ binary(FORMAT_X2=29)→kernel format_x2=CubeFormat::NZ。此为**只能上板证伪**的属性（TransData/binary 选路在 CANN 框架内，本地无法 trace）。
- `SelectNZTiling`：omni 条件 `bFormat==NZ && bType!=DT_FLOAT`；ops-nn 条件 `isNZNZ`(A且B都NZ)。语义不同但对 ND×NZ 两者都不强制 BASE，故非本 case 根因。
- kernel `ai_infra_matmul.cpp` dispatch：omni 新增 `IS_ND_NZ_FP32` 宏(ops-nn 无)。`#elif FORMAT_X2==FRACTAL_NZ && !IS_ND_NZ_FP32`：fp16/bf16-NZ 走受限 `#elif`(仅 BASE/K_SHIFT)；fp32-NZ 走全量 `#else`。omni `#else` 比 ops-nn 精简很多(缺 SingleCore/MultiCore/Deterministic splitK 模板)，且**所有分支都要求 FIXOPTI==BASE_FIXOPTI**——若 tiling 出 BASE_ENABLE_ALIGNOUT/VEC_NZ2ND_UNALIGNOUT(见 DoBL1FullloadWithFixpipeTiling/NeedSolveFixBound)会落到末尾 else 用错配置。**残留风险**：N<256 且 K<=256 的 fp32+NZ 会触发该路径，omni kernel `#else` 无对应模板→可能仍错。当前测试 K=8192 不触发。
- aclnn `GetWeightNzShape` fp32 用 c0=8(`NZ_K0_VALUE_32=8`, 命名反直觉)，omni 与 ops-nn 一致；shape 解析(GetShape/GetInputDims 从 OriginShape 取 K、output 取 N)一致。
- debug_log.md 的 "isNzB=0" 是**误导**：args_.isNzB 是"ND 内轴==c0 且外轴 16 对齐时可当 NZ 传入"的优化标志(ops-nn GetMoreMultiCoreSplitKArgs 设)，**不是** B 真实 NZ 格式的标志。真 NZ 靠 FORMAT_X2 编译宏。故 isNzB=0 对真 NZ 输入是正常的。

测试：`D:\Desktop\Code\CC4Ascend\projects\MM-debug\test.py` (M128 K8192 N128 fp32, b_nz=npu_format_cast(b,29))。
