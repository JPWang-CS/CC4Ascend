---
name: qbmm-mx-batchA-mfusion-guard
description: QBMM V3 A5 CheckFusionBatchA 会把 batchA 合入 M(A_batch-B_no_batch,transA=false),导致 batchC=1→走 MX WITHOUT_BATCH kernel(TilingKey KERNELTYPE=9);MX 必须保留 IsMicroScaling() 禁合轴守卫
metadata:
  type: project
---

# QBMM V3 (A5/950) MX 场景 batchA→M 合轴与 TilingKey 的交互

**结论**：`quant_batch_matmul_v3_tiling_base.cpp::CheckFusionBatchA` 在满足 `x2 为 2D(B 无 batch) && transA==false && x1 有 batch前缀 && !isPertoken` 时会把 batchA 合入 M(`DoBatchFusion`: mSize*=batchA, batchC=1)。MX pergroup 场景 `SetQuantMode` 会把 `isPertoken` 置 false、`isMxPerGroup` 置 true(即使外部传了 pertokenScale;line 546 覆盖 AnalyzeInputs line 566),所以 isPertoken 守卫拦不住 MX。合轴后 batchC=1 → `GetKernelType` 里 `isMxWithoutBatch = IsTensorapiCapable() && isMxPerGroup && batchC==1` 变 true → 选 WITHOUT_BATCH kernel(KERNELTYPE 9/10/13/14)。

**因此 MX 必须有独立的 `if (IsMicroScaling()) return 0;` 守卫禁止合轴**,否则 host 走 WITHOUT_BATCH 单矩阵路径,绕开 kernel 的批量 `AddBatchOffset`(scale 批量偏移,见 [[qbmm-batch-status]] 设计 §2.6),host/kernel 设计不一致。

**Why**: scale-batch 项目提交 `43793e5f6【QBMM】scale支持batch维度` 误删了 `CheckFusionBatchA` 里的 "// mx量化模式不能batch合轴 / if (IsMicroScaling()) return 0;" 三行,导致 `mxfp8-pergroup-A_batch-B_no_batch` UT(test_quant_batch_matmul_v3.csv 该行,期望 TilingKey=8196=KERNELTYPE 0)实际产出 10500=KERNELTYPE 9,`generalTest/89` 与 `multiThread950` 双挂(multiThread950 遍历全部 950 用例复跑同一 Test(),同因同案,非线程/确定性问题)。该守卫自 init(e11fe07f3)起长期存在。

**How to apply**: 改 QBMM V3 batch/合轴/MX tiling 时,先确认 `CheckFusionBatchA` 的 MX 守卫在位;调 A_batch-B_no_batch 类用例的 TilingKey 前先判断是否发生了 M 合轴。修复=在 CheckFusionBatchA 成功分支前恢复 `if (IsMicroScaling()) return 0;`。

## TilingKey 位域(A5 arch35,GET_TPL_TILING_KEY,LSB→MSB)
ATRANS[0:1] | BTRANS[2:3] | BIASMODE[4:7] | KERNELTYPE[8:11] | APILEVEL[12:13]。
MX 走 Blaze → APILEVEL=2。解码示例:8196=0x2004→(A0,B1,bias0,KT0,api2);10500=0x2904→KT9(WITHOUT_BATCH);11524=0x2d04→KT13(MX_L0C_PINGPONG_WITHOUT_BATCH)。KERNELTYPE 枚举见 quant_batch_matmul_v3_tiling_util.h(9/10=WITHOUT_BATCH,11-14=MX_L0C_PINGPONG)。
注意:CSV 里的 batchC 列只供测试建 shape 与选 tilingData 类型;host 真正的 inputParams_.batchC 由 InferOutBatchDim 从 shape 推(无 batch 归一为 1),合轴后强制为 1。CSV 的 caseName ".../104" 只是标签字符串,gtest 的 generalTest/NN 索引是该 socVersion 过滤后向量里的位置,二者不必相等。
