---
name: qbmm-pergroup-int4-kg-semantics
description: QBMM K-G int4×int4 pergroup 非对称量化的精确公式/int4打包方向/转置处理; 公式源是 kernel 不是 ops-nn golden
metadata:
  type: project
---

QBMM-omni V4 K-G int4×int4 pergroup 非对称量化 golden 实现要点 (2026-07-30 落地于 tests/golden.py `_compute_pergroup_kg`):

**公式 (权威源 = ops-nn op_kernel/quant_batch_matmul_v4_pergroup.h:296-398, 非 aclnn doc 非 ops-nn golden)**:
```
acc[M,N] = 0
for t in [0, K/256):
    mm_t  = matmul(x1[:, t*256:(t+1)*256], x2[t*256:(t+1)*256, :])   # int32 精确累加→fp32
    rs_t  = sum(x1_fp32[:, t*256:(t+1)*256], axis=-1)                # [M] x1 K-tile 行和
    acc += (mm_t - rs_t[:,None] * x2Offset[t,:]) * x2Scale[t,:]      # offset 是减性偏移!
out = acc * x1Scale[M,1]    # x1Scale 最后整体后乘 (Brcb)
```
关键: x2Offset 是**减性** (mm - rs×offset), 不是加性; x1Scale 是最后整体乘不是 per-tile。

**Why**: ops-nn v4/tests/assets/golden.py **没有** int4×int4 pergroup 实现 (只有 fp8×fp4 的 _compute_t_cg 和 int8×int8 的 _compute_per_tile_int8, 公式都不同)。aclnn V5.md L715-743 只给约束不给公式。公式必须 trace kernel 源码确认。

**How to apply**: 写/改 pergroup golden 时, 公式以 kernel pergroup.h 为准 (RowSum/Broadcast/Mul/Sub/Mul/Add/Mul 那串); 不要照 fp8×fp4 的 _compute_t_cg (vcvt+yScale 后乘完全不适用 int4)。

---

**语义陷阱 (均已在 golden.py 处理)**:

1. **x1 rowsum 的 cast 路径**: kernel line 291-296 是 int4→fp16(half)→fp32。int4 值域 -8..7 可被 fp16 精确表示 → `int4→fp16→fp32 == int4→fp32` 无损。golden 直接 `x1.astype(np.float32)` 等价, 不必模拟 fp16 中转。

2. **int4 打包方向 (trans_x2=true 时沿 K 打包, 非 N)**:
   - x1_packed = pack_int4_lastdim(x1) → [M, K/8] (沿末维 K 打包, 和 int4sym 一致)
   - x2_packed = pack_int4_lastdim(x2.T) → [N, K/8] (**x2 先转置再沿 K 打包**)
   - 原因: kernel line 457 `x2Global_[(offsetB + kidx*groupSizeK)>>1]` 沿 K 线性步进 → K 是最内维 → 沿 K 打包。
   - binding line 161-162: trans_x2=true → N = x2.size(-1)。x2_packed=[N,K/8] 传 NPU 后 .transpose(-1,-2) → 视图 [K/8, N], binding 读 size(-1)=N ✓。
   - **和 int4sym (trans_x2=false) 不同**: int4sym 沿 N 打包 (x2=[K,N]→[K,N/8]), binding 读 N=size(-1)*8。

3. **transpose 处理 (照 perblock 先例)**:
   - call_npu: x2.transpose(-1,-2), scale.transpose(-1,-2), offset.transpose(-1,-2) (trans_x2=true 时)
   - 转 stride 让 binding is_x_scale_same_transpose 检查通过 (x2↔scale 转置一致性)
   - aclnn V5 PreMatmulCalcProcess (aclnn_quant_matmul_v5.cpp:996/999) 会 contiguous 化 scale/offset, 故 transpose 最终被物化
   - **binding 不检查 x2↔offset** (只查 x2↔scale 和 x1↔pertoken_scale), 但 offset transpose 与 scale 同构处理最稳

4. **早返回避 matmul 维度报错**: pergroup M≠K (128≠1024), 若走 golden_ref 外层预计算 mm (swap x2 后 matmul) 会维度不符。已把 perblock/pergroup 提前 dispatch 到 _compute_perblock_gb/_compute_pergroup_kg 独立函数, 跳过外层 mm。顺带修了 perblock 非方阵的潜在同 bug (B3 受益)。

**约束 (pergroup_tiling.cpp 硬校验)**: K%1024==0, N%256==0, gsK=256(硬编码), bias=null, trans_x2=true(强制), x1Scale=[M,1]fp32, x2Scale=[⌈K/256⌉,N]fp32, x2Offset=[⌈K/256⌉,N]fp16, out=bfloat16。

**未上板验证**: 本机无 torch/numpy, 只做静态 trace + 语法编译。上板首怀疑点: scale transpose 后 tiling 读 shape 是否接受 (perblock 先例背书, 但 pergroup scale shape=[nkgroup,N] vs perblock=[gKtile,gNtile] 结构同构)。见 [[qbmm-omni-migration-progress]]。
