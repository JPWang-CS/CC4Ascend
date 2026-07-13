---
name: mx-quant-scale-semantics
description: MX (microscaling) quant-matmul scale semantics on Ascend 950 — e8m0 dtype, (M,ceil(K/64),2) shape, groupSizeK=32, batch prefix, cube-internal per-group apply
metadata:
  type: reference
---

# MX quant-matmul scale semantics (Ascend 950 / A5)

General facts for the MX full-quant matmul path (`quantBatchMatmulV3` family). MX/fp4/fp8-microscaling is a 950-only feature — see [[cann-blaze-cmct-gating]].

## Scale dtype and shape
- Scale dtype = **FLOAT8_E8M0**, **1 byte, NEVER packed** (contrast fp4 data which is 2:1 packed — see [[fp4-e2m1-packing]]).
- Per-input scale carries a trailing MX "2" dim:
  - `scaleX1`: `(M, ceil(K/64), 2)`
  - `scaleX2`: `(ceil(K/64), N, 2)`
- Constants: `MXFP_DIVISOR_SIZE = 64`, `MXFP_MULTI_BASE_SIZE = 2` → `scaleKLen = ceil(K/64)*2`.
- **Group size over K = 32**: one scale value covers 32 contiguous K elements (NPU `group_sizes = [1,1,32]`). The `ceil(K/64)*2` layout = `ceil(K/32)` groups arranged as `(ceil(K/64), 2)`.

## Batch prefix (when inputs are batched)
- With batch, scale gains a leading batch dim matching the data tensor dim-by-dim (broadcast allowed):
  - `scaleX1`: `(batchA, M, ceil(K/64), 2)`
  - `scaleX2`: `(batchB, ceil(K/64), N, 2)`
- Scale dim constraint is `>= 3` (not `== 3`) once batch is allowed; batch dims must align with x1/x2.
- **Per-batch scale stride**: `scaleX1` = `M*ceil(K/64)*2`, `scaleX2` = `N*ceil(K/64)*2`.
- A batched MX matmul MUST add the batch offset to the scale GM pointer too, not just data/bias. Offset application differs by kernel path:
  - Blaze (tensor-api) kernel: raw-pointer `gmAddr_ += (...) >> sizeShift`; **scale lines never shift** because e8m0 = 1 byte.
  - CMCT (basic-api) kernel: element-indexed tuple `blockOffset_`, **no sizeShift** at all.
  - Both must keep the SAME logical formula or batch>0 silently misreads scale.

## Where scale is applied in the cube
- Scale is applied at the **element/group level INSIDE the matmul (Mmad, `MmadTraitMX` = `MmadType::MX`)**, NOT post-accumulation.
- Dataflow: data GM→L1→L0A/L0B; e8m0 scale GM→L1→**dedicated L0ScaleA / L0ScaleB** buffers; cube does fp4/fp8 decode + per-group scale-apply + accumulate internally. Accumulator / L0C is **float**. Bias added in L0C bias-table on first K-iter; output cast at Fixpipe (L0C→GM/UB).

## Golden modeling of scale (reference impl)
- e8m0 scale → `np.repeat(scale, 32)` broadcast over K → truncate odd `ceil(K/32)` → pad → `x *= per-scale` → matmul. fp4 and fp8 share this SAME scale path (scale handling is dtype-independent; only the DATA tensor packing differs).

See [[fp4-e2m1-packing]], [[scale-transpose-stride-semantics]], [[cann-blaze-cmct-gating]], [[qbmm-batch-status]].
