---
name: scale-transpose-stride-semantics
description: NPU MX quant-matmul requires x/scale transpose attributes match by STRIDE not shape (constraint A); checker logic + how to transpose a scale tensor correctly
metadata:
  type: reference
---

# MX quant-matmul transpose-attribute check is by STRIDE, not shape (constraint A)

Official constraint A: "MX 全量化下 x1/x1Scale 转置属性一致, x2/x2Scale 一致". The NPU enforces this by comparing **stride transpose-state**, NOT shape.

## How the checker decides "transposed"
- `is_transpose_certain_two_dims(t, dim)`: returns `t.stride(dim+1) == t.stride(dim) * t.size(dim)`. Pure stride test — a contiguous tensor → false; a tensor that went through `.transpose()` on those two dims → true.
- `is_x_scale_same_transpose(x, scale, dim_x, dim_scale)`: computes x's transpose state on `[dim_x, dim_x+1]` and scale's on `[dim_scale, dim_scale+1]`; requires they be EQUAL. Early-true guards: <2 dims after start, or both compared dims == 1.
- MX branch sets `dim_x = x.dim()-2` and crucially **`dim_scale = scale.dim()-3`** (non-MX path uses `dim()-2`). The `-3` SKIPS the trailing MX "2" dim, so for scale `(..,N,ceilK,2)` it compares dims `(N,ceilK)` against x2 `(..,N,K)` dims `(N,K)`.
- Checker location: op-plugin `op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp`.

## Practical rule for generating a transposed scale
- When the DATA tensor is transposed via `.transpose(-1,-2)`, its scale MUST be transposed too — via **`.transpose(-3,-2)`** (flips exactly the two non-"2" dims the checker examines). dim_scale uses `dim()-3`.
- Generating scale directly as a CONTIGUOUS target-shape tensor leaves it stride-non-transposed → mismatch → error "Input x2 tensor and scale tensor's transpose are not same" (x1 side: "pertoken_scale ... transpose are not same").
- `.transpose` is a view: data bytes unchanged, only the stride flag flips → numerics / golden mapping unchanged. A shape-only match (right shape form, contiguous strides) passes a local golden but FAILS the on-board stride checker.

See [[mx-quant-scale-semantics]], [[fp4-e2m1-packing]], [[qbmm-batch-status]].
