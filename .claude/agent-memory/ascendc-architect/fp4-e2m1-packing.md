---
name: fp4-e2m1-packing
description: fp4 (float4_e2m1fn_x2 / OCP E2M1) true 2:1 packing on the innermost stored dim for Ascend 950 quant-matmul; NPU x2 doubling; cube decodes e2m1 in HW
metadata:
  type: reference
---

# fp4 / e2m1 packing semantics (Ascend 950 quant-matmul)

General facts for feeding fp4 data into the MX cube path. Scale (e8m0) packing is separate — see [[mx-quant-scale-semantics]].

## True 2:1 packing on the innermost stored dim
- `float4_e2m1fn_x2` packs **2 fp4 values into 1 byte** along the tensor's **innermost (contiguous) STORED dim**, independent of transpose attribute.
- Stored innermost dim = logical / 2. **NPU multiplies the innermost stored dim ×2** (by the dtype tag, `unpack_factor=2`) to recover the logical dim.
- So to pass logical K: store x1 `(M, K/2)`, x2 `(N, K/2)` under trans_x2 — 2 fp4 per byte. NPU ×2 → logical K → `ceil(K_logical/64) == scaleK`. For trans_x2=false: x2 stored `(K, ceil(N/2))`, x1 `(M, ceil(K/2))`.
- **int-passthrough (fill int8 at full logical-K, no /2 pack, hand dtype tag to NPU) is DEAD.** NPU doubles the innermost dim → stored-K=256 reads as logical 512 → fails the scale-K-vs-data-K check (aclnn EZ0027 / device 161002, `CheckDimValueMicroScaling`: `CeilDiv(K,64) != scaleK`). aclnn `GetViewShape()` returns the LOGICAL (doubled) K.

## E2M1 value table
- 16-code OCP **E2M1** LUT: magnitudes `{0,0.5,1,1.5,2,3,4,6}` with sign bit (codes 0-7 positive, 8-15 negative).
- The HW cube decodes e2m1 directly (no software LUT in the kernel) — confirmed ≡ OCP E2M1 on all 16 codes by on-board falsifiable test (see [[golden-falsifiable-testing]]). Because the cube decodes in HW, an int-passthrough golden was numerically valid for *small representable* integers, but only true packing passes the shape checks.

## Inner-axis-even check reads STORED (non-doubled) shape
- Host `CheckInnerAxisIsEven` tests `dimValue[INNER] % 2` on the **raw STORED shape** (no ×2 fp4 doubling; only int4-from-int32 does ×8). So host inner-even sees stored bytes, NOT logical K.
- Consequence: an odd-K (e.g. K=255) negative test must be stored UNPACKED to make the host inner-even check fire; packing 255→128 would mask the rejection. Even-K packs to K/2 and passes legitimately. (The aclnn EZ0027 check is a SEPARATE layer reading logical K — both reject K=255-unpacked.)

## Format / N=1 nuances
- The fp4 `n>1`/`n>2`/`transX1=false` restrictions are **NZ-format-only** (under the "x2 is FRACTAL_NZ" bullet in proto). **ND-format fp4 enforces only K>2 + inner-axis-even** → **N=1 is LEGAL in ND** (x2 packs along innermost K; N at dim -2 untouched; transpose stride-check passes for size-1 dims).

## Two structural blind spots — NOT falsifiable by ANY numeric matmul test
1. **Nibble order** (which nibble of a byte is the K-earlier element): MX group=32 → both nibbles share one e8m0 scale, and the K dot-product is invariant to swapping the pair when x1/x2 swap together. Numeric NO-OP. Confirmable ONLY by spec + kernel read.
2. **Global sign flip**: matmul is bilinear-invariant. Confirmable ONLY by spec + kernel read.

See [[mx-quant-scale-semantics]], [[golden-falsifiable-testing]], [[scale-transpose-stride-semantics]], [[qbmm-batch-status]].
