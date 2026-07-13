---
name: repo-mxfp4-tests-broken
description: The repos contain NO real e2m1 decode and op-plugin's 4 mxfp4 tests are broken (NameError, never ran, data not packed) — not a usable reference for fp4 calls
metadata:
  type: reference
---

# Repo MX fp4 goldens / tests are NOT a usable reference

When looking for a pre-existing fp4 quant-matmul call to copy, do NOT trust these:

- **No real e2m1 decode exists** in ops-nn / op-plugin / ops-transformer goldens (no value-LUT, no `>>4`/`&0x0f` unpack, no `ml_dtypes.float4` for the matmul path). Canonical goldens treat fp4 input as just `x.astype(np.float32)` and share ONE code path with fp8 — the 16-code e2m1→value LUT lives in cube HW only. This is fine for SCALE (e8m0, never packed) but WRONG for the DATA tensor, which must be truly 2:1 packed (see [[fp4-e2m1-packing]]).
- **op-plugin's 4 mxfp4 tests are broken and never ran**: `op-plugin/test/test_custom_ops/test_npu_quant_matmul.py`, the `test_npu_quant_matmul_mxfp4` / `_x1trans` / `_x1trans_x2trans` / `_notrans` fns. Two fatal defects: (1) use-before-assign NameError (`scale_clone = scale_clone.clone()` references the not-yet-defined name) → throws on entry; (2) data is int8 raw-passthrough with NO 2:1 packing → would mismatch even if the NameError were fixed. Also their scale is 2D `(N,30)` with no trailing "2", so not a guide for the transpose stride-check either (see [[scale-transpose-stride-semantics]]).
- Ground a fresh fp4 impl in the official `aclnnQuantMatmulWeightNz.md` convention (`(m, ceil(k/64), 2)` + groupSize 32 + OCP E2M1) and validate with a falsifiable test (see [[golden-falsifiable-testing]]), not in these tests.

See [[fp4-e2m1-packing]], [[mx-quant-scale-semantics]], [[golden-falsifiable-testing]].
