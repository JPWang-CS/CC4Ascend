---
name: cann-blaze-cmct-gating
description: How IS_BLAZE compile-time gating selects Blaze vs CMCT MX kernel by CANN devkit version; dav_c220/c310 chip mapping; CANN 9.0.x fp4x2 LoadData whitelist requirement
metadata:
  type: reference
---

# CANN Blaze vs CMCT path gating + chip/arch mapping (MX quant-matmul)

General CANN/AscendC build facts for the MX quant-matmul kernels (`quant_batch_matmul_v3` arch35).

## IS_BLAZE selects the kernel path at COMPILE time
- `IS_BLAZE = (ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0)`. **No build.sh / cmake / CLI override.** Purely a function of the installed asc-devkit (CANN) version. CANN **9.0.x → MINOR==0 → IS_BLAZE=false → CMCT path**; **>=9.1 → IS_BLAZE=true → Blaze path**.
- The MX header include is mutually exclusive per translation unit: `(SCALE==E8M0 && IS_BLAZE)` → Blaze tensor-api kernel (ops-tensor); `elif (SCALE==E8M0)` → CMCT basic-api kernel (ops-nn matmul/common/cmct). One tilingkey compile takes exactly ONE path. Both headers declare the same kernel class.
- Consequence: choosing which path to test = choosing the devkit version. Swapping packages at runtime does NOT change the path; the static_assert / path are compile-time. To validate CMCT-specific code, stay on 9.0.x (do NOT jump to 9.1 or it flips to Blaze and bypasses the code under test).

## arch / soc / dav-codename mapping
- build.sh SOC_TO_ARCH: `ascend950→3510`, `ascend910b/910_93→2201`, `ascend310p→2002`, `ascend910→1001`; flag `--npu-arch=dav-${arch}`.
- **dav_c220 = davinci v220 = A2A3 = 910B/910C** (arch 2201). **dav_c310 = davinci v310 = A5 = 950** (arch 3510). The `dav_cXXX` header dir CCE compiles against is **LOCKED by the target soc**, not a free toggle.
- `op_kernel/arch35/` = 3510 = 950; `arch20/` = 2201 = 910B.
- **MX/fp4 is a 950-ONLY feature**: fp4/e2m1/e8m0 binary.json entries exist ONLY under `ascend950/` and `ascend350/` (both arch 3510). `ascend910b/...binary.json` has ZERO MX entries → 910B has no MX cube path. "Switch to c220 to dodge a c310 static_assert" is a DEAD END (different chip; c220 lacking fp4 whitelist does NOT imply it supports fp4).

## CANN 9.0.x fp4 CMCT build requires fp4x2 in the LoadData whitelist
- CMCT fp4 LoadData reinterprets L0 to raw `fp4x2_e2m1_t` and calls `AscendC::LoadData<fp4x2>(...)`. Whether this compiles depends on the devkit's `LoadData 2dv2` `SupportType<...>` whitelist including `__fp4e2m1x2`.
- CANN **9.0.0 / T103** lacked fp4x2 in that whitelist → static_assert in CANN's OWN header `.../asc/impl/basic_api/dav_c310/kernel_operator_mm_impl.h` ("LoadData 2dv2 only support uint8_t..."). This is a **toolchain version skew, NOT a code defect** — the main repo ships & precision-tests fp4 CMCT, and the failure is in the toolkit header, not repo code. The downstream symptom is a misleading `ld.lld "cannot open ..._N.o"` (the kernel `.o` never built).
- **Verify a candidate devkit**: grep `.../asc/impl/basic_api/dav_c310/kernel_operator_mm_impl.h` for `__fp4e2m1x2` in the supported-type list. Two independent fixes: (a) install a fp4x2-capable dav_c310 devkit (stays CMCT), or (b) upgrade to >=9.1 (flips to Blaze, never compiles the CMCT branch).
- fp8 always compiles (never hits the fp4 reinterpret).

## Cross-repo build order (workflow)
- Change ops-nn first + compile ops-nn first; only afterwards change ops-tensor — see [[ascendc-workflow-prefs]].

See [[mx-quant-scale-semantics]], [[fp4-e2m1-packing]], [[qbmm-batch-status]].
