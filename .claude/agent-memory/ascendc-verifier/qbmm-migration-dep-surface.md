---
name: qbmm-migration-dep-surface
description: Verified cross-op and matmul/common dependencies QBMM v3/v4 actually pull in (more than architect's migration plan listed) — for checking migration completeness
metadata:
  type: reference
---

Verified by reading ops-nn source on 2026-07-11 (re-read before acting):

**v3 CMakeLists DEPENDENCIES** (`matmul/quant_batch_matmul_v3/CMakeLists.txt`):
`trans_quant_param_v2 convert_weight_to_int4_pack quant_batch_matmul_v4 transpose_batch_mat_mul` — matches what architect's plan listed.

**v4 op_host/CMakeLists.txt DEPENDENCIES** (MISSED by architect's plan):
`quant_batch_matmul_v3 weight_quant_batch_matmul_v2` — **`weight_quant_batch_matmul_v2` is an undeclared cross-op dependency.** It exists in ops-nn (`matmul/weight_quant_batch_matmul_v2/`) but the migration plan never mentions it. Must be migrated OR verified-unused-for-A2A3 before v4 builds.

**Cross-op `#include` deps in v3 (verified by grep)**:
- `transpose_batch_mat_mul/op_host/op_tiling/pp_matmul_default.h` — plan listed (HIGH). CONFIRMED present.
- `transpose_batch_mat_mul/op_host/op_tiling/transpose_batch_mat_mul_einsum_tiling.h` — **MISSED by plan** (referenced from `arch20/pp_matmul_int8_tiling.h:19` and used as `optiling::transpose_batch_mat_mul::TransposeBatchMatMulEinsumTiling` at `pp_matmul_int8_tiling.cpp:90`).
- `matmul/common/op_host/op_api/matmul_util.h` — plan listed. CONFIRMED present in omni-ops matmul/common.
- `matmul/common/op_host/log_format_util.h` — **MISSED by plan** (used by infershape, checker, aclnn_v4). NOT present in omni-ops (verified: find returns nothing).
- `matmul/common/op_host/op_api/quant_matmul_v4.h` — **MISSED by plan** (used by aclnn_quant_matmul_v4.cpp:26 and aclnn_quant_matmul_v5.cpp:22). NOT present in omni-ops.
- `matmul/common/op_host/math_util.h` — present in omni-ops matmul/common (OK).

**Net**: omni-ops `matmul/common` currently lacks `log_format_util.h` and `op_api/quant_matmul_v4.h`. These must be copied from ops-nn `matmul/common/` into omni-ops `matmul/common/` (or the includes rewritten) or v3/v4 host compile fails. Plan only mentioned `matmul_util.h`.

**MX deferral claim nuance**: plan says "MX logic 全在 arch35". Mostly true, BUT `op_kernel/quant_batch_matmul_v3_base.h:363` (top-level, not arch35) references `AscendC::fp8_e8m0_t`. Whether this type exists for arch20/22 bisheng is BOARD-ONLY — the base header is compiled by all SoCs. Deferral is defensible but "A2A3 path completely free of MX logic" is slightly overstated.

**v4 kernel CMakeLists** (`matmul/quant_batch_matmul_v4/op_kernel/CMakeLists.txt`): first block is `add_kernel_sources(AUTO_SYNC false)` with NO KERNEL_SRC / NO COMPUTE_UNITS — i.e. v4 ships NO A2A3 kernel. Only `arch35/quant_batch_matmul_v4.cpp` for ascend950/ascend350/mc62. So `QuantBatchMatmulV4` registers for A2A3 (host config exists: `op_host/config/ascend910b/`, `ascend910_93/`) but has no kernel — runtime call on 910B would fail. This is the substance of plan OPEN QUESTION 1/3.

**Architect's plan CMakeLists template bug** (§6.3): lists `op_api/aclnn_quant_matmul_v3.cpp` in `target_sources(opapi ...)` — that file DOES NOT EXIST (only `aclnn_quant_matmul_v3.h`; V3 aclnn is auto-gen from def via op_host_aclnnInner). Template as-written would hard-fail CMake. Also misplaces where `aclnn_quant_matmul_v5.cpp` lives (v4 `op_host/op_api/`, not v3 `op_api/`).

Related: [[omni-ops-build-model-facts]]
