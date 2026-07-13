---
name: omni-ops-build-model-facts
description: Verified facts about omni-ops inference/ascendc build system (build.sh flags, cmake glob, default SOC, missing macros, vendor isolation) — for falsifying migration plans
metadata:
  type: reference
---

Verified by reading source on 2026-07-11 (treat as snapshot, re-read before acting):

**build.sh** (`D:/Desktop/Code/omni-ops/inference/ascendc/build.sh`):
- `-n`/`--op-name`, `-c`/`--compute-unit` both exist and parse into `-DASCEND_OP_NAME` / `-DASCEND_COMPUTE_UNIT` (lines 268-275, 358-364). Multiple values separated by `;`.
- Default SOC when `-c` omitted is set in CMakeLists.txt:16 `ASCEND_COMPUTE_UNIT "ascend910_93"` — NOT `ascend910b`. build.sh help text (line 55) confirms.
- `set_env` (line 72-82) sources `$ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash` and hard-fails if bisheng missing.

**Auto-discovery glob** (`cmake/func.cmake:41 op_add_subdirectory`):
- Globs `src/ops-transformer/**/**/CMakeLists.txt` AND `src/ops-nn/**/**/CMakeLists.txt` (+ `**/**/ophost/CMakeLists.txt`). The `**/**` = category/op_name (2 levels). A 3-level path like `ops-nn/matmul/quant_batch_matmul_v3/CMakeLists.txt` matches; deeper nesting would NOT.
- Line 56-60: if NOT BUILD_OPEN_PROJECT, built-in ops are filtered out. BUILD_OPEN_PROJECT is ON by default (CMakeLists.txt:13) so this filter is inactive for open migration.
- `_depends` resolution (`op_add_depend_directory`, func.cmake:82-109): reads `${op_name}_depends` var (set PARENT_SCOPE by each op's CMakeLists) and auto-adds those dep op dirs to the build IF they exist under `src/${depend_info}/`. So declared deps DO auto-pull — but only if physically present in omni-ops tree.

**Macros that do NOT exist in omni-ops** (only `_aicpu` variant):
- `add_modules_sources` — grep finds only `add_modules_sources_aicpu` (func.cmake:929+). ops-nn style `add_modules_sources(HOSTNAME ... OPTYPE ... ACLNNTYPE ...)` will NOT resolve; CMakeLists must be rewritten to `target_sources(optiling/opsproto/opapi ...)` + `add_ops_compile_options(OP_NAME ...)`. (See [[omni-ops-cmake-target-sources-model]].)

**vendor_name isolation** (`CMakeLists.txt:18`, `config.cmake:75-77`):
- Default `VENDOR_NAME=omni_custom_transformer`. Install roots: `packages/vendors/${VENDOR_NAME}/op_api/include` (aclnn headers), `.../op_impl/ai_core/tbe/${VENDOR_NAME}_impl` (kernels). Since CANN built-in uses a different vendor namespace, header-name collision with built-in `aclnn_quant_matmul_v3.h` is NOT a real risk — the vendor prefix isolates it. Plans that flag "aclnn header collision with CANN" as HIGH risk are overstating; it's LOW.

**Three host CMake targets** (CMakeLists.txt:83-257): `op_host_aclnn` / `op_host_aclnnInner` (hand-written aclnn; comment line 11 "如果自己实现了aclnn接口使用这个") / `op_host_aclnnExc` + `opsproto` + `optiling` + `opapi` (output `cust_opapi`). ACLNNTYPE in ops-nn `add_modules_sources` can be `aclnn`/`aclnn_inner`/`aclnn_exclude` (func.cmake:930-931, 1013); the mapping to which omni-ops target receives the sources is non-obvious and must be traced per-op.

Related: [[qbmm-migration-dep-surface]]
