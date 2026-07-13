---
name: cann-custom-build-gotchas
description: Two reusable CANN build gotchas — custom single-op pkg tiling-parse failure (legacy.so dlopen), and pip wheel build-isolation hang when building es packages offline
metadata:
  type: reference
---

# CANN custom-build gotchas (reusable mechanisms)

Two CANN/ops-nn build mechanisms worth remembering across projects.

## 1. Self-built single-op custom pkg fails host tiling-parse (561103) → legacy.so dlopen, not missing registration
- Symptom: a self-built single-op custom vendor pkg (`build.sh --pkg --soc=Ascend950 --ops=<op>`) via `ASCEND_CUSTOM_OPP_PATH` reports `561103 / InitTilingParseCtx failed / tiling compile info parse failed`; clearing the env var (use builtin) → all pass.
- It is NOT missing op_proto/op_tiling/aclnn/op_master, and NOT json compile_info (compile_info is a C++ struct filled at runtime in the TilingParse callback). 
- **Real mechanism**: TilingParse/Tiling call `TilingPrepareForOpCache`, which dlsyms `LegacyTilingParsePrepareForOpCache` out of the toolkit's `libophost_comm_legacy.so` via `LegacyCommonMgr`. The custom `libcust_opmaster_rt2.0.so` locates that legacy so by a fragile multi-level `../` walk OUT of the vendor subtree back to the toolkit builtin opp; if the vendor dir isn't nested under the real toolkit opp, the walk lands nowhere → dlopen fails. (Or the master-built opmaster expects a legacy symbol/ABI the installed toolkit's legacy so doesn't match → GetFunc null.) Builtin pkg finds the legacy so in the SAME dir → always robust.
- **Diagnose**: `ASCEND_GLOBAL_LOG_LEVEL=0` grep `LegacyCommonMgr|libophost_comm_legacy|dest func.*null`; `nm -D` symbol match between cust opmaster (undefined) and toolkit legacy so (defined).

## 2. es package whl build hangs offline → pip build isolation
- `add_es_library_and_whl` / `add_es_library` are defined in the **cann/ge** repo `cmake/generate_es_package.cmake` (NOT cann/cmake); ops-nn loads them via `find_package(GenerateEsPackage)` from `${ASCEND_CANN_PACKAGE_PATH}/include/ge/cmake` (`FindGenerateEsPackage.cmake`).
- The real packaging command (only when `SKIP_WHL` unset): `${Python3_EXECUTABLE} -m pip wheel . --no-deps --wheel-dir=${PYTHON_BUILD_DIR}/dist`. Uses `Python3_EXECUTABLE` (not hard-coded python3). Only `--no-deps` is hard-coded; build-isolation is NOT forced either way → `PIP_NO_BUILD_ISOLATION=1` is respected.
- **Offline root cause**: pip default build isolation builds an isolated env that must fetch setuptools/wheel → no network → hang/fail. Set `PIP_NO_BUILD_ISOLATION=1`.
- **Skip mechanisms**: `add_es_library` = `_add_es_library_impl(SKIP_WHL)` (skips, just echoes + touches flag); `add_es_library_and_whl` does NOT skip. Flag `WHL_GEN_FLAG=${PYTHON_BUILD_DIR}/whl_generated.flag` is the custom-command OUTPUT; manually touching it (newer than CODE_GEN_TARGET) makes make treat it up-to-date and skip — no dist/*.whl produced but the whl install is OPTIONAL so missing doesn't error. MX kernel is an independent target not depending on the whl, so `make -k` also gets past it.

See [[cann-blaze-cmct-gating]].
