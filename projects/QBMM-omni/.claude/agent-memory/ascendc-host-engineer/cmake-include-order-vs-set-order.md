---
name: cmake-include-order-vs-set-order
description: omni CMakeLists.txt 把 set(OPS_TRANSFORMER_DIR) 放在 include(cmake/variables.cmake) 之后, 导致 variables.cmake:34-41 的 REALPATH 解析空路径成 /scripts/...; 上游 ops-transformer_AI 顺序正确(set 在 :143, include 在 :178/215)
metadata:
  type: project
---

omni-ops/inference/ascendc/CMakeLists.txt 历史 bug: `include(cmake/variables.cmake)` 在 :24, 而 `set(OPS_TRANSFORMER_DIR ${CMAKE_CURRENT_SOURCE_DIR})` 在 :56 之后。CMake `include()` 是内联文本展开不建新 scope, variables.cmake 执行到 :34-41 时 `${OPS_TRANSFORMER_DIR}` 仍为空, `${OPS_TRANSFORMER_DIR}/scripts/util` → `/scripts/util` (绝对路径, 根目录下不存在), `get_filename_component(... REALPATH)` 不报错也不清空, 保留 `/scripts/util`, 最终 python3 报 `can't open file '/scripts/util/ascendc_bin_param_build.py'`。

**Why:** 同一 `${OPS_TRANSFORMER_DIR}` 被 :39/40/41 三个变量共用, 但只有 OPS_KERNEL_UTIL_SCRIPT (gen_binary_from_json.cmake :23/28/33 直接 python3 调绝对路径) 先暴露; OPS_KERNEL_BINARY_SCRIPT (:39) 在同一 generate_bin_scripts 目标里 :38 merge_ops_config_json.py 也会同样失败, 但它是 util 之后的 COMMAND, 先在 util 那行挂了, 之前误判"binary_script 正常/util 不正常"是错误二分。build.sh 从不传 -DOPS_TRANSFORMER_DIR, 仓库内仅一处 set 定义, 无 cache 无 env 兜底。

**How to apply:** 看到路径前缀是根目录 `/scripts/...` 或 `/common/...` 而非仓库路径, 直接怀疑被插值的 `<DIR>` 变量为空 → 查 include 与 set 的相对顺序。修法: 把 `set(OPS_TRANSFORMER_DIR ${CMAKE_CURRENT_SOURCE_DIR})` 上移到所有 `include(cmake/*.cmake)` 之前 (对齐上游 ops-transformer_AI: :143 set → :153/178/215 include)。已落 CMakeLists.txt :20-22。`CMAKE_CURRENT_SOURCE_DIR` 在 `project()` 后就可用, 上移无依赖问题。

关联 [[qbmm-omni-v4-binary-compile-driver]] / [[binary-json-cmake-target-vs-build-driver]]。
