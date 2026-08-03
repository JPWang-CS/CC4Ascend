---
name: omni-missing-merge-ini-tbe-config
description: omni 仓缺 merge_ini 逻辑致 binary.json 通路 tbe/config/aic-<soc>-ops-info.ini 不存在, gen_opc_info 报 No rule; 根因/修法/对照 nn
metadata:
  type: project
---

omni 仓 (`D:/Desktop/Code/omni-ops/inference/ascendc/`) 缺 merge_ini 逻辑，导致 binary.json 编译通路在 `gen_opc_info` 阶段报 `No rule to make target 'tbe/config/aic-ascend910b-ops-info.ini'`。

**Why:** omni 的 opbuild (CMakeLists.txt:612/628/644 opbuild_gen_default/inner/exc) 把 ops-info.ini 写到 `build/autogen/{,inner/,exc/}aic-<soc>-ops-info.ini` 三层。omni 的 `add_ops_info_target` (func.cmake:153) 只 `parse_ini_to_json` 生成 JSON，**从不**把三层 ini 合并到 `build/tbe/config/aic-<soc>-ops-info.ini`。但 binary.json 通路的脚本（gen_opcinfo_for_socversion.sh + gen_opinfo_json_from_ini.sh:38-39）硬编码读 `tbe/config/aic-<soc>-ops-info.ini`（`${topdir}/build/tbe/config/`，topdir=仓库根），且 gen_binary_from_json.cmake 原 `gen_opc_info` target DEPENDS 这个 OUTPUT 路径。nn 仓有 merge_ini_files 函数 (cmake/gen_ops_info.cmake:239-253) + merge_ini_files.py 把三层 ini merge 到 tbe/config，omni 完全没有（grep 全仓空）。

**How to apply:** 当 omni binary.json 通路报 `tbe/config/aic-*-ops-info.ini` 相关 No rule / does not exist，或任何脚本读 `${topdir}/build/tbe/config/` 下 ini 失败时，根因就在缺 merge_ini。修法（已落地 2026-07-30）：`cmake/gen_binary_from_json.cmake` 加 `merge_ini_files_for_binary` 函数（对齐 nn merge_ini_files，OUTPUT=${ASCEND_KERNEL_CONF_DST}/aic-<soc>-ops-info.ini，DEPENDS opbuild_gen_default/inner/exc），在 `gen_opc_info` target 定义处先挂 merge target 再让 gen_opc_info / config_compile DEPENDS `merge_ini_<unit>` target（而非 OUTPUT 路径）。merge_ini_files.py 之前搬 binary_script 时已一并带入，无需再补脚本。注意：generate_bin_scripts 里 py 查 autogen 三层 ini 只找到 inner 打 "does not exists skip" 是预期（omni default/exc 层 ini 常空），不是 bug，inner 够用 merge 会合并。

关联 [[qbmm-omni-v4-binary-compile-driver]]（binary.json 驱动整体通路）、[[binary-json-cmake-target-vs-build-driver]]（孤岛 target 修法）。
