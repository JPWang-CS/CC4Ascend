---
name: binary-json-cmake-target-vs-build-driver
description: omni binary.json 通路孤岛根因 — cmake 创建 target ≠ build.sh 会跑它; nn 用 build.sh 三步序列 + re-configure 驱动, 不是 cmake 单挂 ops_kernel
metadata:
  type: project
---

omni-ops 的 binary.json 驱动二进制编译通路, cmake 里 compile_binary_from_json 创建的所有 target (ascendc_bin_*, exe_compile_*_out, gen_bin_info_config_*) 是**孤岛** — 不在 ops_kernel ALL 目标的依赖链上。omni build.sh:175 只 `build ops_kernel`, 所以这些 target 创建了但从不执行, build_binary_opc.sh 永不被调用, V4 一个 .o 都没编。

**Why:** compile_from_config 在 cmake configure 时展开 `foreach(idx RANGE 1 ${OPC_NUM_${unit}})`, 但 OPC_NUM_<unit> 此时未定 (要等 prepare_binary_compile_* build 时生成 opc_cmd.sh 才能数行数)。所以即便把 target 挂到 ops_kernel, 也会因 OPC_NUM=空 而 foreach 不展开, 编不出 .o。

**正确驱动是 build.sh 三步序列** (对齐 nn build.sh:1203-1234, 已落地 omni build.sh build_binary_from_json):
1. `prepare_binary_compile_<unit>` → 跑 build_binary_opc.sh, 生成 opc_cmd.sh
2. 数 opc_cmd.sh 行数 → `OPC_NUM_<unit>` → **re-configure** (cmake .. 带 -DOPC_NUM 回灌) → foreach 正确展开
3. `binary` → build_binary_op_exe_task.sh 出 .o
4. `gen_bin_info_config` → 出 binary_info_config.json

**How to apply:** 任何"cmake 改了但上板没生效"的 case, 先区分 target 是**创建了** vs **被 build driver 执行了** — 看目标在不在 ops_kernel/all/ALL 依赖链上, 或 build.sh 有没有显式 --target 它。cmake message(STATUS) 验证 configure 时断点; build 日志 grep 验证 build 时执行。改 build.sh 是 host 工程链正当事 (非算子源码), 见 [[qbmm-omni-v4-binary-compile-driver]]。
