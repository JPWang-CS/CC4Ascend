---
name: dynamic-py-kernel-src-empty-and-path
description: 第九关根因 - compile_op src 文件名空(.cpp)+路径错。omni 缺 insert_kernel_src.py 注入 + 缺 kernel src 拷到 build/tbe/ascendc/<op>/
metadata:
  type: project
---

dynamic py 运行时 `compile_op(src)` 报 `cce file is not exists, file name: .../ascendc/<op>/.cpp`(文件名空 + 路径不对)。

**根因(双因)**:
1. **文件名空**: omni 缺 insert_kernel_src.py 调用, opbuild(CANN 宏)生成的 aic-*-ops-info.ini 无 `kernelSrc.value` 字段 → ascendc_impl_build.py:_write_impl line588 `src=self.kernel_src+'.cpp'` 时 kernel_src 空 → dynamic py 里 ascendc_src_file="" → compile_op 收到 `.../.cpp`
2. **路径不对**: omni 缺 kernel src 拷贝到 `build/tbe/ascendc/<op>/`。dynamic py 的 get_kernel_source 找 `${PYF_PATH}/../ascendc/<op>/<file>.cpp` (PYF_PATH=build/tbe/dynamic), omni 只拷到 build/binary/<soc>/src/<op>/(add_ops_src_copy), 不在 build/tbe/ascendc/

**Why**: nn 在 opbuild.cmake:46-51 (execute_process 调 insert_kernel_src.py) + gen_ops_info.cmake:523-528 (kernel_src_copy) 两步补齐; omni 走 CANN 宏 opbuild_gen_default, 无这俩后处理步骤。opdesc_parser.parse_kernel_src 有 fallback `if not kernel_src: kernel_src=op_file`, 但 omni 的 ini 连 opFile.value 也可能缺(CANN opbuild 不自动生成这俩字段)。

**How to apply**: 修法三件套(已落地 CMakeLists.txt:649-690 + func.cmake 加 get_op_type_from_op_name + scripts/util 搬 insert_kernel_src.py):
1. 搬 nn scripts/util/insert_kernel_src.py 到 omni (改 sys.exit→return 优雅跳过缺 ini 层)
2. func.cmake 加 get_op_type_from_op_name (从 <op>_def.cpp grep OP_ADD 提取驼峰 op_type 匹配 ini section)
3. CMakeLists.txt generate_adapt_py 前加 insert_kernel_src target: 遍历 OP_DIR_LIST, **只对 `${_op_name}_binary_json_config` == TRUE 的算子(V4)** 拷 op_kernel/* 到 build/tbe/ascendc/<op>/ + 调 insert_kernel_src.py 注入三层 ini (default/inner/exc); generate_adapt_py 依赖它; gen_binary_from_json merge_ini 也加依赖(保证 tbe/config 合并 ini 也含 kernelSrc)

**必加 marker guard(第十关教训)**: 最初版遍历所有 OP_DIR_LIST 对 V3+V4 都注入, 但 V3 走 ini 通路 opbuild 已生成 kernelSrc.value → insert 再注入 → `RuntimeError: Op:AiInfraQuantBatchMatmulV3 kernelSrc value is repeated!` (gen_ops_info build.aic-ascend910b-ops-info.json 失败)。修法 = foreach 头加 `if(NOT ${_op_name}_binary_json_config) continue()`, 只 V4(binary.json 通路, opbuild 不生成 kernelSrc)走 insert。marker 由 V4 op_host/CMakeLists:24 PARENT_SCOPE, op_add_subdirectory(:305-344) 执行后 660 时已就绪。
- 关键: op_type 用 OP_ADD 驼峰名(AiInfraQuantBatchMatmulV4), kernel_src 用下划线 op_name(ai_infra_quant_batch_matmul_v4 不带.cpp, ascendc_impl_build 给加 .cpp); optype_snake/ex 转换后 dir=op_name, 匹配拷贝目录
- VERBATIM 必加(bash -c "find \\;" 分号转义, 对齐 nn gen_ops_info.cmake:32); find -exec \; 在 VERBATIM 下被 cmake 吃掉, 改 -exec ... + (POSIX)

相关: [[dynamic-py-asc-op-compile-base-rootcause]] (第八关 asc_op_compile_base), [[binary-json-cmake-target-vs-build-driver]]
