---
name: qbmm-omni-v4-binary-compile-driver
description: QBMM-omni V4 910B binary 编译产物差异的真正根因——nn --pkg 模式由 binary.json(明文 dtype,含 int4)驱动编译,omni -n -c 模式由 ops-info.ini(opbuild 从 config_910BC 切片,INT8-only)驱动;源码全等价(顶层全量dtype/config_910BC/binary.json/tiling_key.h/registry 逐层相同),差异纯在 build 体系驱动的编译源不同
metadata:
  type: project
---

QBMM-omni V4 (ai_infra_quant_batch_matmul_v4) 910B binary 产物差异的真正因果链（2026-07-31 host-engineer 三次 trace 定论，基于 nn gen_ops_info.cmake + omni func.cmake 源码逐行对比 + 服务器铁证，**同时推翻** [[qbmm-omni-a8w4-pergroup-ez1009-real-cause]](说 nn 也走 ini) 与本 memory 上一版(说 config_910BC 是通用闸门)）。

**铁证（服务器实证，2026-07-31）:**
- nn `bash build.sh --pkg --soc=ascend910b --ops=quant_batch_matmul_v4` 编出 **7 个 binary**（含 `QuantBatchMatmulV4_ND_ND_int4_int4_fp16_high_performance.o` 等，**明文 dtype + `_high_performance` 后缀**）。
- omni `bash build.sh -n ai_infra_quant_batch_matmul_v4 -c ascend910b` 只编 **6 个**（int8 系列，**hash 命名** `AiInfraQuantBatchMatmulV4_<hash>.o`，无 int4、无 perblock）。
- **文件名风格差异（明文 vs hash）是 build 驱动不同的第一个信号。**

**Why（根因 = build 体系的编译驱动源不同，源码全等价）:**

1. **源码逐层等价（已 trace 全部）:**
   - 顶层全量 OpDef dtype（nn def.cpp:23-112 vs omni def.cpp:23-112）：逐字符等价，**都含 INT4**（x1 :48-49 `ge::DT_INT4,ge::DT_INT4`，x2 :111-112 同）。
   - `config_910BC`（nn def.cpp:823-971 vs omni def.cpp:811-959）：逐字节等价，**都 INT8-only**（x1/x2 各 16 个 DT_INT8，无 INT4/INT32）。
   - `binary.json`（nn vs omni ascend910b config）：同 7 条结构（int8_int4×4 + int8_int8 perblock×1 + int4_int4×2），`bin_filename` 明文含 dtype。
   - `tiling_key.h` / msd_tiling / perblock_tiling / pergroup_tiling：逐字符等价（memory 记录 + 本次抽样核对）。
   - `tiling_registry.cpp`：都注册 {MSD,PERBLOCK,PERGROUP} 三模板（omni :30-32）。
   - aclnn V5 dispatch（nn:1164 vs omni:1166）：逐字等价。
   - **结论：源码层面 nn 与 omni 完全等价，产物差异不可能由源码解释。**

2. **真正的差异 = binary 编译驱动源（这是根因，定论）:**
   - **nn `--pkg` 模式 → binary.json 驱动编译。** nn build.sh `build_binary`(:1160-1235) 走 `gen_ops_info.cmake`(:621-692)：对每个 op **无条件**调 `prepare_compile_from_config`(:650, `BINARY_JSON ${binary_json}` :654)。该函数 :353 `if(EXISTS ${BINARY_JSON})` → `binary_config_copy`(:355) 把 op_host/config 的 binary.json 拷到 tbe/config → `build_binary_opc.sh`(:380) **按 binary.json 的明文 dtype 列表驱动编译** → `build_binary_op_exe_task.sh`(:445) 每个 opc_cmd 编一个 binary。**binary.json 含 int4_int4/perblock → nn 编出 int4/perblock binary。** 明文文件名（`_ND_ND_int4_int4_fp16_high_performance`）= binary.json 的 `bin_filename` 字段（被 build_binary_opc.sh 用明文，覆盖 `BIN_FILENAME_HASHED=1`）。
   - **omni `-n -c` 模式 → ops-info.ini 驱动编译。** omni build.sh `-n -c` 走 :493 `cmake_config` + :494 `build package`；binary 编译在 CMakeLists.txt :752 `if(ENABLE_OPS_KERNEL)`（config.cmake :42 默认 ON）→ func.cmake `add_bin_compile_target`(:348-505)。该函数 :345 `file(GLOB ${GEN_OUT_DIR}/*.sh)` 只读 `ascendc_bin_param_build.py`(:129-146) 生成的 .sh；该 py 读 **`aic-${unit}-ops-info.ini`**(:130) 生成 .sh。**ini 由 opbuild 从 OpDef 的 config_910BC 切片 → config_910BC INT8-only → ini 不含 int4/perblock dtype → .sh 不含 int4 → omni 不编 int4/perblock。** omni 的 binary.json 在编译期**只被 install**（func.cmake :452 装到 config 目录），**从不作为编译驱动源**。
   - **omni 仓内全仓 grep 无 `prepare_compile_from_config`/`compile_from_config`/`build_binary_opc.sh`/`build_binary_op_exe_task.sh` 任何调用**——omni 的 `OPS_KERNEL_BINARY_SCRIPT` 变量(variables.cmake:39)虽定义但 func.cmake 从不引用其 build_binary_*.sh。**这是 omni 缺 binary.json 驱动路径的硬证据。**

3. **两个仓用的 `ascendc_bin_param_build.py` 版本也不同（次要因素）:**
   - nn：`ops-nn/scripts/util/ascendc_bin_param_build.py`（仓内自带，13284 字节）。
   - omni：`${ASCEND_CMAKE_DIR}/util/ascendc_bin_param_build.py`（CANN 包内置，`ASCEND_CMAKE_DIR=${ASCEND_PROJECT_DIR}/cmake`）。
   - 但这不是主因——即便同一版本 py，nn 走 binary.json 主路径绕过了 ini dtype 限制，omni 只走 ini 路径被 config_910BC 闸住。

**结论（不可逆，2026-07-31 三次 trace 定论）:**
- **根因 = build 体系差异，不是源码差异、不是 config_910BC 本身、不是 stale。** nn `--pkg` 由 binary.json（明文，含 int4）驱动 binary 编译；omni `-n -c` 由 ops-info.ini（config_910BC 切片，INT8-only）驱动。同一份等价源码 + 等价 binary.json，nn 的 build 体系读了 binary.json 编出 int4，omni 的 build 体系没读 binary.json 只读 ini 漏了 int4。
- **config_910BC INT8-only 只对 omni 的 ini 路径是关卡；对 nn 的 binary.json 路径不是关卡。** 这解释了"config 等价但 nn 编 int4 omni 不编"的表面矛盾——两仓 build 路径不同，config 在不同路径里扮演角色不同。
- **memory 上一版"config_910BC 是编译期闸门"只对 omni 成立；旧 memory"nn 也走 ini/改 config 是伪修"完全错。**

**How to apply（修复方案，让 omni 910B 编出和 nn 一样的 7 个 int4/int8 binary）:**
- **方案 A（对齐 nn，正解，推荐）: 给 omni 的 func.cmake / CMakeLists.txt 补 binary.json 驱动编译路径。** 即在 `add_bin_compile_target` 之外（或之前），参照 nn gen_ops_info.cmake :621-692 增加：检测 `${op_dir}/op_host/config/${unit}/${op_name}_binary.json` 存在 → 走 `build_binary_opc.sh` + `build_binary_op_exe_task.sh`（需引入 nn 的 `${OPS_KERNEL_BINARY_SCRIPT}` 脚本）按 binary.json 明文 dtype 编译。这是 host-engineer 范畴（CMake/build 链改动，§2.2.1 必派 agent 落地）。**侵入性中等，但最贴近 nn 原版行为，编出产物与 nn 逐一对齐。**
- **方案 B（最小侵入，治标）: 用 `add_ops_tiling_keys`（func.cmake :212）显式指定 int4/perblock 的 tiling_key 白名单。** 但这只影响 tiling_key 枚举，不改变 ini 的 dtype 切片——若 ini 本身不含 int4 dtype，加 tiling_key 也编不出 int4 binary。**需服务器验证 ini 是否被 tiling_key 白名单旁路；大概率不够。**
- **方案 C（被 nn 先例反证 = 不可行，2026-07-31 客观重评定论）: 给 config_910BC 补 INT4/INT32 dtype 组合。** 表面看能让 ini 含 int4 从而 omni ini 路径编 int4，但**铁证反证**：(a) nn 自己的 config_910BC（nn def.cpp:823-971）与 omni def.cpp:811-959 **逐字节相同**，x1/x2 各 16 槽全 `ge::DT_INT8`，**无任何 INT4**——nn 编出 7 个 INT4 binary 完全靠 binary.json 绕过 config_910BC，config 对 nn 的 INT4 binary **零贡献**；(b) 改 config 无 nn 先例可循，属纯自创偏离；(c) config_910BC 还驱动运行时 OpDef dtype 校验，加 INT4 会污染 dtype 白名单（可能让别的非法输入误 accept，或触发 checker 额外路径）；(d) config dtype 列表与 bias/x2_scale/y 等 16 槽是**成行绑定**（同行跨输入是同一个 binary 的 dtype 组合），单改 x1/x2 会破坏行对齐语义。**结论：方案 C 不是"不推荐"，是"被 nn 先例证伪"。正解唯一 = 方案 A。**
- **omni 若要立即像 nn 一样编 INT4 binary，前提是 omni build 体系必须先具备 binary.json 驱动路径（方案 A）；单靠 config/tiling_key 改动都不够。**

**诊断命令（服务器实证用）:**
- omni build 后查 `build/binary/ascend910b/gen/*.sh` 列表：若不含 int4_int4 的 .sh → 确认 ini 闸住（ini 路径）。
- omni build 后查 `build/binary/ascend910b/bin/AiInfraQuantBatchMatmulV4_*/` 产物清单：确认 6 个 hash 命名、无 int4。
- 对照 nn build 后查 `build/binary/ascend910b/bin/ascend910b/quant_batch_matmul_v4/`：确认 7 个明文命名、含 int4。
- 关键判别：omni build 日志里是否出现 `build_binary_opc.sh` / `build_binary_op_exe_task.sh` 调用——若**无**（预期无），即证实 omni 走 ini 路径不走 binary.json 路径。

- 相关：[[qbmm-v3-v4-directory-split]] [[qbmm-omni-migration-progress-20260723]] [[aclnn-error-msg-build-fingerprint]] [[qbmm-omni-build-merge-v3-v4]]

---

**方案 A 落地状态（2026-07-31 host-engineer 本机改完，待上板编验证）:**

第一步（搬脚本）+ 第二步（挂 cmake 通路）已完成，本机改动在 `D:/Desktop/Code/omni-ops/inference/ascendc/`:

- **新增 `cmake/gen_binary_from_json.cmake`**：从 nn `gen_ops_info.cmake:287-497` 抽 5 函数（generate_bin_scripts / binary_config_copy / prepare_compile_from_config / compile_from_config / gen_binary_info_config_json）+ `get_op_type_from_binary_json` + 入口 `compile_binary_from_json`。变量映射：`ASCEND_AUTOGEN_PATH`→`ASCEND_AUTOGEN_DIR`、`BIN_KERNEL_INSTALL_DIR`→omni `packages/vendors/${VENDOR_NAME}/...`（omni 无 ENABLE_BUILT_IN 时此变量未定义）。`prepare_compile_from_config` 的 build_binary_opc.sh 后 5 个开关参数 omni 全无（ENABLE_OOM/ENABLE_DUMP_CCE/ENABLE_MSSANITIZER/BISHENG_FLAGS/KERNEL_TEMPLATE_INPUT）→ 全传 OFF/空。
- **`CMakeLists.txt:766-767`**：`add_bin_compile_target`(ini 通路) 之后 `include(gen_binary_from_json.cmake)` + `compile_binary_from_json("${OP_DIR_LIST}")`。
- **`cmake/func.cmake:359-362`**：`add_bin_compile_target` 内对 `${op_file}_binary_json_config==TRUE` 跳过（避免 V4 被 ini + binary.json 双编）。
- **`src/ops-nn/matmul/ai_infra_quant_batch_matmul_v4/CMakeLists.txt:24`**：`set(ai_infra_quant_batch_matmul_v4_binary_json_config TRUE PARENT_SCOPE)` 选定 V4 走 binary.json。

**关键工程决策（architect 原假设需修正）:** architect 提示说"ai_infra_matmul 等无 binary.json 的走原 ini 通路"——**实证 ai_infra_matmul 也有 `ai_infra_matmul_binary.json`**（在 `src/ops-nn/matmul/ai_infra_matmul/op_host/config/ascend910b/`）。若用简单 `if(EXISTS binary.json)` guard 会把 ai_infra_matmul 也卷进 binary.json 通路（它当前走 ini 正常工作，不该动）。**正确 guard = per-op opt-in marker**（`<op_name>_binary_json_config=TRUE`），只 V4 op_host CMakeLists 显式置。V3（被 V4 依赖）/ matmul 均不置 marker → 留 ini 通路零影响。

**待上板验证未知（build-time, 本机 cmake 检查无法证明）:**
- `build_binary_gen_opc_info.sh` 读 `${topdir}/build/tbe/config/aic-<unit>-ops-info.ini`——omni 的 generate_compile_cmd 是否把 ini 拷到 `tbe/config`（ASCEND_KERNEL_CONF_DST）而非只在 ASCEND_AUTOGEN_DIR，需上板编看日志。
- CANN 9.x opc 工具链对脚本的兼容（architect 标的最大未知）。
- 编出 binary 数量/dtype（预期 7 个，对齐 nn）。

**上板编命令（用户执行）见最终消息。**

---

**binary.json 编译链完整 trace（2026-08-03 host-engineer 源码逐行模拟，回答上文"待上板验证未知"）:**

实际编译通路是**双路径并存**(nn gen_ops_info.cmake:640,:650 同挂),不是 architect 早先设想的单路径:

- **路径1 = `generate_bin_scripts`(py, 从 ini 抽 dtype)**: gen_binary_from_json.cmake:22-40 调 `ascendc_bin_param_build.py`,读 `${ASCEND_AUTOGEN_DIR}/{default,inner,exc}/aic-<unit>-ops-info.ini`(三层),按 ini 的 `input{}.dtype`(逗号分隔多 dtype 组合)枚举,每组生成一个 `<op_type>-<op_file>-<idx>.sh` + `<bin_file>_param.json`(md5 hash 命名)。产物在 `${OUT_DIR}/gen/<op_name>/`。**这条路径的 bin_filename 是 md5 hash,不是明文 dtype**。
- **路径2 = `prepare_compile_from_config` → `build_binary_opc.sh` → `build_binary_opc_gen_task.sh`(直接读 binary.json)**: 这才是**真正调 asc_opc 编译**的路径。gen_task.sh:189 `get_binary_config_file` 找 `build/tbe/config/<unit>/<op>/<op>_binary.json`(由 binary_config_copy 从 op_host/config 拷来),:302 `grep bin_filename | wc -l` 数 binary.json 里的条目数 = 线程切片数,:317-360 对每条 `bin_filename` 生成一行 `asc_opc <dynamic_py> --main_func --input_param=<binary.json 切片> --soc_version --impl_mode --op_mode=dynamic` 写入 `opc_cmd.sh`。**这条路径用 binary.json 的明文 bin_filename**(配合 `BIN_FILENAME_HASHED=1` env 控制是否 hash)。

**步骤映射(7 步全验证):**
1. build.sh `build_binary_from_json`(:182) 串: `prepare_binary_compile_<unit>` → re-configure 灌 OPC_NUM → `binary` → `gen_bin_info_config`。对齐 nn build.sh:1208-1232。
2. `prepare_binary_compile` 拉起: `generate_bin_scripts`(py,路径1) + `gen_opc_info`(build_binary_gen_opc_info.sh 生成 opc_info.csv) + `config_compile`(build_binary_opc.sh,路径2 生成 opc_cmd.sh) + cp kernel/py 到 out_dir/src。
3. `gen_opc_info` → `gen_opcinfo_for_socversion.sh:67` 读 **CANN 包内置** `${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/config/<unit>/aic-<unit>-ops-info.json` → `gen_opcinfo_from_opinfo.py:69` 只收 `dynamicShapeSupport.flag==true` 的 op → 输出 opc_info.csv(op_type, py 文件名, main_func 名)。V4 def.cpp:953 有 `DynamicShapeSupportFlag(true)`,V4 无 `.OpFile()`→ 文件名走 `convert_to_snake`=`dynamic/ai_infra_quant_batch_matmul_v4.py`。
4. `opc_cmd.sh` 每行: `asc_opc ${dynamic_py} --main_func --input_param=<binary.json切片_${impl}_${i}> --soc_version=Ascend910B1 --impl_mode --simplified_key_mode --op_mode=dynamic`。asc_opc 是 CANN 包自带可执行(在 PATH),nn/omni 共用同一 CANN 包,这层无差异。
5. opc 编 INT4 kernel: 910B(`__CCE_AICORE__==220`)+ INT4(`ORIG_DTYPE_X2=DT_INT4`)。msd.h 用 `int4b_t`(910B 原生 dtype)+ `MatmulType<...,int4b_t>` 高阶 API + 纯向量"2行int4转1行int8"。**无 A5 专属指令**(无 arch35/Regbase/__simd),910B 可编。nn 在 910B 就是用这套编出 INT4 binary。
6. `binary` target → `build_binary_op_exe_task.sh` 按 `OPC_NUM` 切片,每片 sed 取 opc_cmd.sh 第 idx 行执行,产物 `.o` + `.json` 落 `${OUT_DIR}/bin/<unit>/<op>/`。
7. `gen_bin_info_config` → `gen_binary_info_config.py` 扫 bin 目录生成 `binary_info_config.json`(运行时按 dtype/format/attr 选 binary 的索引)。

**回答"待上板验证未知"(本次 trace 定论):**
- (a) ini 路径: omni 的 opbuild(config.cmake:213 prepare.sh)生成三层 ini 到 `${ASCEND_AUTOGEN_DIR}/{default,inner,exc}/`,func.cmake:149 有 `opbuild_gen_default/inner/exc` target。**路径1(py 读 ini)的输入路径对齐**。但 gen_binary_from_json.cmake:22-40 的 `generate_bin_scripts` target **缺 DEPENDS opbuild_gen_*/ascendc_impl_gen**(nn gen_ops_info.cmake:311 有 `DEPENDS ascendc_impl_gen`)——build.sh 顺序(build ops_kernel 先于 build_binary_from_json)兜底了,但单独跑 target 会漏。**小坑,非阻塞**。
- (b) opc_info.csv 的 ini 路径: build_binary_opc_gen_task.sh:122 读 `${topdir}/build/tbe/temp/_<unit>_tmp.csv`,由 `gen_opcinfo_for_socversion.sh:54` 从 `${binary_temp_conf_dir}/_<unit>.json`(gen_opinfo_json_from_ini.sh 从 ini 转)生成。**这条不直接读 ASCEND_AUTOGEN_DIR 的 ini**,而是 gen_opcinfo_for_socversion.sh 内部重新从 `${topdir}/build/tbe/` 路径找。omni 的 ASCEND_KERNEL_CONF_DST=`${CMAKE_BINARY_DIR}/tbe/config`(variables.cmake:83),topdir=`${CMAKE_BINARY_DIR}`。**路径需上板确认 gen_opinfo_json_from_ini.sh 能找到 ini**(它内部从 workdir 推 topdir,build 目录布局对齐则通)。
- (c) opc 工具链版本兼容: asc_opc 来自 CANN 包,nn 与 omni 共用同一 CANN 9.1.0。**无差异,非 omni 改动范畴**。
- (d) INT4 kernel 910B 可编性: 已确认(见步骤5),无 A5 专属指令。
- (e) binary.json 条目: omni ascend910b config 有 **7 个 bin_filename**(int8_int4×4 + int8_int8 perblock×1 + int4_int4×2),与 nn 等价。
- (f) scripts 依赖: omni `scripts/util/` 只需放 CANN 包没有的(已搬 ascendc_bin_param_build.py/opdesc_parser.py/const_var.py + `__init__.py`);`ASCENDC_CMAKE_UTIL_DIR=${ASCEND_CMAKE_DIR}/util`(config.cmake:86)指向 **CANN 包**的 parse_ini_to_json.py/ascendc_impl_build.py/ascendc_ops_config.py,omni 不需自己搬。`binary_script/` 全 24 文件已从 nn 全搬(ls 对照逐一对齐)。

**仍需上板才能证实的(build-time,本机 cmake 模拟无法替代):**
- `gen_opinfo_json_from_ini.sh`(gen_opcinfo_for_socversion.sh step1 调)在 omni build 目录布局下能否找到三层 ini 并合并 → 决定 opc_info.csv 是否含 V4 行。
- `build_binary_opc_gen_task.sh:189 get_binary_config_file` 能否在 `build/tbe/config/ascend910b/ai_infra_quant_batch_matmul_v4/` 找到 binary.json(binary_config_copy 拷贝时机)。
- 实际编出的 binary 数量与 dtype(预期 7 个,含 int4/perblock,明文 bin_filename)。

**小坑清单(可改可不改,非阻塞):**
1. gen_binary_from_json.cmake:22-40 `generate_bin_scripts` 缺 `DEPENDS opbuild_gen_default opbuild_gen_inner opbuild_gen_exc generate_adapt_py`(对齐 nn :311)。
2. gen_binary_from_json.cmake:93-102 `config_compile` 缺 `DEPENDS generate_adapt_py`(dynamic py 依赖)。
3. 上两项被 build.sh:175 `build ops_kernel` 先于 `build_binary_from_json` 顺序兜底,不影响 `bash build.sh -n ai_infra_quant_batch_matmul_v4 -c ascend910b` 全流程;只影响单独 `cmake --build --target prepare_binary_compile_xxx`(nn 也一样)。

**结论(能否复现 nn 的 INT4 binary 编译): 部分能——cmake 通路已全挂(变量/脚本/依赖链对齐 nn),本机 trace 无阻塞;真正编出 7 个 INT4 binary 仍需上板跑 `build.sh -n ai_infra_quant_batch_matmul_v4 -c ascend910b` 看 opc_cmd.sh 行数 + bin 目录产物(这是 build-time 行为,源码 trace 无法替代)。**

