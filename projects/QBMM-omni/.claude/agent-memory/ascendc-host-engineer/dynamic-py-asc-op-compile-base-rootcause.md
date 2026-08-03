---
name: dynamic-py-asc-op-compile-base-rootcause
description: omni opc 编 INT4 时 shape -2 被 E80002 拒的根因=ascendc_impl_build.py 用旧版(CANN tbe para_check 严格); nn 用新版(asc_op_compile_base 宽松); 修法=搬 nn py + 改 CMakeLists 指仓内
metadata:
  type: project
---

omni opc 编 INT4 kernel 时, dynamic py 里 shape -2 被 `tbe.common.utils.para_check` 拒(E80002)。nn 不拒。

**根因(diff 实证)**:
- omni CMakeLists.txt:651 原用 `${ASCENDC_CMAKE_UTIL_DIR}/ascendc_impl_build.py`(CANN 包自带, 旧版) → 生成的 dynamic py `from tbe.common.utils import para_check`(严格, -2 拒)
- nn gen_ops_info.cmake:141 用 `${CMAKE_SOURCE_DIR}/scripts/util/ascendc_impl_build.py`(仓内自带, 新版) → 生成 `from asc_op_compile_base.common.utils import para_check`(宽松, -2 允许)

**asc_op_compile_base 是远端 CANN(8.x)自带的编译基础库, 不是仓内文件**:
- nn 全仓只有 ascendc_impl_build.py 一个文件 import 它, 无 setup.py 安装它
- 本机无 CANN 环境 find 不到, 证明它随 CANN 安装
- 故**只需搬 ascendc_impl_build.py 这层胶水, 不需搬 asc_op_compile_base 库**(nn 也没带)

**修法(已落地)**:
1. cp nn `scripts/util/ascendc_impl_build.py` → omni `scripts/util/`(687 行 1:1)
2. omni CMakeLists.txt:651 改 `${ASCENDC_CMAKE_UTIL_DIR}/ascendc_impl_build.py` → `${OPS_KERNEL_UTIL_SCRIPT}/ascendc_impl_build.py`(variables.cmake:41 已定义该变量=${OPS_TRANSFORMER_DIR}/scripts/util REALPATH)
3. 加 DEPENDS 该 py(对齐 nn gen_ops_info.cmake:147)

**Why**: omni 从 nn 迁移时漏搬了 ascendc_impl_build.py 这层编译胶水, 沿用 CANN 包旧版导致 dynamic py 用严格 para_check, INT4 dynamic shape(-2)编不过。
**How to apply**: 凡涉及 opc 编译期 shape 校验报错(E80002/para_check), 先看生成的 dynamic py 是 `tbe.common` 还是 `asc_op_compile_base` import; 前者=旧胶水, 后者=新胶水。

**接口兼容性已验**: argparse(argv nargs='+' + --opsinfo-dir nargs='*')两版一致; 依赖 const_var.py/opdesc_parser.py 仓内已有(nn 版仅格式差异, opdesc_parser 完全相同)。

相关: [[qbmm-omni-v4-binary-compile-driver]] binary.json 驱动链已通, 本条是 dynamic py 生成层的最后一环。
