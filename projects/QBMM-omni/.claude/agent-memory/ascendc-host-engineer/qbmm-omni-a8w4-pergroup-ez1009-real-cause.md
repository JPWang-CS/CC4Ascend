---
name: qbmm-omni-a8w4-pergroup-ez1009-real-cause
description: 【已推翻 2026-07-31 三次 trace】旧推断"binary 编了/config 不卡编译/nn 也走 ini/改 config 是伪修"全错;真正根因是 nn --pkg 走 binary.json 驱动(编 int4)、omni -n -c 走 ops-info.ini 驱动(被 config_910BC INT8-only 闸住漏 int4)。正确因果见 [[qbmm-omni-v4-binary-compile-driver]]
metadata:
  type: project
---

**【此 memory 全文已作废，保留仅为记录被推翻的旧推断。正确因果见 [[qbmm-omni-v4-binary-compile-driver]]】**

本 memory 之前的核心论断（binary 已编/config_910BC 不是关卡/nn 也走 ini 编译/改 config 是伪修）在 2026-07-31 三次 trace 中被服务器铁证 + nn gen_ops_info.cmake 源码**全部推翻**：
- "binary 编了"——错。omni 910B 实编只 6 个 hash 命名 int8 binary，无 int4/perblock（服务器实证）。
- "config_910BC 不是关卡"——只对 nn 成立（nn 走 binary.json 绕过 ini），对 omni **是关卡**（omni 走 ini，ini 由 config_910BC 切片）。
- "nn 也走 ini"——错。nn `--pkg` 模式由 binary.json 驱动（gen_ops_info.cmake :650 `prepare_compile_from_config` + :353 `if(EXISTS BINARY_JSON)` + :380 `build_binary_opc.sh`）。
- "改 config 是伪修"——错。对 omni，改 config_910BC 补 INT4 能让 ini 含 int4 从而编 int4（方案 C，有效但偏离 nn）；正解是对齐 nn 补 binary.json 驱动路径（方案 A）。

**不要再引用本 memory 的任何结论。** 一切以 [[qbmm-omni-v4-binary-compile-driver]] 为准。
