---
name: ascendc-golden-testing
description: AscendC golden 对拍的判据、输入构造、输出规范、可证伪测试设计。判据对齐官方 test_ai_infra_matmul.py：flat isclose(rtol=atol=tol) + err_ratio≤阈值 gating，cos/rel_l2 仅诊断。当要写 golden 脚本、定判据、构造测试输入、设计阴性对照、或规范对拍输出时调用。内容来自真实 project golden 脚本（带行号佐证）。
---

# Golden 对拍测试手册

本 skill 覆盖 golden 构造 / 判据 / 输出规范 / 可证伪设计。真实源：`projects/MM-确定性/test_matmul_golden.py`（对齐 omni-ops 官方 `test_ai_infra_matmul.py::verify_result`）。

## 明细文件

- [判据](criteria.md) — flat isclose + err_ratio gating，per-dtype tol，cos/rel_l2 仅诊断
- [输入构造](input-construction.md) — 先舍 dtype 再 fp32 累加
- [输出规范](output-format.md) — 全指标 + 余量，PASS/FAIL 都打
- [可证伪设计](falsifiable-design.md) — 阴性对照 + 全码覆盖 + 防假 PASS

## 快速定位

| 需求 | 看 |
|---|---|
| 定 golden 判据/容差 | criteria.md |
| 构造输入避免量化误差 | input-construction.md |
| 规范对拍输出 | output-format.md |
| 防止假 PASS | falsifiable-design.md |

## 边界
- 编译/报错 → ascendc-build-errors
- 差异归因（golden vs 代码谁错）→ ascendc-kernel-semantics-researcher
- 调试工具 → ascendc-debug