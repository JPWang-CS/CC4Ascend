---
name: ascendc-golden-testing
description: AscendC golden 对拍的判据、输入构造、输出规范、可证伪测试设计。判据对齐华为算子精度标准（MARE/MERE/RMSE ratio + L0/L1/L2 + 小值域 + 复检，D:\Desktop\TMP\log.txt）；无 GPU 标杆时降级用 flat isclose+err_ratio 简化判据。当要写 golden 脚本、定判据/容差、构造测试输入、设计阴性对照、或规范对拍输出时调用。内容来自华为精度标准 + 真实 project golden 脚本（带行号佐证）。
---

# Golden 对拍测试手册

本 skill 覆盖 golden 构造 / 判据 / 输出规范 / 可证伪设计。
**权威框架**：华为算子精度标准（`D:\Desktop\TMP\log.txt`，4 类算子分治 + 浮点 MARE/MERE/RMSE ratio + L0/L1/L2 + 小值域 ErrorCount + bootstrap 复检）。

## 明细文件

- [判据](criteria.md) — **华为精度标准为主**（4 类算子、MARE/MERE/RMSE ratio、L0/L1/L2 阈值、小值域、复检、INF/NAN）；flat isclose+err_ratio 为无 GPU 标杆时的简化 fallback
- [输入构造](input-construction.md) — L0/L1/L2 规模/分布生成规则 + 先舍 dtype 再 fp32 累加
- [输出规范](output-format.md) — 全指标 + 余量，PASS/FAIL 都打
- [可证伪设计](falsifiable-design.md) — 阴性对照 + 全码覆盖 + 防假 PASS

## 快速定位

| 需求 | 看 |
|---|---|
| 定算子属于哪类 / 用哪套判据 | criteria.md「算子分类」 |
| 定浮点 golden 判据/容差 | criteria.md「浮点算子判据」 |
| 量化算子（QBMM/quant_matmul）判据 | criteria.md「量化算子判据」 |
| 构造输入规模/分布 | input-construction.md「L0/L1/L2」 |
| 构造输入避免量化误差 | input-construction.md「先舍 dtype」 |
| 规范对拍输出 | output-format.md |
| 防止假 PASS | falsifiable-design.md |
| 只影响 tiling/UB 预算的参数（oracle 不感知）怎么验证 | falsifiable-design.md「只影响 tiling/UB 预算的参数」 |

## 边界
- 编译/报错 → ascendc-build-errors
- 差异归因（golden vs 代码谁错）/ 选标杆 / 定 L 级 → ascendc-kernel-semantics-researcher
- 调试工具 → ascendc-debug
