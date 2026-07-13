---
name: ascendc-debug
description: 调试与性能调优 AscendC 算子。涵盖 CANN Simulator（A5/Ascend950 SoC 级仿真，精度仿真 + 指令流水图、cannsim record/report、chrome://tracing 查看 trace）、以及上板调试与性能采集（Host 侧 plog 日志、aclGetRecentErrMsg、Kernel 内 PRINTF/DumpTensor、msprof op 上板采集、msProf simulator A2A3 流水图、msDebug 单步调试、MindStudio Insight）。当算子结果不对要定位数值差异、卡死/越界要单步调试、或要分析流水瓶颈/吞吐做性能调优时调用。
---

# AscendC 调试与性能调优

本 skill 收录算子调试定位与性能分析工具链。算子精度异常、卡死、性能不达标时按下表选工具。

## 工具按芯片速查

| 工具 | 用途 | 适用芯片 |
|------|------|----------|
| msProf op | 上板性能采集 | A2A3 + A5 |
| CANN Simulator (cannsim) | 仿真流水图 + 精度 | A5 only |
| msProf simulator | 仿真流水图 | A2A3 only |
| msDebug | 单步调试 | A2A3 + A5 |
| MindStudio Insight | 流水图可视化 | A2A3 + A5 |

## 明细文档

- [算子调试与性能调优](op_debug_prof.md) — Host 侧 plog 日志与 aclGetRecentErrMsg、Kernel PRINTF/DumpTensor、上板 msprof op 采集关键指标、A2A3/A5 两种流水图仿真路径、msDebug 单步调试、调试工具速查。
- [CANN Simulator 仿真工具](cann_sim.md) — A5（Ascend950）SoC 级仿真：精度/性能仿真、使用约束、cannsim record/report 编译与执行、指令流水字段解读（VECTOR/Cube/MTE1-3/FIXP）、chrome://tracing 查看 trace。
