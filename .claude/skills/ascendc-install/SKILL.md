---
name: ascendc-install
description: 安装、编译、构建 ops-transformer AscendC 算子工程，以及工程目录结构说明。涵盖环境部署（CANNLab / Docker / 手动安装三种方式）、build.sh 全量参数（--pkg/--soc/--ops/--vendor_name/--static/--jit/--experimental/--cann_3rd_lib_path 等）、三种算子包形态（自定义包 / ops-transformer 整包 / 静态库）、安装/卸载、UT 本地验证、离线编译、以及项目目录结构（单算子 op_host/op_kernel/op_api/op_graph/tests、torch_extension）。当要搭环境、跑 build.sh 编译算子、配 SoC 版本、选包形态、离线编译、跑 UT、或弄清算子工程文件该放哪里时调用。
---

# ops-transformer 安装、编译与目录结构

本 skill 覆盖环境搭建、算子编译部署、UT 本地验证、离线编译、以及算子工程的目录组织。真实源 = `ops-transformer_AI/` 下的 `build.sh` + `docs/zh/install/`。

## SoC 版本速查

`build.sh` 真实支持的 SoC（`ops-transformer_AI/build.sh:46,327`）：

| SoC 参数 | 产品 |
|----------|------|
| `ascend910b` | Atlas A2（默认） |
| `ascend910_93` | Atlas A3 |
| `ascend950` | Ascend 950 (A5)，含 950DT / 950PR |
| `ascend310p` | Ascend 310P |
| `kirinx90` | Kirin X90 |
| `kirin9030` | Kirin 9030 |
| `mc62` | MC62 |

> 文档主线聚焦前三个（A2 / A3 / A5），但 `build.sh` 实际接收 7 个。**不要只按 3 个来判。**

## 明细文档

- [环境部署](quick_install.md) — 三种安装方式（CANNLab / Docker / 手动）、依赖清单、环境变量、环境验证。
- [源码构建](compile.md) — 三种包形态、联网/离线编译、第三方依赖版本表、UT 本地验证。
- [编译部署](build.md) — `build.sh` 全量参数、参数冲突规则、包产物命名与安装/卸载。
- [项目目录结构](dir_structure.md) — 顶层目录、单算子 `op_host/op_kernel/op_api/op_graph/tests` 组织、torch_extension 结构、可选项说明。