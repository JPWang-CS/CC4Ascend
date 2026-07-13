---
name: ascendc-install
description: 安装、编译、构建 ops-transformer AscendC 算子工程，以及工程目录结构说明。涵盖快速安装（CANN toolkit 安装、依赖、source set_env.sh）、源码构建（联网/离线编译、硬件软件要求、生成 .run 产物）、编译部署（build.sh --pkg 参数 --soc/--vendor_name/--ops、自定义算子包安装与卸载）、以及项目目录结构（ops-transformer 顶层目录、单算子 op_host/op_kernel/arch22/arch35 组织规律）。当要搭环境、跑 build.sh 编译算子、配 SoC 版本（ascend910b/ascend910_93/ascend950）、或弄清算子工程文件该放哪里时调用。
---

# ops-transformer 安装、编译与目录结构

本 skill 覆盖环境搭建、算子编译部署、以及算子工程的目录组织。配环境或跑构建命令时按下表查阅。

## SoC 版本速查

| SoC 参数 | 芯片 |
|----------|------|
| `ascend910b` | Atlas A2 |
| `ascend910_93` | Atlas A3 |
| `ascend950` | Ascend 950 (A5) |

## 明细文档

- [快速安装指南](quick_install.md) — 硬件/软件要求、源码获取、CANN toolkit 社区版安装、install_deps.sh 依赖、编译全部算子、验证安装。
- [源码构建指南](compile.md) — 环境准备、联网/离线编译、生成 .run 自解压产物。
- [编译部署](build.md) — 环境变量、build.sh --pkg 参数详解（--soc/--vendor_name/--ops/--experimental）、自定义算子包安装与卸载。
- [项目目录结构](dir_structure.md) — ops-transformer 顶层目录、单算子 op_host/op_kernel/op_api/op_graph/tests 结构、A2A3 平铺 vs arch22/arch35 子目录的代码组织规律。
