---
name: ascendc-build-errors
description: AscendC 编译/构建/checker/安装失效排查手册。覆盖构建链（build.sh/CMake/install/package/stale binary）、aclnn checker 报错码（EZ 系列）、框架/注册问题（op_def/proto/schema/torch binding/符号）、改了但没生效（so 未替换/常量指纹/未重装）。当编译失败、报错码定位、checker 拒绝、安装后行为没变、或符号找不到时调用。内容来自真实 build 链 + 已验证项目教训。
---

# 编译/构建/安装失效排查手册

本 skill 是工程失效排查手册，不是报错码字典。按失效层分类，每类给：现象 → 根因机制 → 诊断 → 修法。

## 按失效层选文件

- [构建链](build-chain.md) — build.sh / CMake / install / package / 依赖 / 离线
- [Checker 报错码](checker-errors.md) — aclnn checker EZ 系列报错（EZ0012/0013/0020...）
- [框架/注册问题](registration.md) — op_def/proto/schema/torch binding/符号找不到/日志宏冲突
- [改了但没生效](stale-deployment.md) — stale binary / so 未替换 / 常量指纹 / 未重装

## 快速定位

| 现象 | 先看 |
|---|---|
| build.sh 编译失败 / 依赖问题 | build-chain.md |
| 算子编译过但 aclnn 调用报 EZ 错误 | checker-errors.md |
| 编译过但链接/符号/注册问题 | registration.md |
| 代码改了重编译了但行为没变 | stale-deployment.md |
| pip / whl 离线卡死 | build-chain.md §离线 |

## 边界
- 怎么调用算子 → ascendc-operator-invocation
- 怎么编译/参数 → ascendc-install
- 精度/golden 问题 → ascendc-golden-testing