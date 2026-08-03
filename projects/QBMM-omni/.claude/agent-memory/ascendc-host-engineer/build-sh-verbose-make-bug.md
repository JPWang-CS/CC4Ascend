---
name: build-sh-verbose-make-bug
description: build.sh build_binary_from_json 三处 make 调用误用 ${VERBOSE:+--verbose}, 位置(--后=make参数)+测试(:非空,omni VERBOSE="false"也非空)双重错, make 不认 --verbose 报 Usage 退出中断所有算子 build; 已删
metadata:
  type: feedback
---

`build.sh` 的 `build_binary_from_json` 三处 `cmake --build ... --target xxx -- ${VERBOSE:+--verbose} -j N` 是 bug, 已修(2026-08-03)。

**根因(双重错):**
1. 位置错: `${VERBOSE:+--verbose}` 在 `--` **后** → 传给底层 make。但 GNU make **没有 `--verbose` 选项** → 打印 `Usage: gmake` 退出。`--verbose` 是 **cmake** 的 flag, 必须在 `--` **前**: `cmake --build path --verbose --target xxx`。
2. 测试语义错: omni build.sh 的 `VERBOSE` 变量是字符串 `"false"`(默认)/`"true"`(第 27 行 + 第 368 行 --verbose 参数)。`${VAR:+word}` 只测**非空**, "false" 也非空 → **always-on** 产出 `--verbose`。即使用户 env 不设 VERBOSE 也触发。

**Why:** 上板实证 /tmp/v3_build.log 第 65-67 行: `binary.json build start` 后紧跟 `Usage: gmake [options] [target] ...` 中断。用户 env 里 VERBOSE 之前查 opbuild 用过 VERBOSE=1 也加剧, 但根本原因是 omni 的 VERBOSE 语义(字符串 bool)与 nn 不同。

**对比 nn:** nn build.sh:1212/1224/1230 用裸 `${VERBOSE}`(无 `:+`), nn 里 VERBOSE 默认 `""`, `-v` 时 `VERBOSE="VERBOSE=1"`(make 变量赋值语法 `make VERBOSE=1`, make 认识)。omni 的 VERBOSE 是 cmake bool 风格, 不能照抄 nn 的 `${VERBOSE}` 也不能用 `${VERBOSE:+--verbose}`。

**How to apply:**
- omni build.sh 的 cmake --build 调用: 默认**不要传**任何 verbose 透传, cmake --build 默认行为够。要调试时用 `cmake --build path --verbose --target xxx`(verbose 在 -- 前, 是 cmake flag)。
- `${VAR:+word}` 慎用: 如果 VAR 可能是 "false"/"0"(非空但语义 falsy), `:+` 会误产 word。对 bool 风格变量用 `[[ "$VAR" == "true" ]]` 显式判断。
- 改 cmake --build 命令时务必区分 `--` 前后: 前是 cmake 参数(--verbose/-j/--target), 后是底层构建系统参数(make/ninja 的 -j / var=val)。
- 关联 [[binary-json-cmake-target-vs-build-driver]]: build_binary_from_json 即该 entry 落地的三步序列。

**修法(已落地):** 三处 `${VERBOSE:+--verbose}` 全删, 并给三个 guard 的 else 分支加 `[binary.json] no xxx target, skip` 日志让上板可观测跳过。
