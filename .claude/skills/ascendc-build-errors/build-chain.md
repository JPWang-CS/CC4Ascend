# 构建链失效排查

真实源：`ops-transformer_AI/build.sh` + `docs/zh/install/`。

## 自建单算子 custom 包 host tiling-parse 失败（561103）

### 现象
自建单算子 custom 包（`build.sh --pkg --soc=... --ops=<op>`）经 `ASCEND_CUSTOM_OPP_PATH` 挂载，报：
```
561103 / InitTilingParseCtx failed / tiling compile info parse failed
```
清掉 `ASCEND_CUSTOM_OPP_PATH`（用 builtin）则全过。

### 根因机制（已验证）
不是缺 op_proto/op_tiling/aclnn/op_master，也不是 json compile_info。
真实机制：TilingParse/Tiling 调 `TilingPrepareForOpCache`，从 toolkit 的 `libophost_comm_legacy.so` 经 `LegacyCommonMgr` dlsym `LegacyTilingParsePrepareForOpCache`。custom 的 `libcust_opmaster_rt2.0.so` 用脆弱的多级 `../` 回退到 builtin opp 找 legacy.so；若 vendor 目录未正确嵌套在 toolkit opp 下，回退落空 → dlopen 失败。builtin 包同目录找 legacy.so → 总是稳。

### 诊断
```bash
ASCEND_GLOBAL_LOG_LEVEL=0 grep "LegacyCommonMgr|libophost_comm_legacy|dest func.*null"
nm -D  # 对比 cust opmaster(undefined) 与 toolkit legacy so(defined) 符号
```

### 修法
优先用 builtin 包；若必须 custom，确保 vendor 目录正确嵌套在 toolkit opp 下。

## pip whl 离线编译卡死

### 根因（已验证）
`add_es_library_and_whl` 真实打包命令 `${Python3_EXECUTABLE} -m pip wheel . --no-deps --wheel-dir=...`。只硬编码 `--no-deps`，build-isolation 未强制 → `PIP_NO_BUILD_ISOLATION=1` 被尊重。pip 默认 build isolation 建隔离 env 需拉 setuptools/wheel → 无网 → 卡死。

### 修法
```bash
export PIP_NO_BUILD_ISOLATION=1
```
或跳过 whl：手动 `touch ${PYTHON_BUILD_DIR}/whl_generated.flag`（make 视为 up-to-date）。

## build.sh 常见参数误用

- `--ops` 与 `--ophost/--opapi/--opgraph` 不可同时用
- `--pkg` 与 `-u` 或 `--ophost/--opapi/--opgraph` 不可同时用
- `--soc` 每次只支持 1 个（transformer 7 个 / ops-nn 12 个 SoC 可选值）
- 详参数见 ascendc-install

## 离线编译依赖缺失

联网自动下载第三方依赖；离线需预先放 `third_party/` 或用 `--cann_3rd_lib_path`。依赖清单见 ascendc-install §compile。