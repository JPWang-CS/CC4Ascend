# 源码构建

## 三种包形态

真实源：`ops-transformer_AI/docs/zh/install/compile.md`

| 包形态 | 作用方式 | 支持调用 |
|--------|----------|----------|
| **自定义算子包** | 挂载到 CANN 包，不改原始包，优先级高于原始包 | aclnn + 图模式调用 AI Core / AI CPU 算子 |
| **ops-transformer 整包** | 完整替换 CANN 包对应部分 | aclnn + 图模式调用 AI Core 算子 |
| **ops-transformer 静态库** | 编译为 `libcann_transformer_static.a` + aclnn 头文件；用于二次发布最小化部署 | 仅 aclnn 调用 AI Core 算子 |

> 静态库仅支持 A2/A3/A5；experimental 算子不支持静态库。

---

## 第三方依赖（联网自动下载，离线需手动准备）

| 开源软件 | 版本 |
|---|---|
| googletest | 1.14.0 |
| json | 3.11.3 |
| makeself | 2.5.0 |
| pybind11 | 2.13.6 |
| eigen | 5.0.0 |
| protobuf | 25.1.0 |
| abseil-cpp | 20230802.1 |
| opbase | master（CANN 9.0.0+ 需要） |
| cann-cmake | master-026 |
| ops-tensor | master（CANN 9.1.0+ 需要） |

---

## 联网编译

### 自定义算子包
```bash
bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
# 编译 experimental 算子需加 --experimental
```
- `vendor_name` + `ops` 都不传 → 编译 ops-transformer 整包
- 只传 `vendor_name`（不传 ops）→ 编译所有算子的自定义包

成功标志：
```
Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
```

### ops-transformer 整包
```bash
bash build.sh --pkg --soc=${soc_version}
```
成功标志：
```
Self-extractable archive "cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run" successfully created.
```
`${soc_name}` = `${soc_version}` 去掉 "ascend" 后剩余部分。

### 静态库
```bash
bash build.sh --pkg --static --soc=${soc_version}
```
产物：`build_out/cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}.tar.gz`

---

## 离线编译

1. 检查基础环境已搭好（CANN 包 + 源码）
2. 在联网环境下载第三方依赖：
   - 方式 1：按上表手动下载（确保版本号一致）
   - 方式 2：`python scripts/tools/third_lib_download.py`
3. 上传到离线环境，放在 `third_party/`（推荐）或自定义目录
4. 若放自定义目录，编译命令需额外加 `--cann_3rd_lib_path=${cann_3rd_lib_path}`
5. 安装/卸载命令与联网一致

---

## UT 本地验证

```bash
pip3 install -r tests/requirements.txt

# 编译并执行指定算子和对应功能的 UT
bash build.sh -u --[opapi|ophost|opkernel] --ops=abs

# 编译并执行所有 UT
bash build.sh -u

# 只编译不执行
bash build.sh -u --noexec

# 指定 soc 编译 UT
bash build.sh -u --[opapi|ophost|opkernel] [--soc=${soc_version}]
```

---

## 来源
- `ops-transformer_AI/docs/zh/install/compile.md`