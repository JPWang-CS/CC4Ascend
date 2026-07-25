# 编译部署 — build.sh 参数

真实源：`ops-transformer_AI/build.sh` + `ops-transformer_AI/docs/zh/install/build.md`

## 环境变量

```bash
source /usr/local/Ascend/cann/set_env.sh       # 默认路径
source ${install_path}/cann/set_env.sh          # 指定路径
```

---

## 常用编译命令

```bash
# 自定义算子包（部分算子）
bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list}

# ops-transformer 整包（全部算子）
bash build.sh --pkg --soc=${soc_version}

# 静态库
bash build.sh --pkg --static --soc=${soc_version}

# experimental 算子需加 --experimental
```

---

## 全量参数（build.sh --help 真值）

### 通用
| 参数 | 说明 |
|------|------|
| `-j${n}` | 编译线程数，默认 8 |
| `-v` | 查看 CMake 编译配置 |
| `-O${n}` | 编译优化 O0/O1/O2/O3 |
| `-u` | 编译所有 UT |
| `--help / -h` | 打印帮助 |

### 包/目标
| 参数 | 说明 |
|------|------|
| `--pkg` | 生成安装包（含 kernel bin）；不可与 `-u` 或 `--ophost/--opapi/--opgraph` 同时用 |
| `--ops` | 指定算子（snake name，逗号分隔）；不可与 `--ophost/--opapi/--opgraph` 同时用 |
| `--soc` | NPU 型号，每次只支持 1 个；支持 `[ascend910b ascend910_93 ascend950 ascend310p kirinx90 kirin9030 mc62]` |
| `--vendor_name` | 自定义算子包名，默认 custom |
| `--jit` | 编译整包不含 kernel bin（图运行态在线编译），提升编译速度 |
| `--static` | 生成静态库；仅 A2/A3/A5 |
| `--experimental` | 编译 experimental 目录下的用户算子 |
| `--build-type` | Release / Debug，默认 Release |
| `--make_clean` | 清理编译产物 |

### 库目标（互斥）
| 参数 | 说明 |
|------|------|
| `--ophost` | 编译 libophost_transformer.so |
| `--opapi` | 编译 libopapi_transformer.so |
| `--opgraph` | 编译 libopgraph_transformer.so |
| `--opkernel` | 编译二进制内核 |

### UT 目标
| 参数 | 说明 |
|------|------|
| `--ophost_test` | 编译 ophost UT（= `-u --ophost`） |
| `--opapi_test` | 编译 opapi UT |
| `--opkernel_test` | 编译 opkernel UT |

### 调试/特殊
| 参数 | 说明 |
|------|------|
| `--debug` | 调试模式 |
| `--asan` | host 侧 asan |
| `--valgrind` | valgrind 跑 ut（禁用 asan/noexec） |
| `--cov` | 本地 UT 覆盖率统计 |
| `--oom` | kernel 侧 oom 内存检测 |
| `--bisheng_flags` | 毕昇编译器参数；不可与 `--mssanitizer/--oom/--dump_cce` 同时用 |
| `--kernel_template_input` | 指定 kernel tilingKey 模板（仅 1 个算子，不编译依赖算子） |
| `--cann_3rd_lib_path` | 离线编译第三方库目录 |
| `--noexec` | 只编译 ut 不执行 |
| `--version` | 指定版本 |

### 样例/脚手架
| 参数 | 说明 |
|------|------|
| `--run_example` | 编译并执行 test_aclnn_xxx.cpp / test_geir_xxx.cpp |
| `--simulator` | 仿真器模式（目前仅 Ascend950） |
| `--genop` | 创建 AI Core 算子初始目录 |
| `--genop_aicpu` | 创建 AI CPU 算子初始目录 |

---

## 参数冲突规则（真实，易踩坑）

- `--ops` 与 `--ophost / --opapi / --opgraph` 不可同时用
- `--pkg` 与 `-u` 或 `--ophost / --opapi / --opgraph` 不可同时用
- `--bisheng_flags` 与 `--mssanitizer / --oom / --dump_cce` 不可同时用
- `--kernel_template_input` 只能指定 1 个算子

---

## 包产物命名与安装

### 自定义算子包
```bash
# 产物
cann-ops-transformer-${vendor_name}_linux-${arch}.run

# 安装（安装路径: ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}）
./cann-ops-transformer-${vendor_name}_linux-${arch}.run

# 指定安装路径时，使用前需 source
source ${install-path}/vendors/${vendor_name}/bin/set_env.bash

# 卸载（自定义包不支持卸载命令，手动删除）
rm -rf ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}
# 并删除 vendors/config.ini 中 load_priority 对应 ${vendor_name} 的配置项
```

### ops-transformer 整包
```bash
# 产物
cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run

# 安装
./cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run --full --install-path=${install_path}

# 卸载
./${install_path}/cann/share/info/ops_transformer/script/uninstall.sh
```

### 静态库
```bash
# 产物
cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}.tar.gz

# 解压
tar -zxvf ./cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}.tar.gz -C ${static_lib_path}
```

---

## 来源
- `ops-transformer_AI/build.sh`
- `ops-transformer_AI/docs/zh/install/build.md`