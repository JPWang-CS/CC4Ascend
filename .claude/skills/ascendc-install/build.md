# 编译部署

## 环境变量

```bash
# 默认路径
source /usr/local/Ascend/cann/set_env.sh
# 指定路径
source ${install_path}/cann/set_env.sh
```

## 编译自定义算子包

```bash
bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list} [--experimental]
```

| 参数 | 可选 | 说明 |
|------|:---:|------|
| `--soc` | 必选 | `ascend910b` (A2) / `ascend910_93` (A3) / `ascend950` (A5) |
| `--vendor_name` | 可选 | 自定义算子包名，默认 custom |
| `--ops` | 可选 | 待编译算子列表，如 `--ops=add_example,flash_attention_score` |
| `--experimental` | 可选 | 编译贡献算子时需配置 |

## 安装与卸载

```bash
# 安装
./build_out/cann-ops-transformer-${vendor_name}_linux-${arch}.run
# 安装路径: ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}

# 卸载（手动）
rm -rf ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}
# 编辑 vendors/config.ini，删除对应 load_priority 行
```

## 来源
- `ops-transformer_AI/docs/zh/install/build.md`
