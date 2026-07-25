# 环境部署

## 安装方式总览

本项目提供三种搭建昇腾环境的方式（真实源：`ops-transformer_AI/docs/zh/install/quick_install.md`）：

| 安装方式 | 使用场景 |
|----------|----------|
| **CANNLab** | 无昇腾设备；一站式云平台，默认最新商发版 CANN |
| **Docker** | 有昇腾设备需快速搭建；当前仅 Atlas A2 系列 + Ubuntu |
| **手动安装** | 有昇腾设备，想体验 master 或手动装 CANN |

> 编译态：只编译不运行，只需 CANN toolkit 包。  
> 运行态：编译运行或纯运行，需驱动+固件、CANN toolkit、CANN ops。

---

## 方式 1：CANNLab

一站式开发平台，已预装驱动固件、软件包、依赖。通过项目页面"CANNLab"按钮进入。默认最新商发版 CANN。

---

## 方式 2：Docker

1. **安装驱动与固件**（运行态依赖）：`npu-smi info` 检查；若无可参考《CANN快速安装》。
2. **拉取镜像**：从昇腾镜像仓库拉取已预集成 CANN 及依赖的镜像。
3. **启动容器**：需 `--device /dev/davinci0`、`davinci_manager`、`devmm_svm`、`hisi_hdc`，并挂载驱动库 / dcmi / npu-smi。

> Docker 当前仅适用于 Atlas A2 系列、Ubuntu。

---

## 方式 3：手动安装

### 3.1 安装驱动与固件（运行态依赖）

`npu-smi info` 检查；缺失时参考《CANN快速安装》。

### 3.2 安装 CANN 包

- **体验 master**：从 artifactory 下载最新时间版本，按架构/OS 选 toolkit + ops 包：
  ```bash
  bash ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
  bash ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
  ```
  - ops 包是运行态依赖；仅编译可不装
  - `${soc_name}` = NPU 型号名称
  - `${install_path}` 需与 toolkit 同路径

- **体验已发布版本**：CANN 官网下载中心，选 CANN 8.5.0+ 发布版本。

### 3.3 安装基础依赖

真实依赖清单（`quick_install.md:127-136`）：

| 依赖 | 要求 |
|------|------|
| python | >= 3.7.0（建议 <= 3.10） |
| gcc | >= 7.3.0 |
| cmake | >= 3.16.0 |
| pigz | 可选，建议 >= 2.4 |
| dos2unix | 必需 |
| make | 必需 |
| patch | 必需 |
| googletest | 仅 UT 时依赖，建议 release-1.11.0 |

一键安装：
```bash
bash install_deps.sh          # 项目根目录脚本
pip3 install -r requirements.txt
```

---

## 环境验证

```bash
# 检查 NPU 设备
npu-smi info

# 检查 CANN toolkit 版本
cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info
# 检查 CANN ops 版本
cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
```

`${arch}` 通过 `uname -m` 查询（aarch64 / x86_64）。

## 环境变量

```bash
# 默认路径（root）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径
source ${install_path}/cann/set_env.sh
```

---

## 来源
- `ops-transformer_AI/docs/zh/install/quick_install.md`