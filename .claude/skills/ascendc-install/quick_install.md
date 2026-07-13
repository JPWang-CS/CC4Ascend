# 快速安装指南

## 环境准备

### 硬件
- Atlas A2/A3 训练卡 或 Ascend 950
- x86_64 / aarch64 架构

### 软件
- Linux (Ubuntu 20.04+ / CentOS 7+ / openEuler 22.03+)
- CANN toolkit (社区版或商业版) 8.0.RC1+
- Python 3.8+
- CMake 3.14+
- GCC 7.3+

## 源码获取

```bash
git clone https://gitcode.com/cann/ops-transformer.git
```

## 安装 CANN Toolkit

### 社区版
下载对应架构和 OS 的 run 包，执行：
```bash
chmod +x Ascend-cann-toolkit_*.run
./Ascend-cann-toolkit_*.run --install
source /usr/local/Ascend/cann/set_env.sh  # root用户
# 或 source ${HOME}/Ascend/cann/set_env.sh  # 非root用户
```

### 依赖安装
```bash
bash install_deps.sh
```

## 编译全部算子

```bash
bash build.sh --pkg --soc=${soc_version}
```

SoC版本:
- `ascend910b` → Atlas A2 训练卡
- `ascend910_93` → Atlas A3 训练卡
- `ascend950` → Ascend 950

## 验证安装

```bash
# 运行示例算子
bash build.sh --run_example add_example eager cust
```

## 来源
- `ops-transformer_AI/docs/zh/install/quick_install.md`
