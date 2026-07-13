# CANN Simulator 仿真工具

## 概述

CANN Simulator 是 SoC 级芯片仿真工具，用于分析 AI 任务的精度和性能。与板上运行保持二进制兼容，同一 kernel 可同时在仿真和真实 AI 处理器执行。

## 主要功能

- **精度仿真**：bit 级精度结果，验证算子精度
- **性能仿真**：输出指令流水图，定位性能瓶颈

## 使用约束

- 仅支持 Ascend 950PR 芯片（尝鲜版本）
- 仅支持单卡场景（只能设置 0 卡）
- 仅支持 AI Core 计算类算子（不支持 MC2/HCCL 类型）
- 不支持 ARM 环境
- 推荐 CPU 16 核 + 内存 32GB+
- 依赖 CANN 软件包，需先 `source set_env.sh`

## 快速开始

### 编译与执行

```bash
# 1. 编译算子（指定 Ascend950 SoC）
bash build.sh --pkg --soc=Ascend950 --vendor_name=custom --ops=add_example

# 2. 仿真执行
cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report

# 3. 查看结果
cat cannsim_*/cannsim.log
```

### 仿真参数

| 参数 | 说明 |
|------|------|
| `-s` / `--soc_version` | 必选，目标芯片版本（如 Ascend950） |
| `-o` / `--output` | 可选，输出目录 |
| `-g` / `--gen-report` | 可选，自动生成分析报告 |
| `user_app` | 必选，算子可执行文件 |

### 查看指令流水

```bash
# 生成指定核的流水图
cannsim report -e /path/to/cannsim_* -n '0-1, 11-12'

# Chrome 中打开 trace_core0.json
# chrome://tracing → 拖入 json 文件
# W:放大 S:缩小 A:左移 D:右移
```

## 流水字段解读

| 字段 | 含义 |
|------|------|
| VECTOR | 向量运算单元 |
| SCALAR | 标量运算单元 |
| Cube | 矩阵乘运算单元 |
| MTE1 | L1 → {L0A/L0B, UBUF} |
| MTE2 | {DDR/GM, L2} → {L1, L0A/B, UBUF} |
| MTE3 | UBUF → {DDR/GM, L2, L1} 或 L1→{DDR/L2} |
| FIXP | FIXPIPE L0C → OUT/L1 |
| FLOWCTRL | 控制流指令 |
| ICACHELOAD | 未命中 ICache |

## 来源
- `ops-transformer_AI/docs/zh/debug/cann_sim.md`
