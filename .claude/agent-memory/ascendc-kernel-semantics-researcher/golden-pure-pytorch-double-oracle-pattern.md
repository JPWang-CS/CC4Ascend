---
name: golden-pure-pytorch-double-oracle-pattern
description: 纯 PyTorch golden 的双 oracle 互比模式 + 离线机结构性自检降级策略，用于 custom op 无 npu 环境时的 oracle 可证伪
metadata:
  type: feedback
---

纯 PyTorch 项目 golden（G3，不调 npu 算子）的可证伪设计采用**双 oracle 互比**：
- oracle A = 继承自 atk 的原版（如 `causal_conv1d_golden`，einsum/向量化路径）
- oracle B = 独立朴素实现（显式循环/不同代码路径）
- 两者均 fp64，ratio 应→0；任何非零 ratio 指向 oracle 实现 bug（任一方）

**Why:** 单 oracle 自比无法发现 oracle 本身的语义 bug（oracle 复制实现 = 弱证据，违反 [[verification-discipline]] "copied oracle ≠ trustworthy oracle"）。双独立实现互比才能给数值正确性证据。

**How to apply:**
- 项目 golden G3 类任务（不依赖 npu binding）默认写双 oracle
- 朴素版必须走**不同代码路径**（显式循环 vs einsum/向量化），不能只是改个函数名
- 离线开发机（无 numpy/torch）降级：用纯 python 写结构性自检脚本（shape/offset/cache 越界/boundary assert），覆盖 m/K/accepted 边界组合，验证 oracle 的硬约束（atk 里真正有 assert 的那些）
- 数值正确性必须显式标 WEAK（本机未跑），不能假装 PASS
- 自检断言只检查 atk oracle 真正的 assert（如 `assert boundary_idx > 0`），不要加自己的约束（如 `reset_idx > seq_len` 是 python slice 合法行为不是 bug）

参考实现：`D:\Desktop\Code\CC4Ascend\projects\omni-fused_causal_conv1d算子新增投机tokens数参数\golden\golden_fused_causal_conv1d_mtp.py`
