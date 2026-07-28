# Golden 输入构造

真实源：`projects/MM-确定性/test_matmul_golden.py:209` + `projects/QBMM-Batch/golden_mx_matmul_batch.py`。
权威框架：华为算子精度标准第 2 节（`D:\Desktop\TMP\log.txt`）。

## 用例规模与生成规则（华为精度标准第 2 节）

按精度等级（L0/L1/L2）生成，dtype/format/dim/分布/attr 相互正交，所有用例输出总元素 ≥ 100 万：

| 项 | L0（≥5000） | L1（≥10000） | L2（≥30000） |
|---|---|---|---|
| dtype 覆盖 | 所有支持，每种 ≥200 | 每种 ≥700 | 每种 ≥700 |
| format | 覆盖所有支持（ND/NCHW/NZ） | 同 | 同 |
| 维度 | 1-8 维，值 1-2^31，总元素 ≤2^34 | 同 | 同 |
| 维度生成 | 步长(15,16)泛化 | 随机 | 泛化 + 实际业务模型用例(2:1) |
| 值域分布 | 均匀[-5,5] 50% + 正态(μ∈[-100,100],σ∈[1,25]) 50% | 均匀[-0.001,0.001] 10% + 均匀[-5,5] 30% + 正态 40% + 离群点 20% | 同 L1 + 模型用例 |

- **离群点分布（带噪 Noisy）**：k=max(1,⌊n/1000⌋) 个不重复位置 ×1000，多输入各自独立；生成后校验值域仍在算子有效 [min,max] 内
- 正态分布 μ/σ 在范围内随机选取
- attr：标量覆盖等价类、布尔覆盖 True/False、枚举覆盖所有值、组合遍历

### 特殊场景（不计入用例规模）

空 Tensor（某维=0）/ 上下边界（全 1 标量、某维=2^31+1）/ 标量 Tensor[1] / 输入 INF/-INF/NAN / 异常值（边界外、约束外）覆盖。

## 核心原则：输入先舍到 dtype，再 fp32 累加（已 trace :209）

```python
# 正确：输入先舍到实际 dtype，golden 用 fp32 累加
x1 = x1.to(x1_dt)           # 输入舍到算子实际 dtype（fp16/bf16/int8）
golden = x1.float() @ x2.float()   # golden 在 fp32 累加
```

## 为什么必须先舍 dtype（已验证）

若用 **fp32 输入** 做 golden，再与 fp16/bf16 实现比对：
- fp32 输入无量化误差
- fp16/bf16 实现输入有 ULP 级量化误差
- golden 没对齐实现的输入量化 → 多一重误差 → err_ratio 虚高 2~22% → 假 FAIL

正确做法：输入 `.to(dtype)` 让 golden 和实现**从同一份量化后的输入出发**，golden 只负责"用 fp32 累加"这一步更精确。

## matmul golden 累加（已验证）

```python
# golden: a.to(out_dtype).float() @ b.to(out_dtype).float()
# 对齐 op 的 a.to(dtype) 行为
a_fp = a.to(out_dtype).float()
b_fp = b.to(out_dtype).float()
golden = a_fp @ b_fp   # 或带 scale 的量化 matmul golden
```

## 量化 matmul 输入

- weight / activation 按算子要求 dtype 构造（int8/fp16/bf16）
- scale 按 quant 粒度（pertoken/perchannel/perblock/MX）构造对应 shape
- golden 里 scale apply 用 fp32

## 避免的输入构造

- ❌ 全 1 / 全 0 / 单值（掩盖精度问题）
- ❌ 纯 2 的幂（掩盖 rounding）
- ❌ 太小量级（floor 噪声主导）
- ❌ fp32 输入直接做 golden 再比 fp16 实现（量化误差不对齐）

## 推荐

- 随机但有区分度（覆盖值域）
- 多 batch / 多 shape 覆盖边界
- 含负数 / 大小值混合（激活 rounding）