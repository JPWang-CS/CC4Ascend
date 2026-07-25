# Golden 输入构造

真实源：`projects/MM-确定性/test_matmul_golden.py:209` + `projects/QBMM-Batch/golden_mx_matmul_batch.py`。

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