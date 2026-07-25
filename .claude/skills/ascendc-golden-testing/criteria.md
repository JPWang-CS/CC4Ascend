# Golden 判据

真实源：`projects/MM-确定性/test_matmul_golden.py:67-271`，对齐 omni-ops 官方 `test_ai_infra_matmul.py::verify_result`。

## 判据（gating，已 trace :269-271）

```python
close = torch.isclose(vf, gf, rtol=tol, atol=tol, equal_nan=True)
err_ratio = float((~close).sum()) / float(gf.numel())
passed = (err_ratio <= ERR_RATIO_TOL) and (m["err_nan"] == 0)
```

- **flat isclose**：`rtol=tol, atol=tol`（同一个 tol），不是逐 dtype 用 2^-mantissa
- **PASS 条件**：`err_ratio ≤ ERR_RATIO_TOL`（isclose=False 占比低于阈值）且无 NaN
- **不用 cos/rel_l2 做 gating**（见下）

## per-dtype tol（已 trace :68，官方值）

```python
CMP_TOL = {torch.float16: 1e-3, torch.float32: 1e-3, torch.bfloat16: 5e-3}
tol = CMP_TOL.get(out_dtype, 1e-3)
```
- fp16 / fp32 = 1e-3
- bf16 = 5e-3

## 为什么不用 2^-mantissa / cos 做 gating（已 trace :69）

旧辅助门用 `2^-mantissa`（逐 dtype mantissa 推阈值）对 **fp32 跨实现累加** 过紧：floor ≈ `sqrt(K)·eps`，导致全 fp32 case 假 FAIL。已弃。

cos / rel_l2 同理：仅诊断打印，不 gating。跨实现累加的乘性偏置会让 cos 在大 K 下偏低，误判。

## cos / rel_l2 诊断（已 trace :244-262）

`_cos_rell2` 算诊断指标（不参与 PASS 判定）：
```python
cos = gf @ vf / (||gf|| * ||vf||)       # 余弦相似
rel_l2 = ||vf-gf|| / ||gf||              # 相对 L2
max_abs_diff, worst (逐元素相对误差), err_nan
```
用途：看量级/偏置趋势，辅助定位，但不下结论。

## 判据选择原则

- **matmul / 累加类**：flat isclose + err_ratio（本文件）
- **elementwise / 无累加**：可紧一点，但仍建议 flat isclose
- **不要用 cos 单独 gating**：乘性偏置盲区
- **不要用 2^-mantissa**：跨实现累加 floor 过紧

## 来源
- `projects/MM-确定性/test_matmul_golden.py:67-271`
- 对齐 omni-ops 官方 `test_ai_infra_matmul.py::verify_result`