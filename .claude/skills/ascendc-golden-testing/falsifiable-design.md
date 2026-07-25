# 可证伪测试设计

原则：**测试必须能在实现错误时 FAIL**，否则无验证价值。一个不能 FAIL 的测试等于没有测试。

## 阴性对照（必须）

### 做法
故意把 golden 改错（如翻转符号、偏移 1、换 scale），确认测试 **FAIL**。

### 为什么
- 确认判据不是"永远 PASS"（太松）
- 确认 harness 真的比对了输出（没跳过比对）
- 确认 err_ratio 计算正确（不是恒 0）

### 实践
```python
# 阴性对照：故意污染 golden，验证测试 FAIL
golden_wrong = golden + 1e-2  # 注入误差
assert not passed  # 若 passed，说明判据/harness 有问题
```

## 全码覆盖

### 要覆盖
- **dtype 全覆盖**：fp16 / bf16 / fp32 / int8（每种 tol 不同）
- **shape 边界**：最小（1×1）、非对齐（尾块）、大 shape（多 tile）
- **量化粒度**：pertoken / perchannel / perblock / MX（如适用）
- **batch 维**：batch=1（退化）、batch>1、无 batch
- **特殊 layout**：transpose / NZ（如适用）

### 易漏
- 尾块（非 tile 整数倍）
- 单 batch（退化为无 batch 路径）
- 小 shape（走 fast path）
- K=1 / N=1（退化维度）

## 防假 PASS 模式

### 假 PASS 来源
1. **cherry-picked 数据**：全 1/全 0/2 的幂 → 掩盖 rounding
2. **oracle 抄实现**：golden 照着 kernel 逻辑写 → 永远自洽
3. **tol 太松**：tol=0.1 什么都能过
4. **只测小 shape**：小 shape 误差小，大 shape 累加才暴露
5. **没阴性对照**：不知道测试能不能 FAIL

### 防
- 随机有区分度输入（见 input-construction.md）
- golden 独立实现（numpy/torch 原生，不抄 kernel）
- tol 用官方值（fp16/fp32=1e-3, bf16=5e-3）
- 含大 shape / 多 batch case
- 必须有阴性对照

## 多例泛化

- 单 case PASS ≠ 正确
- 要多 shape / 多 dtype / 多模式都 PASS
- 边界 case 单独标注（擦边的要警觉）

## 来源
- agent memory `golden-falsifiable-testing` / `repo-mxfp4-tests-broken`
- `projects/MM-确定性/test_matmul_golden.py` 实践