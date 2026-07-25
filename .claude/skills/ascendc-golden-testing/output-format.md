# Golden 输出规范

原则：**PASS/FAIL 都打全指标 + 余量**，拒绝黑盒 PASS。改打印不改判定。

## 每行输出（已验证 test_matmul_golden.py 实践）

```
[case] shape=... dtype=... tol=1e-3 | err_ratio=3.2e-5/1e-4 PASS | cos=0.99999 rel_l2=1e-4 max_abs=2.1e-3
```

字段：
- **case 标识**：shape / dtype / 模式（一眼定位）
- **gating 指标**：err_ratio / ERR_RATIO_TOL（PASS/FAIL + 余量倍数）
- **诊断指标**：cos / rel_l2 / max_abs_diff（量级 + 偏置）
- **结论**：PASS / FAIL

## 为什么 PASS 也要打全指标

- 看余量：err_ratio=3e-5 vs tol=1e-4，稳过；err_ratio=9e-5 vs 1e-4，擦边
- 擦边的 PASS 可能下次 case 变就 FAIL，需提前发现
- cos 量级异常（如 0.99 而非 0.9999）提示系统性偏置，即使 err_ratio 过了

## 余量（margin）

- `err_ratio / ERR_RATIO_TOL` 的比值：越小越稳
- `max_abs_diff / tol`：最大单点误差离 tol 多远
- 擦边（比值 > 0.5）要标注，提醒不稳定

## bench_metrics 纯增量返回（已验证）

判据函数返回 metrics dict（cos/rel_l2/err_ratio/max_abs_diff/worst），调用方打印。判据逻辑（PASS/FAIL）不依赖诊断指标，只依赖 err_ratio + err_nan。打印是增量的，不改判定。

## 禁止

- ❌ 只打 `PASS` / `FAIL` 不打数字
- ❌ 只 FAIL 打、PASS 不打
- ❌ 打了数字但不打 tol / 无法判断余量

## 来源
- `projects/MM-确定性/test_matmul_golden.py` 输出实践
- agent memory `test-output-verbose-margins`