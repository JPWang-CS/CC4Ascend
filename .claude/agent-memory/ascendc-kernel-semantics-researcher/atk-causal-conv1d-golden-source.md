---
name: atk-causal-conv1d-golden-source
description: atk fused_causal_conv1d 的 golden oracle 源位置 + continue/prefill 两 executor 的 oracle 语义等价结论
metadata:
  type: reference
---

atk `AiInfraFusedCausalConv1d` 的 golden oracle（`causal_conv1d_golden`）位置：
- `D:\Desktop\Code\CustomOP\atk\AiInfraFusedCausalConv1d\executor_ai_infra_fused_causal_conv1d_continue.py:73-248`（decode 主用）
- `D:\Desktop\Code\CustomOP\atk\AiInfraFusedCausalConv1d\executor_ai_infra_fused_causal_conv1d_pref.py:95-268`（prefill）

**关键结论**：continue 和 prefill 两个 executor 的 `causal_conv1d_golden` 逐行语义等价（已 trace 比对），唯一差异在 inplace 返回值包装。同一个 oracle，TilingKey=0/1/2 三个 kernel 模板共用。

atk cpu 真值用 `x.to(torch.float64)` 跑（continue executor:694-697），即 ground truth 是 fp64 卷积。

判据配置在 json `standard.acc.cv_fused_double_benchmark`：`max_re_ratio=5, avg_re_ratio=1.5, root_mean_squared_ratio=1.5`（双标杆 ratio 制）。
