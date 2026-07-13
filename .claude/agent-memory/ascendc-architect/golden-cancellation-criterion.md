---
name: golden-cancellation-criterion
description: BenchmarkCompareStandard 的固定小值门(small=1e-3)在极端K+强对称抵消下会误杀零误差核;修法=噪声地板检测+peak缩放分类+逃生口;证伪要点
metadata:
  type: reference
---

# 极端K+对称抵消会误杀零误差核 —— 判据修法与证伪

> **状态(2026-07-07):此篇描述的 `bench_metrics` peak-scaling 修法已从 `golden_mx_matmul_batch.py` 删除**,判据整体换成 cos为主+相对L2 辅助门(见 [[golden-cos-rell2-criterion]]),cos 判据直接免疫本病态。本篇保留的价值是**病态机理诊断**(fp32 累加噪声地板、近零抵消残差、为何零误差核会被固定小值门误杀)——是 cos 判据免疫的这类误判的根因分析,不再是现行修法。

`golden_mx_matmul_batch.py` 的 `bench_metrics`/`bench_verdict` 是官方 `BenchmarkCompareStandard`(quant_batch_matmul_v3 golden.py + attention compare.py checkResultNew new=True)的忠实复刻。它有一个**对官方标准也成立的盲区**:大 K + 强对称抵消时的近零抵消残差被误判。

## 病理(以 mxfp8_mixed_e5m2e4m3 B13/M1011/K19085/N3873, e5m2×e4m3, out=bf16 为例)
- 输出 peak≈1.0e5、rms≈1.6e4;fp32 累加噪声地板(与量级无关)实测:两种合法求和顺序分歧 |k32-g32| median≈0.002 / max≈0.039;fp32 vs fp64 truth 同量级。
- 官方固定 `small=1e-3` 把量级远低于噪声地板(|g|≈1e-3)的近零残差归入"大值",对其施加**纯相对**误差判据(大值门 `max(bench_dbr,lo=2^-8)*mx=10 ≈ 0.039`)。
- 盈亏点 |golden|≈噪声/门≈0.4:任何 |g|<0.4 的近零点纯粹因 fp32 换序就 dbr≫0.039。**一个零误差理想核(与 golden 同 fp32、仅换求和顺序再舍 bf16)在旧判据即 FAIL(dbr_max≈1.2)**——决定性证据:FAIL 是判据不适配病态 shape,不是算子 bug。
- 头条指标反证算子无真实误差:diff=256=peak∈[2^16,2^17) 的 bf16 半 ULP;cos=0.999999(1-cos=1e-6)= 纯 bf16 舍入正确 fp32 结果的理论值(σ²/2,σ=2^-8/√3)。真大值子集 |g|>1e-4·peak(≈10)重算 dbr_max≈0.0039≈bench → 天然 PASS。

## 修法(已落地 `bench_metrics`,病态才启用、正常逐字走官方口径)
- 检测:`atol_floor = sqrt(K)*rms(golden)*2^-24`(fp32 递归求和误差:K 次加法各 ~u·|部分和|,部分和 RMS≈RMS(golden),独立方均根→sqrt(K));`pathological = atol_floor >= small`。
- 分类随 peak 缩放:`small_eff = max(small, 1e-4*peak)`(1e-4 远高于盈亏点、远低于 bf16 分辨率 2^-8;实测 small_eff≈10)。
- 逃生口:`atol_abs = max(atol_floor, small_eff*lo)`;大值 `|v-g|<=atol_abs` 视为通过,小值错误数阈值改用 atol_abs。实测 atol_floor≈0.135:≥噪声 max 0.039(3.5×余量)且 < small_eff·门(0.4)→ **不遮蔽大值真误差**。

## 不放松正常用例(可证明的单调松弛)
病态分支只让 **dbr_max↓、err_small↓**(大值集收缩+逃生口;逃生口 atol_abs≥small_eff·lo 保证标杆小值错误恒 0)、**RMSE 完全不动** → **绝不把正常 PASS 翻成 FAIL**。非病态时 small_eff/atol_abs 退化为官方 small/small_atol,逐字原口径。回归:int K=256 exact 检测关闭走原路径;K=4096/K=512 检测开启但 NEW==OLD 不翻转。

## 证伪电池(numpy 镜像,全通过;系数 REL_FLOOR∈[1e-5,5e-4]×margin∈[0.3,3] 稳健)
A 零误差核→PASS;B +0.5%全局偏置→FAIL(RMSE);C +5%→FAIL(四门全破);D 最大64点×10%误差→FAIL(dbr_max);E 2万近零点置5.0→FAIL(err_small,证小值逃生口非空头支票);F 边界带 [small_eff,2·small_eff] ×6%→FAIL(dbr_max,证无遮蔽洞)。

## 环境/证据强度(诚实标注)
- 本机无 torch/torch_npu/NPU、框架不 dump 张量 → **真实 NPU 张量不可得**;上述真大值子集/零误差核实验是 **numpy+ml_dtypes 忠实复刻(K=19085 精确、数据分布同 gen_data)**,证的是"判据在此 shape 误杀零误差核类"(强);"故真实算子无真实误差"依赖 diff=256/cos=0.999999(中等)。
- torch 脚本仅 `py_compile` 通过 + 与 numpy 镜像逐行等价核对;未端到端跑(无 torch/NPU),真实用例 PASS 为**推断**。复现证据 `/d/tmp/qbmm_cancellation_probe.py`、`/d/tmp/qbmm_criterion_validation.py`(未碰项目文件)。
- 单把 golden 提 fp64/Kahan **不充分**:算子自身 fp32-cube 在抵消点对任何高精参考都有同量噪声,根子必须在判据口径。

See [[golden-falsifiable-testing]], [[qbmm-batch-status]], [[mx-quant-scale-semantics]].
