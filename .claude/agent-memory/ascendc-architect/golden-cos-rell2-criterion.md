---
name: golden-cos-rell2-criterion
description: cos-primary + relative-L2 auxiliary double-gate verdict for quant-matmul golden; threshold derivation (cos 0.9999, relL2=2^-mantissa per out_dtype), why relL2 plugs cos's multiplicative-bias blind spot, falsification battery, and the C3 honesty caveat (absolute bias probe is scale-dependent)
metadata:
  type: reference
---

# cos 为主 + 相对L2 辅助门 的量化 matmul 判据 (取代官方 BenchmarkCompareStandard)

`golden_mx_matmul_batch.py` 2026-07-07 起判据换范式。业界 matmul/量化最主流是 **cos 为主 + 双保险辅助门**,天然免疫官方固定小值门在近零抵消/大K/fp4-bias 擦门处误杀零误差核的病态(见 [[golden-cancellation-criterion]] [[golden-fp4-bias-nearzero]] —— 那两篇诊断的病态,cos 判据直接免疫,不再需要 peak-scaling 补丁)。

## 判据
PASS ⟺ **主门 cos ≥ COS_TOL** 且 **辅助门 相对L2 ≤ RELL2_TOL[out_dtype]** 且 nan/inf 掩码与 golden 一致。
- 主门 `cos = ⟨npu,golden⟩/(‖npu‖‖golden‖)`(拉平 fp64)。方向一致性,免疫一致乘性偏置与近零点。
- 辅助门 `相对L2 = ‖npu-golden‖₂/‖golden‖₂`。分母是整张范数(大值主导)→ 近零点噪声贡献可忽略(免疫近零点);一致乘性偏置 δ 使 相对L2 = |δ| 恰好被抓 → 堵住 cos 唯一盲区。

## 阈值推导 (numpy 镜像实测,真实 K + gen_data 分布)
- `COS_TOL = 0.9999`。bf16 舍入 cos 地板 ≈ 1-(2^-8/√3)²/2 ≈ 1-2.5e-6(实测 1-cos≈1.3e-6,真实 bf16 误差优于最坏均匀)。取 0.9999 留 ~75× 余量:正确核绝不误判,方向真错(漏 batch 偏移/错位/垃圾)把 cos 打到 ≪0.99。cos 对一致偏置几乎不动 → 精细鉴别全交辅助门,主门取宽余量即可。
- `RELL2_TOL = 2^-mantissa` **逐 out_dtype**:bf16=2^-8≈3.9e-3、fp16=2^-11≈4.9e-4、fp32=2^-24。推导:纯舍入下 相对L2 ≈ 每元素相对舍入尺度 σ=2^-mantissa/√3;取 2^-mantissa 使阈值≈1.41σ,实测地板落门下 **2.36×(bf16/fp16 一致)**,破门盈亏偏置 bf16≈0.35%/fp16≈0.05%。**必须逐类型**:fp16 地板(2e-4)比 bf16(1.6e-3)紧 8×,统一常量会让 fp16 辅助门形同虚设。

## 证伪电池 (numpy 镜像全过;命脉是反例1 cos 盲区)
- C1(命脉,cos盲区):大值区×1.005 系统偏置 → 相对L2≈5.3e-3>门3.9e-3 → **FAIL**(cos 仍 0.9999986 过,辅助门抓住)。
- C2 最大64点×1.1 → cos 破+相对L2 破 → FAIL。C4 半数元素×1.01 → 相对L2 破 → FAIL。
- P1 大K强抵消零误差核(peak1e5+近零残差,fp32换序 jitter~0.04)→ cos0.9999986/相对L2 1.7e-3 → **PASS**;同场景 OLD 固定门 dbr_max=156 **FAIL**(即所修病态)。P2 fp4+bias 近零擦门正确核 → PASS。

## C3 诚实坑:绝对偏置探针是量级相关的,不是万能反例
`bias+0.01` 绝对误差 → 相对L2 = 0.01/rms_g,**只在 rms_g < 0.01/门 ≈ 2.56 才 FAIL**。实际 fp4_bias 用例 rms_g≈113(整数 e2m1×scale matmul 主导,bias~N(0,0.5) 相对微不足道)→ +0.01 相对 8.8e-5 → **PASS 是正确的**(cos 也会 PASS,用户自选的相对L2 亦然)。要测"bias 真错"须注入**相对**误差(如 1%·rms),电池里 C3 改成 1%·rms(任意 shape 都 FAIL),并附一个 rms~1 shape 上 literal+0.01≈1% 的子例。启示:证伪反例的扰动量级必须相对信号非平凡,否则是无害误差不该 FAIL。

## 证据强度 (诚实)
- 本机无 torch/torch_npu/NPU → **真实 NPU 张量不可得**;上述全是 numpy+ml_dtypes 忠实镜像(官方 mx_quantize、真实 K、gen_data 分布)。脚本本体仅 `py_compile` 通过 + 逐行核对镜像等价,**未端到端跑**,真实用例判定为**推断**。
- 电池里 P1 有两版:`qbmm_cos_battery.py` 的 P1 是well-conditioned大K(near-zero 0%,弱);真正抵消正例是 `qbmm_p1_documented.py`(peak1e5+近零残差,OLD FAIL/NEW PASS,强)——引用时用后者。
- 复现脚本(未碰项目文件):`/d/tmp/qbmm_cos_threshold_derive.py`、`qbmm_rell2_margin.py`、`qbmm_cos_battery.py`、`qbmm_p1_documented.py`、`qbmm_old_vs_new.py`、`qbmm_print_samples.py`、`qbmm_c3_diagnose.py`。

See [[golden-cancellation-criterion]], [[golden-fp4-bias-nearzero]], [[golden-falsifiable-testing]], [[qbmm-batch-status]], [[device-print-verification]].
