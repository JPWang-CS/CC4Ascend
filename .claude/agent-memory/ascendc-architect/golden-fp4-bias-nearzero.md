---
name: golden-fp4-bias-nearzero
description: 1b_fp4_bias 偶发 err_small=1 FAIL 的根因分析——fp4 matmul 是 fp32 精确(dyadic),近零点靠 bias 抵消;正确核也会因 L0C bias-first 累加序偶发擦门(~1.7%),而 bias 降精 bug 则远超门;区分靠失败点 |err| 量级
metadata:
  type: reference
---

# 1b_fp4_bias 偶发 err_small=1 —— (A)判据边缘 vs (C)算子bias降精 的判别

> **状态(2026-07-07):本篇分析的 `err_small` 门属旧官方判据,已从 `golden_mx_matmul_batch.py` 删除**(换成 cos为主+相对L2,见 [[golden-cos-rell2-criterion]])。cos 判据下,fp4+bias 近零擦门(L0C bias-first 累加序的 ~1e-5 边缘)不再触发 FAIL —— 已列为证伪电池正例2 且 PASS。本篇保留价值:**(A)判据边缘 vs (C)算子bias真降精 的机理判别**(靠失败点 |npu-golden| 量级:~1e-5=无害边缘,~1e-4..1e-2=算子降精须停手查 Fixpipe/epilogue),此判别在任何判据下都适用。

`golden_mx_matmul_batch.py` 的 `1b_fp4_bias`(M64 N128 K256, fp4_e2m1, out=bf16, batch=[2], has_bias, **无 seed → 随机**)偶发 `res=FAIL`,唯一原因 `err_small 1>标杆0×2`。诊断产出(numpy+ml_dtypes 忠实复刻,本机无 torch/NPU):

## 关键事实:fp4 matmul 在 fp32 里是精确的(与大K抵消案本质不同)
- fp4 e2m1 值 × e8m0(2的幂)scale = **dyadic**;K=256 点积部分和 ≪ 2^24 粒度 → **fp32 累加与求和序无关、且 == fp64 truth**(1500 seed / 112 近零点:非精确 0%、fwd-vs-rev 分歧恒 0)。
- 故近零输出**不是 fp32 换序噪声**造成(那是 mxfp8/K19085 案,见 [[golden-cancellation-criterion]]);这里近零点是 **matmul 出整数值(-1/-0.5/0/0.5/0.75)、bias≈其相反数抵消**(L1≈2600 但和精确)。
- `atol_floor=sqrt(K)*rms*2^-24≈2.3e-4 < small=1e-3` → **pathological=False,判据走原官方口径**(病态分支未触发,原小值门=1e-3、atol_abs=1e-5、err_small 标杆=0)。

## 头条 max_diff=1.9932 是无害的:出在大值点,与失败点无关
- max_diff 恒落在 |golden|≈514~1227 的大值(300 seed:0% 落近零点);1.99≈bf16 对 ~512 量级的半 ULP。**max_diff/cos 是 report-only,不参与 verdict**(golden 脚本 L1170)。cos=0.999999 = bf16 舍入理论值。

## 判据是健全的:正确核 0% 被误杀(排除"纯判据过严无害")
- 理想核(exact fp32 matmul + fp32 bias, matmul-then-+bias):1500 seed **err_small 恒 0**,从不 FAIL。→ 小值门本身没系统性误杀。

## 但两个"擦门"来源都存在,靠失败点 |npu-golden| 量级区分:
- **(A) 正确 MX cube 的 L0C bias-first 累加序**:算子 `BiasType=float`(硬编码 @ `ops-nn/matmul/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mx_{tensor_api_blaze,without_batch_tensor_api_blaze,basic_api_cmct}.h`),bias 进 L0C bias 表**先置 bias 再累加各积**,与 golden 的 matmul-then-+bias **求和序不同**。忠实模型(fp32 bias-first + bf16 out):**~1.7% 随机 run 出 err_small=1**,失败点 **|err|≈1.2e-5~2.0e-5(刚过 1e-5 门一点点)**。这属判据/用例设计边缘(门=0 对单个擦线点零容忍),非算子错。
- **(C) bias 被降到 bf16 再加**(假想 bug):失败点 **|err|≈1e-4~1e-2**(bias 的 bf16 量子),~17%/run FAIL,症状与用户行几乎逐字吻合(err_small=1、max_diff≈2.0@大值、cos=0.999999、fail_batches 含 1)。但**算子 BiasType=float 已排除此路径**在标准 MX kernel 成立——除非某分支/Fixpipe 真降精。

## 判别与处置(需要用户提供真实失败点 |npu-golden|,当前脚本没打印该值)
- 失败点 |err| ~1e-5(<5e-5)→ **(A)**:L0C bias-first 无害边缘。可考虑放宽,但方向应是**给 has_bias 用例一个与 bias 量级挂钩的绝对逃生口**(如 atol_abs ≥ few·2^-24·|bias|·ULP,或对 bias-first 序引入的 O(u·|bias|) 容差),**不要无差别抬 small 门**;并做证伪:注入真实 bias 误差须仍 FAIL、正确核 PASS。
- 失败点 |err| ~1e-4..1e-2 → **(C)**:算子在近零点 bias 真降精,**停手不改判据**,查 Fixpipe/epilogue 是否把 fp32 bias 或 L0C 结果降到 out_dtype 再加。
- 诚实:本机拿不到真实 NPU 张量,(A)/(C) 的最终裁定必须靠那一个失败点的真实 golden/npu 值(或多 run 统计失败点 |err| 分布)。复现脚本 `/d/tmp/qbmm_fp4bias_probe.py`、`/d/tmp/qbmm_fp4bias_mechanisms.py`(未碰项目文件)。

See [[golden-cancellation-criterion]], [[golden-falsifiable-testing]], [[fp4-e2m1-packing]], [[mx-quant-scale-semantics]], [[qbmm-batch-status]].
