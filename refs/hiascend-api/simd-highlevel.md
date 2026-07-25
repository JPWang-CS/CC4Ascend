# SIMD 高阶 API

官网高阶 API（`atlasascendc_api_07_0489.html` 起）无分类索引页，是叶子页平铺（0489 = Tanh 叶子）。逐个 API 名按需 WebFetch。

## 本地已覆盖（ascendc-api skill，签名以 ops-tensor 头文件为准）

| 高阶 API | 本地明细 |
|---|---|
| Matmul | `MatMul高阶API.md` — MatmulConfig 模板 / 双缓冲 / A5 V3 |
| Softmax | `SoftMax高阶API.md` — Online / Masked / FlashV2 |
| Activation | `Activation激活函数.md` — GELU / SwiGLU |
| Cast | `Cast与类型转换.md` — RoundMode / SatMode |
| Fixpipe | `Fixpipe.md` — L0C→L1/UB/GM 随路量化 |
| DataCopy | `DataCopy与DataCopyPad.md` + [simd-basic-datacopy.md](simd-basic-datacopy.md) |

数学库 / Reduce / Compare 等未单独列，按需取。

## 按需取法

1. WebFetch 总览 `atlasascendc_api_07_0003.html` 找 API 所属分类
2. 或按 URL 规律 `atlasascendc_api_07_XXXX.html` 试取（0489 起，编号非连续）
3. 签名回 ops-tensor 头文件核对（`ops-tensor/include/tensor_api/`、`include/blaze/`）
