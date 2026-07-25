# SIMD 基础 API — DataCopy 搬运家族（官网捕获）

> 源：`atlasascendc_api_07_0014.html`（2026-07-25 WebFetch）。签名以 ops-tensor 头文件为准（`ops-tensor/include/tensor_api/`）。

## 8 个搬运子类

| 子类 | 用途 |
|---|---|
| 基础数据搬运 | 连续/非连续数据搬运，保持原格式 |
| 增强数据搬运 | 在基础上增加 CO1→CO2 通路随路计算 |
| 切片数据搬运 | 多维 Tensor 子集切片搬运 |
| 随路转换 ND2NZ 搬运 | 搬运同时 ND→NZ |
| 随路转换 NZ2ND 搬运 | 搬运同时 NZ→ND |
| 随路转换 DN2NZ 搬运 | 搬运同时 DN→NZ |
| 随路量化激活搬运 | Local→Global 随路量化 + ReLU + NZ→ND |
| 多维数据搬运 | 比基础更自由的搬入维度 / Stride 配置 |
