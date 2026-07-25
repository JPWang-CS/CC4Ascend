# hiascend Ascend C API 官网索引

> 官网 SPA，按需 WebFetch 单页（正文可提取）。本文件 = **分类索引 + URL 规律 + 源策略**。
>
> 抓取日期：2026-07-25。URL 以官网为最终真值。

## 源策略

| 需求 | 首选源 |
|---|---|
| API 函数签名/模板/参数 | **ops-tensor 头文件**（本地，权威）：`ops-tensor/include/tensor_api/`、`ops-tensor/include/blaze/` |
| API 用法/约束/示例 | 官网（按需 WebFetch 单页） |
| API 存在性/分类 | 本 INDEX + 官网总览 |

> 头文件是 C++ 签名的权威源（比官网 doc 更准）。官网补充用法约束与示例。

## 官网总览入口

`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html`

## API 分类树（已捕获）

### 1. SIMD API
- **基础 API**（`atlasascendc_api_07_0014.html`）：硬件能力抽象，标注 ISASI 的不保证跨版本兼容 — [DataCopy 搬运家族清单](simd-basic-datacopy.md)
- **C API**：纯 C 接口，指针编程（官网外部 .md，按需 WebFetch）
- **高阶 API**（`atlasascendc_api_07_0489.html` 起；0489 实为 Tanh 叶子页，[无单页索引 / 按需抓取说明](simd-highlevel.md)）：数学库 / Matmul / Softmax 等，保证兼容性

### 2. SIMT API（入口 `atlasascendc_api_07_10825.html`）
- [9 子类清单](simt.md)（同步/原子/Warp/数学/地址谓词/地址转换/访存/协作组/调测）
- SIMD 与 SIMT 混合编程 API 列表（`atlasascendc_api_07_10847.html`）

### 3. AI CPU API（[接口清单](aicpu.md)｜入口 `atlasascendc_api_07_11096.html`）
- 非矩阵、分支密集型计算（printf / assert 调试）

### 4. Utils API（[10 类清单](utils.md)｜入口 `atlasascendc_api_07_11095.html`）
- 标准库 / 平台信息 / 原型注册 / Tiling 注册 / Tiling 模板 / Tiling 下沉 / RTC / Log / 调测

## URL 规律

- 统一格式：`atlasascendc_api_07_XXXX.html`（XXXX 数字编号）
- 基础 API：0014 起；高阶 API：0489 起；SIMT：10825-10847；总览：10894-11096
- 基础路径：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/`

## 按需查官网方法

1. 确定目标 API 属于哪个分类（SIMD 基础/C/高阶、SIMT、AI CPU、Utils）
2. WebFetch 对应分类入口页，拿该分类的 API 子链接
3. WebFetch 具体 API 页，提取用法/约束
4. 签名同时回 ops-tensor 头文件核对

## 已验证的本头文件入口（ops-tensor）

- `include/tensor_api/impl/tensor_api/arch/cube/gm_to_l1/copy_impl/instruction.h` — DataCopy 签名
- `include/tensor_api/impl/.../dn2nz.h` / `dn2zn.h` — ND→NZ 随路转换
- `include/blaze/epilogue/block/*.h` — HardEvent::MTE3_MTE2/MTE2_V/V_MTE3 SetFlag/WaitFlag 用法