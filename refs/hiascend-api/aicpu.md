# AI CPU API 清单（官网捕获）

> 源：`atlasascendc_api_07_11096.html`（父主题 `atlasascendc_api_07_00165.html`，2026-07-25 WebFetch）。
> AI CPU API 承担非矩阵、分支密集型计算，是 SIMD/SIMT 的补充。当前官网公开调试接口较薄。

## 接口

| API | 用途 | 页 |
|---|---|---|
| printf | AI CPU 算子 Kernel 调试格式化输出，默认解析后打印到屏幕 | `atlasascendc_api_07_00166.html` |
| assert | AI CPU 算子 Kernel 调试 assert 断言 | `atlasascendc_api_07_00167.html` |

## 备注

- 算子走 AI CPU 的判定 / `--genop_aicpu` 流程见 `ascendc-development`（AI Core 开发指南的 AI CPU 小节）
- AI CPU 算子 Kernel 内的数值定位同 SIMD 路线（PRINTF/DumpTensor）见 `ascendc-debug`
