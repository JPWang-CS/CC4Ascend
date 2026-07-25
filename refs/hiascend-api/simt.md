# SIMT API 分类清单（官网捕获）

> 源：`atlasascendc_api_07_10825.html`（2026-07-25 WebFetch）。URL 以官网为最终真值。
> 头文件依赖：`simt_api/asc_simt.h`（通用入口）/ `asc_fp16.h`（half/half2）/ `asc_bf16.h`（bfloat16）/ `asc_fp8.h`（fp8）。

## 9 个子类

| 子类 | 用途 | 入口页 |
|---|---|---|
| 同步与内存栅栏 | 内存管理与同步接口，解决不同核内线程间数据竞争与同步 | `atlasascendc_api_07_10314.html` |
| 原子操作 | 对 Unified Buffer / Global Memory 数据与指定数据执行原子操作 | `atlasascendc_api_07_10374.html` |
| Warp 函数 | 单个 Warp 内 32 线程的数据处理 | `atlasascendc_api_07_10386.html` |
| 数学函数 | 数学运算及不同精度/数据类型转换函数集合 | `atlasascendc_api_07_10309.html` |
| 地址空间谓词函数 | 判断输入指针是否为指定空间地址 | `atlasascendc_api_07_10812.html` |
| 地址空间转换函数 | 地址值↔指针 / 指针↔内存空间地址值转换 | `atlasascendc_api_07_10816.html` |
| 访存函数 | 数据加载与缓存相关接口 | `atlasascendc_api_07_10555.html` |
| 协作组 | 标准安全机制，实现高效线程并行协作 | `atlasascendc_api_07_11067.html` |
| 调测接口 | SIMT VF 调试场景相关接口 | `atlasascendc_api_07_10425.html` |

SIMT 与 SIMD 混合编程：`atlasascendc_api_07_10847.html`。
