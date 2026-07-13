# MatMul 高阶 API

## 四步使用法

```cpp
#include "kernel_operator.h"

// Step 1: 注册与初始化
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);
matmulObj.Init(&tiling);

// Step 2: 设置矩阵
matmulObj.SetTensorA(gm_a);     // 左矩阵 [M, K]
matmulObj.SetTensorB(gm_b);     // 右矩阵 [K, N]
matmulObj.SetBias(gm_bias);     // 可选 Bias

// Step 3A: 直出到 GM（简单场景）
matmulObj.IterateAll(gm_c);

// Step 3B: 分步 + UB 后处理（融合场景）
while (matmulObj.Iterate()) {
    matmulObj.GetTensorC(ub_c);  // 获取中间结果到 UB
    // 在 UB 中做激活、量化等后处理...
}
matmulObj.End();
```

## MatmulConfig 模板选择

| 模板 | 适用场景 |
|------|----------|
| **Norm**（默认） | L1 缓存多个 base block，MTE2 多次搬运，通用场景 |
| **MDL** | L1 一次性"大包"搬运，大 shape 场景 |
| **SpecialMDL** | MDL 扩展，K 方向未满载时 stepN=2 |
| **IBShare** | 多核共享 L1 上相同 A/B 矩阵 |
| **BasicBlock** | 固定 baseM/baseN/baseK，无尾块、shape 规整场景 |

## 关键配置参数

```cpp
MatmulConfig cfg;
cfg.doTiling           = true;   // 启用自动 Tiling
cfg.doDoubleBuffer     = true;   // ★ 双缓冲，强烈建议开启
cfg.doMultiDataLoad    = true;   // MDL 模板
cfg.doIBShareNorm      = true;   // IBShare 模板
cfg.enableL1CacheUB    = true;   // L1 缓存 UB 计算块
cfg.enableStaticPadZeros = true; // 自动 padding
cfg.ScheduleType       = ScheduleType::OUTER_PRODUCT;
cfg.IterateOrder       = IterateOrder::ORDER_M;
```

## 注意事项

| 事项 | 说明 |
|------|------|
| **数据通路** | Cube 计算必须 GM→L1→L0A/L0B→Cube→L0C，不能直接 UB→Cube |
| **数据格式** | Cube 使用 NZ（分形格式），高阶 API 自动处理 ND→NZ 转换 |
| **累加器精度** | FP16 输入 → 中间累加器用 FP32，否则大矩阵精度丢失 |
| **双缓冲** | 不开启 doDoubleBuffer，Cube 和 MTE 互相等待，性能差 2-3× |
| **Tiling** | 切太大→UB/L1 溢出；切太小→搬运开销吃掉性能。推荐 MatmulTiling 库自动计算 |
| **尾块** | 矩阵维度不是 16 的倍数时，需配置 SetTail 或 enableStaticPadZeros |
| **VecND2NZ 对齐** | GM 地址和 size 必须 512B 对齐 |
| **L2 Cache 切分** | 数据量超 L2 Cache（192MB）时应启用 |
| **ScheduleType** | OUTER_PRODUCT 仅在 K 方向满载 + 输出到 GM 时可用 |

## A5 专用：MatMul V3

A5 新增 `Mc2MatMulV3TilingData`（与旧版 matmulTiling 不同）：
- baseM/baseN/baseK 从 UB/L0C 容量反推
- 支持 `CUSTOM_CFG_MDL` 配置（enableGetTensorC=true）
- 配合 cgmct 框架实现 BlockMmad/BlockEpilogue 模块化组合
