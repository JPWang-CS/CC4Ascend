# Blaze 编程范式（ops-tensor）

Blaze（Basic Linear Algebra Optimized Engine）是 ops-tensor 的 header-only Kernel 端 matmul 加速引擎，CUTLASS 式分层模板。trace 自 `ops-tensor/include/blaze/`。

## 定位（trace 自 blaze/README.md）

- **聚焦 Kernel 端**：只管 matmul 类算子的 Kernel 端计算组件（搬运/MMAD/调度），不管 aclnn/Host
- **服务于**：Matmul、GroupedMatmul、MC2 等矩阵乘类算子
- **依赖**：`include/tensor_api/`（Layout/Shape/Coord）+ AscendC Kernel 接口

## 三层抽象（trace 自 README + 目录）

```
kernel/   Blaze::Gemm::Kernel    完整内核入口（组合 Block + Epilogue + Scheduler → 可启动 Kernel）
block/    Blaze::Gemm::Block     Block 级 Mmad 抽象 + 调度器（按 Policy 选实现）
epilogue/ Blaze::Gemm::Block     后处理策略（Bias/激活/反量化 可扩展）
tile/     Blaze::Gemm::Tile      Tile 级原语（K 方向补零等细粒度）
policy/   Blaze::Gemm::          DispatchPolicy（全载/量化/scale 等编译期派发）
utils/    Blaze::Gemm::          CeilDiv/Layout 推导/量化常量
```

自上而下：Kernel 组合 Block + Epilogue + Scheduler；Block 做 Mmad；Tile 做最细粒度搬运/计算。

## DispatchPolicy（策略驱动，trace 自 README）

```cpp
// Policy 把算法变体作为类型参数，编译期选最优实现
struct DispatchPolicy {
    bool fullLoad;        // 全载 vs 非全载
    QuantMode quant;      // 量化模式
    bool withScale;       // 是否带 scale
    // ...
};
// 不同 Policy → 不同 Block 特化，编译期生成最优代码路径
```

## 类型安全的组合（trace 自 README）

A/B/C/Bias 的 dtype + Layout（NDExt/DNExt/NZ/ZN）作为类型参数透传，编译期生成对应代码路径。

## 充分利用 Cube 架构（trace 自 README）

直接对接 L1/L0A/L0B/L0C 存储层级 + MMAD 指令，结合：
- double-buffer（双缓冲）
- ND2NZ 自动补零

## 与各仓的关系

| 仓 | 用 Blaze 吗 |
|---|---|
| ops-transformer（attention/gmm/mc2） | gmm 用 gmm_infra（自家 CUTLASS 式，类似 Blaze）；mc2 用 Blaze |
| ops-nn（matmul 类） | 用 Blaze（weight_quant_batch_matmul 等） |
| ops-tensor | Blaze 本体在此 |

> 注：ops-transformer 的 gmm 有自己的 `gmm_infra/`（类似 Blaze 的模板库，见 kernel-optimization gmm base），与 ops-tensor Blaze 是两套独立但理念相似的模板库。

## 何时用 Blaze
- 复杂 matmul 类算子（量化、grouped、通算融合）
- 需要 block/tile/epilogue 灵活组合
- A5 上想用 CUTLASS 式声明式 Kernel

## 编写方式
Blaze 是 header-only，在算子 Kernel 里 `#include` 对应 block/epilogue 模板，组合 DispatchPolicy 实例化。具体 API 见 `ops-tensor/include/blaze/{gemm,epilogue}/` 头文件。