# A5 SIMT 优化

## SIMT vs SIMD

| 特性 | SIMT (A5 新增) | SIMD (Regbase) |
|------|---------------|----------------|
| 并行粒度 | 线程级（每个线程独立指令流） | 向量级（同指令多数据） |
| Kernel 限定 | `__simt_vf__` | `__simd_vf__` |
| 分支处理 | 各线程可独立分支 | 所有 lane 必须同路径 |
| 适用场景 | 不规则计算、动态 mask、稀疏 | 规则密集计算 |

## 性能优势场景

1. **不规则 Mask**：Sparse attention 中 causal/band/prefix mask 各线程可以独立判断
2. **动态 Group 边界**：GMM 中不同 group 的 M 大小不同
3. **条件计算**：量化中 per-element 的动态 scale 选择

## 与 Regbase 的配合

A5 中 SIMT 和 Regbase 通常配合使用：
- Regbase 处理规则密集部分（MatMul、Reduce）
- SIMT 处理不规则条件部分（Mask、Scatter/Gather）

## 代码仓中的 SIMT 使用

A5 block_sparse_attention 中的条件分支利用了 SIMT 的线程独立分支能力，避免了 Membase 中全 lane 都执行相同分支路径的浪费。

## 注意事项

- SIMT 模式下每个线程的寄存器文件独立，总寄存器压力更大
- 建议只在确实需要线程级分支差异时才使用 SIMT
- 规则密集计算仍用 Regbase/SIMD 以获得更高吞吐
