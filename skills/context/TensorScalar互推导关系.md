# Tensor-Scalar 互推导关系

## 概述

当 API（如 `aclnnAdds`, `aclnnMuls`）输入的 **Tensor 类型**与 **Scalar 类型**不一致时，API 推导后转换。

## 关键差异（vs Tensor-Tensor 推导）

| 对比维度 | Tensor-Tensor | Tensor-Scalar |
|----------|:---:|:---:|
| 推导方向 | 向更宽类型提升 | **Scalar 向 Tensor 类型靠拢** |
| 示例 | f16 + f32 → f32 | f16 Tensor + f32 Scalar → f16 |
| 示例 | bool + f32 → f32 | bool Tensor + f32 Scalar → f32 |

## 推导规则表

首行 = Tensor 类型，首列 = Scalar 类型，交点为推导结果。

| 数据类型 | f32 | f16 | f64 | bf16 | s8 | u8 | s16 | u16 | s32 | u32 | s64 | u64 | bool |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **f32** | f32 | f16 | f64 | bf16 | f32 | f32 | f32 | × | f32 | × | f32 | × | f32 |
| **f16** | f32 | f16 | f64 | bf16 | f32 | f32 | f32 | × | f32 | × | f32 | × | f32 |
| **bf16** | f32 | f16 | f64 | bf16 | f32 | f32 | f32 | × | f32 | × | f32 | × | f32 |
| **s8** | f32 | f16 | f64 | bf16 | s8 | u8 | s16 | u16 | s32 | u32 | s64 | u64 | s8 |
| **s16** | f32 | f16 | f64 | bf16 | s8 | u8 | s16 | u16 | s32 | u32 | s64 | u64 | s16 |
| **s32** | f32 | f16 | f64 | bf16 | s8 | u8 | s16 | u16 | s32 | u32 | s64 | u64 | s32 |
| **bool** | f32 | f16 | f64 | bf16 | s8 | u8 | s16 | u16 | s32 | u32 | s64 | u64 | bool |

## 关键记忆

- 浮点 Scalar 配浮点 Tensor → 以 **Tensor 类型**为准（如 f32 scalar + f16 tensor → f16）
- 整数 Scalar 配整数 Tensor → 按整数提升规则
- bool Scalar 配任何 Tensor → 以 **Tensor 类型**为准
- **不支持 uint16/uint32 混合**：u16 Scalar 不能与 float32 Tensor 组合（×）

## 来源
- `ops-transformer_AI/docs/zh/context/TensorScalar互推导关系.md`
