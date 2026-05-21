# Fixpipe（矩阵乘结果搬运与随路处理）

## 功能说明

Fixpipe 负责将矩阵乘结果从 **L0C Buffer 搬出到 L1 Buffer 或 GM**，搬运过程中可完成：
- 数据类型转换（float → half / int8_t）
- 随路量化（Scalar 量化 / Vector 量化）
- ReLU 激活
- 通道拆分

硬件对应 FIXP 流水线：`FIXPIPE L0C → OUT/L1`

## 函数原型

```cpp
template <typename outputType, typename l0cType, CO2Layout layout>
__aicore__ inline void Fixpipe(
    const LocalTensor<outputType>& dstL1,
    const LocalTensor<l0cType>& srcL0C,
    const FixpipeParams& params);
```

## A2A3 (FixpipeParamsV220)

| 参数 | 说明 |
|------|------|
| `nSize` | 输出矩阵 N 方向大小 |
| `mSize` | 输出矩阵 M 方向大小 |
| `srcStride` | 源 Nz 矩阵相邻 Z 偏移 |
| `dstStride` | 目的矩阵相邻 Z 偏移 |
| `quantPre` | 量化模式：`QF322B8_PRE`(Scalar) / `VQF322B8_PRE`(Vector) |
| `deqScalar` | Scalar 量化参数 |
| `reluEn` | ReLU 使能 |
| `unitFlag` | Mmad 与 Fixpipe 细粒度并行控制 |
| `isChannelSplit` | 通道拆分 |

## A5 (FixpipeParamsC310)

A5 使用 `FixpipeParamsC310<CO2Layout::ROW_MAJOR>`，支持：
- `TransformParams`：根据 CO2Layout 自动选择参数类型
- `dualDstCtrl`：双目标模式控制
- `subBlockId`：单目标模式下目标 UB 编号

## 数据通路

| 方向 | A2A3 | A5 |
|------|:---:|:---:|
| L0C → L1 | ✅ | ✅ |
| L0C → GM | ✅ | ❌ (被删除) |
| L0C → UB | ❌ | ✅ (新增 L0C2UB 直连) |

**A5 关键变化**：L0C→GM 被移除，改为 L0C→UB→GM 两跳通路。UB2L1 也新增了直连通路用于 Cube↔Vector 融合。

## 使用示例

### 基础搬出（float→half，无量化）

```cpp
FixpipeParamsV220 params;
params.nSize = 128; params.mSize = 128;
params.quantPre = QuantMode_t::NONE;  // 仅cast
params.reluEn = false;

Fixpipe<half, float, CFG_Nz>(l1Tensor, l0cTensor, params);
```

### Scalar 量化 + ReLU（float→int8_t）

```cpp
params.quantPre = QuantMode_t::QF322B8_PRE;
params.deqScalar = scalarQuantParam;
params.reluEn = true;
Fixpipe<int8_t, float, CFG_Nz>(l1Tensor, l0cTensor, params);
```
