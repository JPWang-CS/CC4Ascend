# Cube-Vector 融合通路（A5 专属）

## 概述

Ascend 950 引入 **UB↔L1** 和 **L0C→UB** 直连通路，让 Cube 和 Vector 计算在片上直接交换数据，无需经 GM 中转。

## 新增通路

### 矩阵搬入：UB → L1 (UB2L1)

向量计算结果直接搬入 L1，供 Cube 使用：

```cpp
template <typename T>
__aicore__ inline void DataCopy(
    const LocalTensor<T>& dst,   // L1 目标
    const LocalTensor<T>& src,   // UB 源
    const Nd2NzParams& intriParams);
```

### 矩阵搬出：L0C → UB (L0C2UB)

矩阵计算结果直接搬入 UB，供 Vector 使用：

```cpp
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(
    const LocalTensor<T>& dst,    // UB 目标
    const LocalTensor<U>& src,    // L0C 源
    const FixpipeParamsC310<config.format>& intriParams);

template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsC310 {
    uint8_t dualDstCtl = 0;
};
```

## 应用场景

### 切 K 累加

传统 A2A3：L0C → GM → UB（每次切 K 都要 GM 往返）
A5：L0C → UB（直达，消除 GM 带宽压力）

### 后处理融合

```
Cube MatMul → L0C2UB → Vector Act/Quant/Norm → UB2L1 → 下一轮 Cube
```

### 迁移建议

1. 中间结果归并、激活/量化前处理 → 放 UB 侧完成
2. 显式梳理 MTE1/MTE2/MTE3 与计算单元的事件同步顺序
3. 确保新增通路不引入数据可见性或同步时序问题

## 来源
- `ops-transformer_AI/docs/zh/develop/cross_platform_migration_guide.md` §Cube-Vector融合
