# SIMT 编程范式（A5 专属）

## 概述

Ascend 950 新增 **SIMT** 单元，处理非规整离散访问方面比 SIMD 更优。适合地址不连续、访存跨度变化大、分支路径不一致的场景（scatter/gather, 索引重排, 稀疏更新）。

## SIMD vs SIMT 核心差异

### 编程模型

**SIMD**（传统向量化）：
```cpp
// 队列机制管理数据缓冲
TQueBind<VECIN, VECOUT, BUFFER_NUM> inQueue_;
TBuf<VECCALC> indexBuf_;

// 逐行处理，显式搬运和同步
for (int64_t j = 0; j < rows; j++) {
    INDICES_T index = GetIndex(yIdx, indiceEndIdx);
    DataCopyPad(xLocal[j * colsAlign], xGm[offset], ...);
}
inQueue_.EnQue<int8_t>(xLocal);  // 入队
```

**SIMT**（线程级并行）：
```cpp
__simt_vf__ LAUNCH_BOUND(2048) void GatherSimt(...) {
    for (INDEX_SIZE_T index = Simt::GetThreadIdx();
         index < currentCoreElements;
         index += Simt::GetThreadNum()) {

        INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);
        INDICES_T values = indices[gatherI];
        y[yIndex] = idxOutOfBound ? 0 : x[xIndex];  // 直接写 GM
    }
}
```

### 全面对比

| 特性 | SIMD | SIMT |
|------|------|------|
| 数据访问 | `DataCopyPad` 显式搬运到 UB | 线程直接通过 `__gm__` 访问 GM |
| Buffer 管理 | 需 AllocTensor/EnQue/DeQue/FreeTensor | 无需显式 buffer，硬件自动管理 |
| 同步机制 | 显式事件同步 (`HardEvent::MTE2_V`) | 线程间隐式同步 |
| 并行粒度 | 向量级 (Vec) | 线程级 (Thread) |
| 适用场景 | 连续访问大块地址 | 离散访存，线程并行处理 |

## 适用场景决策：以 `gather_v2` 为例

- **尾轴 ≤ 2048** → SIMT（离散小块地址多，线程并行高效）
- **尾轴 > 2048** → SIMD（大块连续数据，向量化高效）

## 迁移注意事项

1. **线程任务切分匹配数据稀疏性** — 避免线程负载极不均衡
2. **减少高频随机访存** — 上游完成索引规整与分桶
3. **边界处理与主路径解耦** — 避免热点循环中引入过多分支
4. **对比 SIMD +SIMT 混合 vs 纯 SIMD** — 按数据分布选择最优策略

## 来源
- `ops-transformer_AI/docs/zh/develop/cross_platform_migration_guide.md` §SIMT
