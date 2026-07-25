# SIMT 编程范式（A5）

A5 新增 SIMT（单指令多线程）单元，处理**离散访存**（gather/scatter/索引重排）。每个线程独立处理元素，适合地址不连续场景。

## 与 SIMD（Membase/Regbase）的区别

| | SIMD | SIMT |
|---|---|---|
| 并行粒度 | 向量级（一条指令处理一个向量） | 线程级（多线程各处理元素） |
| 访存 | DataCopy 连续搬进 UB | 线程直接 `__gm__` 访问 GM |
| 缓冲 | 显式 TQue/UB 管理 | 无显式 buffer，硬件管 |
| 同步 | 显式事件 SetFlag/WaitFlag | 线程间隐式同步 |
| 适用 | 连续大块地址 | 离散、不连续地址 |

## 代码骨架（trace 自 gather_v2 范式，ascendc-development §SIMT）

```cpp
__simt_vf__ LAUNCH_BOUND(2048) void GatherSimt(
    __gm__ float* x, __gm__ float* y, __gm__ int32_t* indices,
    int64_t currentCoreElements, ...) {
    // 每个线程跳跃式并行处理
    for (INDEX_SIZE_T idx = Simt::GetThreadIdx();
         idx < currentCoreElements;
         idx += Simt::GetThreadNum()) {
        INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);  // 线程内算索引
        INDICES_T val = indices[gatherI];                           // 直接访问 GM
        y[yIndex] = outOfBound ? 0 : x[xIndex];                     // 直接写 GM
    }
}
```

## 关键要素
- `__simt_vf__` — SIMT 函数标记
- `LAUNCH_BOUND(2048)` — 线程数上限
- `Simt::GetThreadIdx()` — 当前线程号
- `Simt::GetThreadNum()` — 总线程数
- 线程循环步长 = GetThreadNum()，各线程交错覆盖全部元素

## 为什么 SIMT 适合离散访存
SIMD（DataCopy）要求地址连续，离散地址只能逐元素标量搬，慢。SIMT 每个线程独立算索引、独立访问 GM，多线程并行掩盖随机访存延迟。尾轴 ≤2048 时离散小块多，SIMT 效率高于 SIMD。

## 选路规则（gather_v2 实证）
- 尾轴 ≤ 2048 → SIMT（离散小块多）
- 尾轴 > 2048 → SIMD（大块连续，向量化高效）

## 何时用 SIMT
- gather / scatter / embedding（按索引离散搬运）
- 索引重排、稀疏更新
- 地址不连续、访存跨度大、分支不一致

## 编写注意
- 线程任务切分匹配数据稀疏性，避免负载不均
- 边界处理与主路径解耦，热点循环少分支
- 减少高频随机访存，上游尽量索引规整/分桶