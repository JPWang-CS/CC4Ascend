# GlobalTensor / LocalTensor / TPipe / TQue / TBuf 核心数据结构

## GlobalTensor（全局张量）

**位置**：Device 端 Global Memory (DDR/GM)

**作用**：管理 AI Core 外部全局内存中的数据，通常用于与 Host 侧数据交互。

```cpp
AscendC::GlobalTensor<half> xGm;
xGm.SetGlobalBuffer((__gm__ half*)x, TOTAL_LENGTH);
```

- 通过 `SetGlobalBuffer` 绑定 GM 地址和长度
- 配合 `DataCopy` 与 LocalTensor 之间搬移数据

## LocalTensor（局部张量）

**位置**：AI Core 内部 Local Memory（UB / L1 / L0 Buffer）

**逻辑位置 TPosition**：VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2

```cpp
// Pipe 范式：从 Queue 分配
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();

// 底层范式：直接构造
LocalTensor<DTYPE_X> xLocal(TPosition::VECIN, xAddr, tileLength);
```

- 从 Queue 通过 `AllocTensor` 分配 → 用完 `FreeTensor` 释放
- 是计算 API (Add/Mul/Sub) 的直接操作数

## TPipe（内存管理管道）

**定义**：`AscendC::TPipe`

**作用**：统一的 Device 端内存管理框架，屏蔽 DoubleBuffer 流水和同步细节。

```cpp
TPipe pipe;
pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(half));  // 双缓冲
pipe.InitBuffer(tempBuf, TILE_SIZE * sizeof(half));               // 单缓冲 TBuf
```

- 一个算子类定义一个 TPipe 实例
- 底层编程方式下 **不能使用** TPipe

## TQue（队列管理）

**定义**：`AscendC::TQue<TPosition, BUFFER_NUM>`

| 类型 | 说明 |
|------|------|
| `TQue<TPosition::VECIN, N>` | 输入队列，GM → UB |
| `TQue<TPosition::VECOUT, N>` | 输出队列，UB → GM |

经典三阶段：
```
CopyIn:  AllocTensor → DataCopy(GM→Local) → EnQue
Compute: DeQue → 计算 → EnQue
CopyOut: DeQue → DataCopy(Local→GM) → FreeTensor
```

内部自动插入 SetFlag/WaitFlag 同步事件。

## TBuf（临时缓冲区）

**位置**：UB / L1 Buffer 中的临时存储

```cpp
TBuf<TPosition::VECCALC> tempBuf;
pipe.InitBuffer(tempBuf, TILE_SIZE * sizeof(half));
LocalTensor<half> tempLocal = tempBuf.Get<half>(TILE_SIZE);
```

- 只分配一块（不像 TQue 可多块）
- **只能参与计算，不能 EnQue/DeQue**
- 不需要 FreeTensor
