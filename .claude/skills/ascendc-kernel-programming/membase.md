# Membase 编程范式（A2A3 经典）

trace 自 `ops-transformer_AI/examples/add_example/op_kernel/add_example.h`。这是 AscendC 最基础的 kernel 写法，A2A3 主力范式。

## 核心三件套

- `TPipe pipe` — 流水管道，管理 UB 分配
- `TQue<Position, BUFFER_NUM>` — 队列，管 LocalTensor 的 Alloc/EnQue/DeQue/Free + 流水同步
- `LocalTensor<T>` — UB 上的数据句柄（逻辑位置），不直接持内存

三阶段流水靠 TQue 的 EnQue/DeQue 隐式触发 MTE2↔V↔MTE3 事件同步。

## 完整骨架（add_example 实证）

### 数据成员（:43-46）
```cpp
template <typename T>
class AddExample {
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;   // 输入队列（double buffer）
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ; // 输出队列
    GlobalTensor<T> inputGmX, inputGmY, outputGmZ;      // GM 句柄
};
```

### Init（:58-70）
```cpp
void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData *tilingData) {
    // 1. GM 句柄
    inputGmX.SetGlobalBuffer(x);  // ... y, z 同理
    // 2. pipe 分配 UB buffer 给各队列
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, tileLength * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, tileLength * sizeof(T));
    // 3. 读 tiling 参数
    totalLength = tilingData->totalLength;
    // ...
}
```

### Process（三阶段循环）
```cpp
void Process() {
    int32_t loopCount = tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);      // GM → UB
        Compute(i);     // UB 上计算
        CopyOut(i);     // UB → GM
    }
}
```

### CopyIn（:74-81）
```cpp
void CopyIn(int32_t progress) {
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();  // 申请 UB
    LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    DataCopy(xLocal, inputGmX[progress * tileLength], tileLength);  // GM→UB
    DataCopy(yLocal, inputGmY[progress * tileLength], tileLength);
    inputQueueX.EnQue(xLocal);   // 入队 → 隐式 MTE2→V 同步
    inputQueueY.EnQue(yLocal);
}
```

### Compute（DeQue → 算 → EnQue）
```cpp
void Compute(int32_t dataLength) {
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();   // 出队（等 MTE2 完成）
    LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    Add(zLocal, xLocal, yLocal, dataLength);          // Vector 计算
    inputQueueX.FreeTensor(xLocal);                    // 释放
    inputQueueY.FreeTensor(yLocal);
    outputQueueZ.EnQue(zLocal);                        // 入队 → 隐式 V→MTE3 同步
}
```

### CopyOut（:85-）
```cpp
void CopyOut(int32_t progress) {
    LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();   // 出队（等 V 完成）
    DataCopy(outputGmZ[progress * tileLength], zLocal, tileLength);  // UB→GM
    outputQueueZ.FreeTensor(zLocal);
}
```

## kernel 入口（add_example.cpp）
```cpp
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    AddExample<float> op;
    op.Init(x, y, z, tiling);
    op.Process();
}
```

## 为什么这样组织
- **三阶段**：MTE2（搬入）/ V（计算）/ MTE3（搬出）是三个独立硬件单元，三阶段让它们流水重叠
- **TQue 双缓冲**（BUFFER_NUM=2）：第 n 轮搬入时，第 n-1 轮在算、第 n-2 轮在搬出，三者并行
- **AllocTensor/EnQue/DeQue/FreeTensor 四件套**：EnQue/DeQue 触发隐式事件同步，无需手写 SetFlag/WaitFlag

## 何时用 Membase
- A2A3 算子（主力范式）
- A5 上简单的连续数据处理（非离散、非需寄存器级控制）
- elementwise / reduce / 简单向量

## 何时不用
- 离散访存 → SIMT
- 需精细寄存器控制 / 复杂掩码 → Regbase
- 矩阵乘 → 高阶 Matmul<> 或 Blaze