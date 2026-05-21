# A2A3 双缓冲

## BigDoubleBuffer 控制

TilingKey 中 BigDoubleBuffer (2 bits)：
- 0 = NONE — 单缓冲
- 1 = BIG_BUFFER — 单个大缓冲
- 2 = BIG_DOUBLE_BUFFER — 双倍大缓冲

通过 `ASCENDC_TPL_UINT_DECL(BigDoubleBuffer, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2)` 编译期选择。

较大的 S1/S2 尺寸会通过 `GM_DOUBLE_BUFFER` 常量（定义为 2）影响 workspace 布局中的 `bmm1AndVec1Ratio`。

## TQue 双缓冲

```cpp
// 声明双缓冲队列
TQue<TPosition::VECIN, 2> inQueueX;   // BUFFER_NUM=2
TQue<TPosition::VECOUT, 2> outQueueZ;

// InitBuffer 分配两块内存
pipe.InitBuffer(inQueueX, 2, TILE_SIZE * sizeof(half));  // ping-pong
```

TQue 内部自动管理 ping-pong 切换和同步：
- `AllocTensor` 自动选择空闲 buffer
- `EnQue` 将数据放入队列并自动 SetFlag
- `DeQue` 等待数据就绪（内部 WaitFlag）并取出

## UB 缓冲分配示例（FlashAttention S1S2）

```
maskTBufPing     9KB  (atten mask)
maskTBufPong    16KB  (dropout mask)
pseTBuf         16KB
stage1PingBuf   32KB  (t.a)
stage2TBuf      32KB  (t.c)
commonTBuf      32KB  (t.b)
softmaxSumBuf[2] 2KB×2 (ping-pong)
softmaxExpBuf[2] 2KB×2 (ping-pong)
softmaxMaxBuf    single
softmaxTempBuf  16KB  (仅 reduceSize==1 时)
stage1PongBuf   34KB  (NZ2ND)
```

## 双缓冲判定公式

```
tile_size × 2 ≤ UB_SIZE → 可以用双缓冲
tile_size > UB_SIZE / 2 → 被迫单缓冲
```

## Triple Buffer 使用场景

FlashAttention 的 3 缓冲（taskId 0/1/2+）：
- taskId 0: Bmm1 计算
- taskId 1: Bmm1 结果处理 + Bmm2 计算（与 taskId 0 的 Bmm1 重叠）
- taskId 2: Bmm2 结果处理（与 taskId 1 的计算重叠）

3 缓冲解耦了三个独立阶段：Cube(Bmm1) → Vector(Softmax) → Cube(Bmm2) → Vector(后处理)

## GMM Pre-deferred MMCompute

grouped_matmul_add 使用软件流水线预取模式：

```cpp
while (curBlock < curCount) {
    MnBlockIdxCompute(mnConfig, ...);
    computeOp.MmCompute(lastMnConfig, blockIdx);     // 算上一个
    computeOp.MmComputePrepare(groupIdx, mnConfig);   // 准备当前
    lastGroupIdx = groupIdx;
    lastMnConfig = mnConfig;
    curBlock += coreNum;
}
computeOp.MmCompute(lastMnConfig, blockIdx);  // 算最后一个
```

第一个 MmCompute 因为 `cubeNum==0` 直接返回，仅做 prepare。这样 SetTensorA/SetTensorB 调用时上次 IterateAll+DataCopyPad 已完成 — 软件流水线化数据准备与结果输出。
