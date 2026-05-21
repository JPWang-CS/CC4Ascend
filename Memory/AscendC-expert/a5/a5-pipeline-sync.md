# A5 流水同步

## 四阶流水线（FlashAttention A5）

A5 使用 4 级 runInfo 缓冲区：

```
taskId 0:       IterateBmm1
taskId 1:       IterateBmm1 → ProcessVec1
taskId 2:       IterateBmm1 → ProcessVec1 → IterateBmm2
taskId 3+:      IterateBmm1 → ProcessVec1 → IterateBmm2 → ProcessVec2
```

比 A2A3 的 3 缓冲多一级，允许更深的流水重叠。

## CrossCoreSetFlag / CrossCoreWaitFlag

A5 新增跨核同步原语，用于 AIC-AIV 配对核通信：

### 初始化（block_sparse_attention A5）

```cpp
template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
void InitSyncFlags() {
    // CUBE: CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(2), (18), (3), (19), (4), (20)
    // VEC:  CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(0), (1)
    //       CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(5), (6)
}
```

三个独立可配置的跨核同步模式：
- **MM1_SM_MODE**：MM1 → Softmax 同步（mode=4）
- **MM2_RE_MODE**：MM2 → Rescale 同步（mode=4）
- **SM_MM2_MODE**：Softmax → MM2 同步（mode=4）

### GMM 跨核同步

```cpp
// AIC 侧
CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
// ... Cube 计算 ...
CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);

// AIV 侧
CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
// ... Vector 计算（Dequant + SwiGLU + Quant）...
CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);
```

mode=2 表示组内同步，mode=4 表示全流水同步。eventID 可达 22。

## cgmct 框架流水线

A5 quant_grouped_matmul_inplace_add 使用 cgmct 模块化调度：

```cpp
// BlockMmadFixpipeDequant: Cube + Fixpipe + Dequant
// BlockEpilogueDequant: 最终 Dequant（支持 PERTENSOR + PERCHANNEL）
// GroupedMatmulAswtWithTailSplitScheduler: 对角线分组调度
```

三个 cgmct 变体：
1. **CubeBasicAPI**：`BlockMmadFixpipeDequant` + `BlockEpilogueEmpty`（fixpipe 内 dequant）
2. **CubeOnTheFly**：`QuantMatmulMultiBlock` + `BlockEpilogueEmpty`（on-the-fly quant matmul）
3. **MixOnlineDynamic**：`MatmulMultiBlock` + `BlockEpilogueDequant`（vector epilogue dequant）

## TSCM 双缓冲

A5 IFA 引入 TSCM（TSCM double buffer）用于 work queue：

```cpp
TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPing;
TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPong;
tPipe.InitBuffer(vec1ScmPing, 1, vec1ResultSize);
tPipe.InitBuffer(vec1ScmPong, 1, vec1ResultSize);
```

TSCM 提供比 TQue 更轻量的 ping-pong 管理，适用于 AIC-AIV 间细粒度数据传递。

## 三阶流水线（GMM SwiGLU Quant A8W4 MSD）

```
for each workspaceSplitLoopIdx:
  // Stage 1: PreProcess (n+1) - weight cast in AIV
  // Stage 2: MidProcess (n) - matmul in AIC
  // Stage 3: PostProcess (n-1) - dequant + SwiGLU + requant in AIV
```

滑动窗口：迭代 n 的 MidProcess 与 n+1 的 PreProcess 和 n-1 的 PostProcess 重叠。

## A5 死锁排查要点

- CrossCoreSetFlag/WaitFlag 必须在**所有分支路径**上成对出现
- flagId 不可跨 pipeline 复用
- 循环内必须确保每次迭代的 Set/Wait 对齐（循环边界处特别容易漏）
- `SyncAll<false>()` / `SyncAll<true>()` 用于全核 Barrier
