# A2A3 流水同步

## 三缓冲流水线模式

A2A3 FlashAttention S1S2 使用 **三缓冲**（taskId 0/1/2+）：

```
taskId 0:       IterateBmm1        (无等待)
taskId 1:       WaitBmm1Result → IterateBmm1 → ProcessVec1 → SetFlag(MTE3_MTE2) → WaitFlag(MTE3_MTE2) → IterateBmm2
taskId 2+:      WaitBmm1Result → IterateBmm1 → ProcessVec1 → SetFlag → WaitFlag → IterateBmm2 → WaitBmm2Result → ProcessVec2
```

## ProcessVec1 内部事件链（S1S2）

```
For each loopIdx:
  WaitBmm1Result (MTE3_MTE2 event)
  GetBmm1Result → SetFlag(MTE2_V) → WaitFlag(MTE2_V)
  Muls(scale)
  CopyInAttenMask → SetFlag(MTE2_V) → WaitFlag(MTE2_V)
  ComputeAttenMask
  [PSE 条件路径 SetFlag/WaitFlag]
  SetFlag(V_MTE2) / WaitFlag(V_MTE2)  // inter-vec ping-pong
  CopyInDropMask (条件路径)
  SoftMaxCompute
  ComputeDropMask (条件路径)
  Cast + DataCopy / NdToNz
  SetFlag(V_MTE3) → WaitFlag(V_MTE3)
```

## Events 全谱系

| Event | 含义 |
|-------|------|
| `MTE2_V` | MTE2 搬入完成 → Vector 可开始 |
| `V_MTE2` | Vector 完成 → MTE2 可继续 |
| `MTE3_V` | MTE3 完成 → Vector 可开始 |
| `V_MTE3` | Vector 完成 → MTE3 可搬出 |
| `MTE3_MTE2` | MTE3 完成 → MTE2（跨任务握手） |
| `MTE2_MTE3` | MTE2 完成 → MTE3 |
| `S_V` | Scalar → Vector（sink 初始值） |

## Quant Dequant 完整事件链

quant_grouped_matmul_dequant 的 ProcessX：

```
Init: SetFlag(V_MTE2)[0] + SetFlag(MTE3_V)[0,1]  // 预置所有 event 就绪

Loop over K:
  SetFlag(V_MTE2)[1] → WaitFlag(V_MTE2)[1]        // Copy per-token scale
  WaitFlag(V_MTE2)[0]                               // 等上一轮
  DataCopy X                                         // MTE2 搬入
  SetFlag(MTE2_V)[0]                                 // 通知 Vector
  WaitFlag(MTE2_V)[0]
  // Vector compute (Cast, Mul, Muls...)
  WaitFlag(MTE3_V)[0]
  TransDataTo5HD                                     // ND→NZ
  SetFlag(V_MTE2)[0]                                 // 通知下一轮 Copy
  Cast<int8_t>
  SetFlag(V_MTE3)                                    // 通知 MTE3 搬出
  WaitFlag(V_MTE3)
  DataCopy to GM
  SetFlag(MTE3_V)[0]                                 // 通知下一轮 Pipeline
```

## B 拆分流水线

```
taskId 0: IterateBmm1
taskId 1: WaitBmm1Result → IterateBmm1 → ProcessVec1 → SetFlag(MTE3_MTE2) → WaitFlag → IterateBmm2
taskId 2+: WaitBmm1Result → IterateBmm1 → ProcessVec1 → SetFlag → WaitFlag → IterateBmm2 → WaitBmm2Result → ProcessVec2
```

## GM Workspace 布局（双缓冲）

```
mm1Res[2]    = 2 * mmNRatioOffset     (bmm1 结果, ping-pong)
stage1Res[2] = 1x or 2x mmNRatioOffset (NZ/FP32 需要额外空间)
mm2Res[2]    = 2 * mm2Offset           (bmm2 结果, ping-pong)
vec2Res[2]   = 2 * mm2Offset           (vec2 结果, ping-pong)
totalOffset  = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER
```
