# SetFlag / WaitFlag 流水线同步

## 功能说明

同一核内**不同流水线之间**的同步指令。NPU 内部多条流水线（MTE1/MTE2/MTE3/VECTOR/CUBE/FIXPIPE）异步并行执行，当存在数据依赖时必须插入同步。

| API | 行为 |
|-----|------|
| SetFlag | 前序指令**所有读写完成后**，硬件标志位设为 1 |
| WaitFlag | 若标志位为 0 则阻塞；若为 1，置 0 并放行 |

**SetFlag/WaitFlag 必须成对出现。**

## 函数原型

```cpp
// 老版本（TEventID）
SetFlag(TEventID id);
WaitFlag(TEventID id);

// ISASI 新版本（HardEvent 枚举）
template <HardEvent event>
SetFlag(int32_t eventID);

template <HardEvent event>
WaitFlag(int32_t eventID);
```

## 同步事件类型 (HardEvent)

| 事件 | 含义 | 源 → 目标流水线 |
|------|------|----------------|
| `MTE2_V` | MTE2 搬入完成 → Vector 可开始 | MTE2 → VECTOR |
| `V_MTE2` | Vector 完成 → MTE2 可继续 | VECTOR → MTE2 |
| `MTE3_V` | MTE3 完成 → Vector 可开始 | MTE3 → VECTOR |
| `V_MTE3` | Vector 完成 → MTE3 可搬出 | VECTOR → MTE3 |
| `MTE2_MTE3` | MTE2 完成 → MTE3 可开始 | MTE2 → MTE3 |
| `MTE3_MTE2` | MTE3 完成 → MTE2 可开始 | MTE3 → MTE2 |
| `MTE1_MTE2` | MTE1 完成 → MTE2 可开始 | MTE1 → MTE2 |
| `S_V` | Scalar 完成 → Vector 可开始 | SCALAR → VECTOR |
| `V_S` | Vector 完成 → Scalar 可开始 | VECTOR → SCALAR |
| `S_MTE3` | Scalar 完成 → MTE3 可搬出 | SCALAR → MTE3 |

## 典型同步模式

### 三阶段流水线

```
MTE2: DataCopy(GM→UB) → SetFlag(MTE2_V) → WaitFlag(V_MTE2) → 下一轮
                                ↓
VECTOR: WaitFlag(MTE2_V) → 计算 → SetFlag(V_MTE3)
                                        ↓
MTE3: WaitFlag(V_MTE3) → DataCopy(UB→GM) → SetFlag(MTE3_V)
```

### eventID 获取（ISASI 新版本）

```cpp
int32_t eventID = GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3);
SetFlag<HardEvent::MTE2_MTE3>(eventID);
WaitFlag<HardEvent::MTE2_MTE3>(eventID);
```

## A5 跨核同步：CrossCoreSetFlag / CrossCoreWaitFlag

A5 新增跨核同步原语用于 AIC-AIV 配对核通信：

```cpp
// 初始化跨核同步（A5 block_sparse_attention）
InitSyncFlags<4, 4, 4>();  // <MM1_SM_MODE, MM2_RE_MODE, SM_MM2_MODE>

// 跨核 Set/Wait
CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
```

跨核同步模式值（mode=2 表示组内同步，mode=4 表示全流水同步），使用更高 eventID（可达 22）。

## 注意事项

- 大多数场景 TPipe + TQue 已自动处理同步（EnQue/DeQue 内部包含 SetFlag/WaitFlag）
- 仅在手动管理 Buffer 或跨流水线精细控制时显式使用
- A5 必须确保 CrossCoreSetFlag 和 CrossCoreWaitFlag 在 **所有分支路径** 上成对出现，否则死锁
