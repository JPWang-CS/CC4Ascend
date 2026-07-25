# Attention 流水掩盖优化

内容均经真实代码 trace。源：`attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h`。

## A2A3 流水结构（arch22）

### 三阶段 Cube↔Vector 交替（已 trace :95,131,140,155,172）
```
IterateBmm1(extraInfo)    # Cube: QK = Bmm1（MTE2 搬入 Q/K → MatMul → MTE3 搬出）
ComputeBmm1Tail            # 尾块处理
ProcessVec1(extraInfo)     # Vector: Softmax（scale + mask + exp + sum）
IterateBmm2(extraInfo)     # Cube: AV = Bmm2（MTE2 搬入 A → MatMul → MTE3 搬出）
ProcessVec2(extraInfo)     # Vector: 输出后处理 + 搬出 GM
```

### 双缓冲 Ping/Pong（已 trace :209-219）
```
maskTBufPing / maskTBufPong      # attenmask 双缓冲（:507-509，Ping=stage1AttenSize，Pong=16KB）
stage1PingBuf / stage1PongBuf    # Bmm1 结果双缓冲（轮转掩盖 Bmm1↔Vec1）
stage2TBuf                         # Bmm2 结果（:513）
softmaxSumBuf[2] / softmaxExpBuf[2]  # softmax 状态双缓冲（配合 Ping/Pong）
```
Ping/Pong 让第 n 轮 Bmm1 的搬出（MTE3）与第 n+1 轮 Bmm1 的搬入（MTE2）overlap。

### 同步事件（已 trace :787,882,891,948,957）
```cpp
event_t eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);   # 搬出完成
WaitFlag<Hard Event::MTE3_MTE2>(eventIdMte3ToMte2);  # 等待可搬入
```
`MTE3_MTE2` 事件保证 Bmm 结果 GM 可见后，下轮输入才能搬入。

## BigDoubleBuffer 优化（已 trace TilingKey 字段 :112）

TilingKey `BigDoubleBuffer`（2 bit，取值 0/1/2）：开启更大双缓冲，增加流水掩盖深度。适用场景：UB 足够大、S2 循环轮次多。风险：挤占 softmax/mask buffer。

## EnableL1Reuse 优化（已 trace TilingKey 字段 :132）

`EnableL1Reuse`（1 bit）：Q/K 在 L1 复用，减少 GM→L1 重复搬入。适用：S2 多轮循环复用同一 Q/K 块。风险：L1 占用增加，可能挤压其他 tile。

## A5 流水（arch35）

### 四阶流水 + 跨核同步
A5 用 `CrossCoreSetFlag` / `CrossCoreWaitFlag`（mode 2/4）实现多核四阶流水（MTE2→MTE1→Cube→MTE3→Vec），比 A2A3 三阶段更细。详见 ascendc-development §核间同步。

### A5 流水注意
A5 要求 `CrossCoreSetFlag`/`CrossCoreWaitFlag` **严格一一匹配**，无 HWTS 兜底，不匹配必死锁。异常分支（提前 return）最易踩：只 Set 没 Wait 或反之。

## 优化偏离判断

- **S2 轮次多、UB 够** → 开 BigDoubleBuffer 加深流水
- **Q/K 复用度高** → 开 EnableL1Reuse 省 GM 往返
- **A5 迁移** → 三阶段→四阶段，补 CrossCore 同步，检查异常分支 Set/Wait 配对