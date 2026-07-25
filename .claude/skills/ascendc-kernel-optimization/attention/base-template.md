# Attention 类 base 模板（A2A3 深度版）

以 `flash_attention_score` A2A3（arch22）为基准。内容全部 trace 自 `attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h`（2546 行），带行号。

## 数学本质（优化的出发点）

Attention = `Softmax(Q·Kᵀ / √d) · V`。两个 Bmm（矩阵乘）夹一个 Softmax：
- **Bmm1**：Q·Kᵀ → attention 分数（shape S1×S2）
- **Softmax**：对 S2 维做归一化
- **Bmm2**：attn·V → 输出

核心矛盾：**Bmm 在 Cube 单元，Softmax 在 Vector 单元**，二者必须交替。若串行执行（Bmm1 完→Softmax→Bmm2），Cube 和 Vector 互相空等，利用率减半。base 模板的全部优化都围绕"让 Cube 和 Vector 同时忙"。

## 三缓冲软件流水（base 核心，已 trace :785,861-898）

### 数据结构
```cpp
SplitExtraInfo extraInfo[3];   // :785 三槽环形缓冲，taskId % 3 寻址
event_t eventIdMte3ToMte2 = FetchEventID(Hard Event::MTE3_MTE2);  // :787
```

### 每轮（taskId = N）做的事（:861-898）

```
1. WaitBmm1Result(extraInfo[(N+2)%3])    # 等 N-2 轮的 Bmm1 完成（要复用这个槽）
2. IterateBmm1(extraInfo[N%3])            # Cube: 启动当前 N 的 Q·Kᵀ
3. ProcessVec1(extraInfo[(N+2)%3])        # Vector: 对 N-2 的 Bmm1 结果做 Softmax
4. SetFlag<MTE3_MTE2>                     # 标记 Softmax 结果已写 GM，可被下轮读
5. WaitBmm2Result()                       # 等 N-1 轮的 Bmm2 完成
6. WaitFlag<MTE3_MTE2>                    # 确认 N-2 的 Softmax 结果可读
7. IterateBmm2(extraInfo[(N+2)%3])        # Cube: 用 N-2 的 Softmax 结果做 attn·V
8. ProcessVec2(extraInfo[(N+1)%3])        # Vector: N-1 的输出后处理 + 写 GM
taskId++
```

### 为什么是三缓冲（不是双缓冲）

稳态下同时有 3 轮在飞：
- 第 N 轮：Bmm1 在 Cube 跑
- 第 N-2 轮：Softmax + Bmm2 交替（Softmax 写完 → Bmm2 读）
- 第 N-1 轮：Vec2 收尾

双缓冲只能让 2 轮重叠，Cube 做 Bmm2 时就没法做 Bmm1。三缓冲让 **Cube 在 Bmm1 和 Bmm2 间无缝切换**，Vector 在 Vec1/Vec2 间无缝切换，两边都不空等。代价：3× extraInfo 的 UB 占用。

### 为什么 MTE3_MTE2 事件对

Softmax（Vec1）结果太大放不下 UB，必须写回 GM（MTE3 = UB→GM 搬出）。Bmm2 要读这个结果（MTE2 = GM→UB/L1 搬入）。MTE3_MTE2 事件保证"写完才能读"，否则 Bmm2 读到半成品。其他事件对（如 MTE2_V）不对，因为这里同步的是**跨单元的 GM 数据可见性**，不是同单元内流水。

## enableL1Reuse + AIC-AIV 配对模式（已 trace :774-901）

### 机制
```cpp
this->l1ReuseBlockMod2 = this->blockIdx % 2;   // :775 奇偶核配对
multiCoreInnerOffset = this->blockIdx / 2 * splitFactorSize;  // :776 两核共享一份偏移
```
AIV0 和 AIV1 配对，共享 L1 里的 Q/K。Bmm1 算 Q·Kᵀ 时 Q/K 留在 L1 不重复搬入（L1Reuse），两核轮流用。

### 为什么这样优化
- **省 GM→L1 带宽**：S2 循环多轮复用同一 Q/K，不 reuse 则每轮重搬
- **AIC-AIV 配对翻倍吞吐**：两个 Vector 核共享一份 L1 数据，并行处理相邻 S1 块

### 适用范围
- S2 循环轮次多（Q/K 复用收益大）
- L1 容量够放 Q/K tile

### 代价 / 反效果
- 占 L1，挤压其他 tile
- 奇数 S1 块需 fake pair（:809-812 `needFakePair` 补空循环），有额外调度开销
- S2 少时 reuse 收益不够抵消配对开销

## 核内切分（已 trace :773,778-781）

```cpp
multiCoreInnerOffset = blockIdx * splitFactorSize;   // :773 每核负责 splitFactorSize 个块
multiCoreInnerLimit = offset + splitFactorSize;
if (totalSize < limit) limit = totalSize;            // :779 尾核不足
```
核间按 `splitFactorSize` 切分，核内循环 multiCoreInnerIdx，每个对应一个 S1 块（再内层 S2 循环）。

## Softmax 三件套（数值稳定，已 trace buffer :215-217）

```cpp
TBuf<> softmaxSumBuf[2];   // :215 指数和（归一化分母）
TBuf<> softmaxExpBuf[2];   // :216 exp 中间结果
TBuf<> softmaxMaxBuf;      // :217 每行最大值（减去后 exp 不溢出）
```
**为什么三件套**：直接 `exp(x)` 在 fp16 下 S2 大时溢出。先减行最大值（softmaxMax），再 exp（softmaxExp），累加求和（softmaxSum），最后相除。这是 **online softmax**（flash attention 的核心）——可以在分块流水中边算边更新 max/sum，不需要先看完整个 S2。双缓冲 `[2]` 配合 Ping/Pong 流水。

## 四种分核类（文件名即证据）与适用 shape

| 类 | 拆分 | 适用 | 选错后果 |
|---|---|---|---|
| S1s2Bn2gs1 | S1+S2 | S1、S2 都大 | 只拆一维 → 核不够或单核任务过大 |
| S1Bn2gs1 | 仅 S1 | S2 小、S1 大 | S2 也拆 → 无谓同步开销 |
| Bn2gs1s2B | Batch | Batch 多、单 batch 小 | 不拆 Batch → 核数不够用 |
| S1s2Bn2gs1SameAB | S1+AIC-AIV | 想榨 L1Reuse + 双 Vector | 不配对 → L1Reuse 无法启用 |

选择由 host tiling 据 shape 决定，编码进 TilingKey 的 UB0/UB1/Block 字段（`template_tiling_key.h:34-62`）。

## A5（arch35）base 差异

A5 入口 `*_apt.cpp` → `KernelTrain`（kernel_train.h:27）→ 引用 `common/op_kernel/arch35/flash_attention_noquant_kernel_base.h`。差异：
- Regbase（MicroAPI 寄存器级）替代 Membase TBuf
- 四阶流水（三阶→加 CrossCore 跨核同步阶段）
- 分核用 `CalcRealCoreIdx`（顺序/对称/TND），A5 支持同地址并行，无需 A2A3 错位规避
- 量化三档共享模块：noquant / fullquant_gqa / fullquant_mx（见 quant.md）

## 相关
- [分核详解](split-and-core.md) / [流水掩盖](pipeline.md) / [量化](quant.md)