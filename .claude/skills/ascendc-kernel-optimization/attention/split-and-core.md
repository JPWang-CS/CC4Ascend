# Attention 分核优化

内容均经真实代码 trace。源：`attention/flash_attention_score/op_kernel/arch22/` 与 `arch35/`。

## A2A3 分核（arch22）

### 四种分核 Kernel 类（文件名即证据，已验证）

| 文件 | 类 | 拆分维度 |
|---|---|---|
| `flash_attention_score_s1s2_bn2gs1.h` | S1s2Bn2gs1 | S1 + S2 |
| `flash_attention_score_s1_bn2gs1.h` | S1Bn2gs1 | 仅 S1 |
| `flash_attention_score_bn2gs1s2_b.h` | Bn2gs1s2B | Batch |
| `flash_attention_score_s1s2_bn2gs1_sab.h` | S1s2Bn2gs1SameAB | S1 + AIC-AIV 配对 |

选哪个类由 host tiling 根据 shape 决定，通过 TilingKey 的 **UB0/UB1/Block** 字段编码（`template_tiling_key.h:34-62`，各 4 bit，取值 0/1/2/3/4/5/9）。

### 分核索引计算（已 trace s1s2_bn2gs1.h:138,356,379-381）
```
ComputeAxisIdx(multiCoreInnerIdx)   # 算 S1/S2 块索引
blockIdx = GetBlockIdx()            # 普通模式
blockIdx = GetBlockIdx() * 2        # SameAB：AIC-AIV 配对（subBlockIdx 偏移）
if blockIdx < coreNum: ...          # 核内多轮迭代（:356）
```
关键 tile 尺寸：`s1BaseSize` / `s2BaseSize`（:182-183），由 host tiling 据 UB 容量算出。

### SameAB 模式（AIC-AIV 配对，已 trace :381）
`blockIdx = GetBlockIdx() * 2`：一个物理核内 AIC（Cube）与 AIV（Vector）配对，AIC 算 Bmm、AIV 算 softmax/output，减少跨核同步。适用于 S1 较小、想充分利用 AIC-AIV 并行的场景。

## A5 分核（arch35）

### 三种模式（已验证 grep，在 CalcRealCoreIdx 实现）
1. **顺序分核**：s1 块依次分发给 core0/1/2...，相邻核处理相邻 s1 块，提高 L2 命中
2. **对称分核**：N 维上半顺序 + 下半对称，平衡多核负载
3. **TND（正倒序循环）**：正序分发后倒序回发，`varlenCycleCoreNums = coreNum × 2`，适配 varlen 场景的动态核数

### A2A3 → A5 分核差异
- A2A3 需**错位规避同地址冲突**（多核写同一 GM 地址）
- A5 硬件支持**同地址并行**，可简化为规则分核（顺序/对称/TND），减少无效偏移

## 相关 TilingKey 字段（已 trace template_tiling_key.h）

| 字段 | bit 宽 | 说明 |
|---|---|---|
| UB0 | 4 | S1 方向拆分控制 |
| UB1 | 4 | S2/D 方向拆分控制 |
| Block | 4 | Batch 方向拆分控制 |
| BigDoubleBuffer | 2 | 大双缓冲开关 |
| EnableL1Reuse | 1 | L1 复用（减 GM 往返） |
| Sparse | 4 | 稀疏模式 |

## 优化偏离判断

- **S1 大、S2 大** → S1s2Bn2gs1（S1+S2 双拆）
- **S1 小、S2 大** → S1Bn2gs1（仅 S1，S2 整体循环）
- **Batch 多、单 batch 小** → Bn2gs1s2B（Batch 拆分）
- **想榨 AIC-AIV 并行** → SameAB（blockIdx*2 配对）