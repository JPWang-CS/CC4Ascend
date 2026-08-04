# fused_causal_conv1d UB 占用分析 — m 扩展至 16 后不溢出论证

> **文档目的**：论证本次改动（`MAX_M` 7→16、`maxDraftTokens` ∈[0,16]、`effectiveStateLen = max(stateLen, maxDT+K-1)`）之后，三个 kernel 模板在 Ascend 910B 上 **UB 不会溢出**。
> **依据**：硬件真值只认 `AscendC_platform/Ascend910B.ini`；预算公式逐行对照 `op_host/ai_infra_fused_causal_conv1d_tiling.cpp` 与 `op_kernel/*.h` 的 `InitBuffer`。
> **覆盖**：TilingKey 0/1/2 三个模板 × 最坏情况（m=16 / maxDT=16 / K=6 / D=16384）。

---

## 1. 硬件真值：910B 的 UB 物理大小

来源：`D:\Desktop\Code\CC4Ascend\AscendC_platform\Ascend910B.ini`

| 项 | 值 |
|---|---|
| `ub_size` | **262144 字节 = 256 KB** |
| `ubbank_size` | 4096 B |
| `ubbank_num` | 64 |
| `ubblock_size` | 32 B |

预算公式里用的 `ubSize_` 就是 262144（tiling 运行时从 `GetCoreMemSize(UB)` 取，`tiling.cpp:50-52`）。
本次分析所有字节数都以 **262144** 为分母。

---

## 2. 预算机制总览：为什么 m 变大不会溢出

**核心保证**：三个模板的 buffer 全部**按 tiling data 里的 `stateLen`/`multiTokenNum` 动态分配**，tiling 在 `GenerateInfo()`（`tiling.cpp:544-668`）里按"每个核要放下的 buffer 总和"反推 `baseDim` / `ubFactor` / `maxDim`，**自动收缩**以塞进 256KB。m 变大 → `stateLen` 变大 → 预算自动收紧 → **溢出被设计排除**，代价只是性能（核切分增多）。

分三部分论证：
1. tiling 预算公式与 kernel `InitBuffer` 逐项**对得上**（无遗漏 buffer）
2. 每个核实际分配的 `curDim`/`curBaseDim` **≤ 预算用的 baseDim**
3. 数值代入最坏情况（m=16, K=6, D=16384）**余量 ≥ 0**

---

## 3. TilingKey 2（Update，decode/MTP 主路径）—— 本次重点

### 3.1 kernel 实际分配（`update.h:96-104`）

| buffer | 大小（字节） | 类型 |
|---|---|---|
| `xQueue_` | `2 × curDim × dtypeSize` | 双缓冲 |
| `weightQueue_` | `1 × K × curDim × 4` | fp32 |
| `convStatesQueue_` | `1 × stateLen × curDim × dtypeSize` | ★ 随 m 线性增长 |
| `outputQueue_` | `1 × curDim × dtypeSize` | |
| `yBuf_` | `K × curDim × 4` | fp32 |
| `xBuf_` | `curDim × 4` | fp32 |

### 3.2 tiling 预算（`tiling.cpp:612-627`）

```
db=2, fpSize=4
xQueueSpace        = 2 × dtypeSize
weightQueueSpace   = K × 4
convStateQueueSpace= stateLen × dtypeSize      ← 与 kernel 同一 stateLen
outputQueueSpace   = dtypeSize
yBufSpace          = K × 4
xBufSpace          = 4
spaceWithDim       = 上面 6 项之和                    （= 每 1 维 dim 的"单位成本"）
maxDim = ((ubSize - UB_RESERVED) / spaceWithDim / 16) × 16
```

**逐项完全一致**：tiling 的 6 个 `*Space` 与 kernel 的 6 个 `InitBuffer` 一一对应（系数 dtypeSize/4/双缓冲完全吻合），**没有遗漏的 buffer**。

### 3.3 每个核的实际用量 ≤ 预算

- `curDim_ = (blockIdx+1) % coreDimAct == 0 ? tailDim : baseDim`（`update.h:63`）→ `curDim ≤ baseDim`
- `baseDim_ = ceil(dimSize/cD)` 向上 16 对齐（`tiling.cpp:660-661`）→ `baseDim ≤ maxDim`
- 一个核的 buffer 总量 = `spaceWithDim × curDim + (stateLen×dtypeSize×curDim 里已含)`

**每个核的实际总量 ≤ spaceWithDim × baseDim ≤ spaceWithDim × maxDim = (ubSize-UB_RESERVED)/16 × 16 = ubSize - UB_RESERVED < 262144** ✓

> 因为 6 个 buffer 都带 `curDim` 因子，且 `baseDim ≤ maxDim` 由构造保证——**尾核（更小 curDim）只更安全**。

### 3.4 数值代入：最坏情况 m=16, K=6, D=16384, bf16

```
stateLen = K-1 + m = 5 + 16 = 21        （K=6, m=16 —— 最坏）
dtypeSize = 2（bf16）

xQueueSpace        = 2×2   = 4
weightQueueSpace   = 6×4   = 24
convStateQueueSpace= 21×2  = 42
outputQueueSpace   = 2
yBufSpace          = 6×4   = 24
xBufSpace          = 4
spaceWithDim       = 100

maxDim = (262144 - 1024) / 100 / 16 × 16
       = 261120 / 100 / 16 × 16
       = 2611 / 16 × 16      （261120/100=2611.2 → 整数除 2611 → 2611）
       = 2608

单核实际：spaceWithDim × maxDim = 100 × 2608 = 260800 B ≤ 261120 = ubSize - UB_RESERVED
余量：261120 - 260800 = 320 B（外加 UB_RESERVED 1024 B 本身不参与分配）
```

→ **最坏情况单核占 260800 B ≤ 261120（UB 扣除保留区），总 ≤ 262144** ✓

### 3.5 顺带验证现状（m=7, K=6）与各 dtype

| 配置 | stateLen | spaceWithDim | maxDim | 单核实际 | 余量 |
|---|---|---|---|---|---|
| bf16 m=7 K=3 | 9 | 52（=4+12+18+2+12+4） | 5008 | 260416 | 704 |
| bf16 m=16 K=3 | 18 | 70（=4+12+36+2+12+4） | 3728 | 260960 | 160 |
| bf16 m=16 K=6 | 21 | 100（=4+24+42+2+24+4） | 2608 | 260800 | 320 |
| fp16 m=16 K=6 | 21 | 100（dtypeSize 相同） | 2608 | 260800 | 320 |

> 有趣：fp16/bf16 的 dtypeSize 都是 2，预算完全一样；m 从 7→16 只是 `spaceWithDim` 里 `stateLen×2` 从 18→42 增大、`maxDim` 相应收缩，**单核总量恒 ≤ 261120**。

---

## 4. TilingKey 1（CUTBS，prefill dim<3152）—— 次重点

### 4.1 kernel 实际分配（`fn_cutbs.h:77-92`）

| buffer | 大小（字节） | 依赖 |
|---|---|---|
| `inQueueX` | `2 × (ubFactor+K-1) × baseDim × dtypeSize` | ubFactor |
| `inQueueW` | `2 × K × baseDim × dtypeSize` | |
| `cacheQue` | `2 × stateLen × baseDim × dtypeSize` | ★ 随 m |
| `yFp32` | `K × baseDim × 4` | |
| `xFp32` | `(ubFactor+K-1) × baseDim × 4` | |
| `weightFp32` | `K × baseDim × 4` | |
| `cacheBuf` | `stateLen × baseDim × dtypeSize` | ★ 随 m |
| `outQueue` | `ubFactor × baseDim × dtypeSize` | |

### 4.2 tiling 预算（`tiling.cpp:586-600`）

```
inQueueX_ = 2 × (K-1) × dtypeSize × 512       # 双缓冲 × (ubFactor 里 K-1 固定项)
inQueueW  = 2 × K × dtypeSize × 512
cacheQue  = 2 × stateLen × dtypeSize × 512
yBuf      = K × 4 × 512
xFp32Buf_ = (K-1) × 4 × 512
weightBuf = K × 4 × 512
cacheBuf  = stateLen × dtypeSize × 512
inQueueX  = 2 × dtypeSize × 512               # ubFactor 的单位行成本
outQueue  = dtypeSize × 512
xFp32Buf  = 4 × 512

spaceWithUbfactor = ubSize - (UB_RESERVED/2 + 固定7项)
ubFactor_ = spaceWithUbfactor / (inQueueX + outQueue + xFp32Buf)
```

**预算覆盖了全部 8 个 buffer**：`inQueueX` 拆成"固定 (K-1) + 每行 1"两项、`xFp32` 同理；`outQueue` 按 ubFactor 行 × baseDim 算。`ubFactor` 每多一行要多占 `(2·dtypeSize + dtypeSize + 4)·baseDim`，预算把它从 256KB 里"逐行扣" → **每行成本 ≤ 剩余空间**。

### 4.3 数值代入：最坏 m=16, K=6, baseDim=512, bf16

```
固定项（stateLen=21）：
  inQueueX_ = 2×5×2×512 = 10240
  inQueueW  = 2×6×2×512 = 12288
  cacheQue  = 2×21×2×512 = 43008
  yBuf      = 6×4×512 = 12288
  xFp32Buf_ = 5×4×512 = 10240
  weightBuf = 6×4×512 = 12288
  cacheBuf  = 21×2×512 = 21504
  固定合计 = 121856
行成本 = inQueueX(2×2×512=2048) + outQueue(2×512=1024) + xFp32Buf(4×512=2048) = 5120
ubFactor = (262144 - (1024/2 + 121856)) / 5120
         = (262144 - 512 - 121856) / 5120
         = 139776 / 5120 = 27.3 → 27

单核总 = 512 + 121856 + 27×5120 = 512 + 121856 + 138240 = 260608 B ≤ 262144 ✓
```

→ **CUTBS 最坏（m=16, K=6）ubFactor=27，单核 260608 B，余量 1536 B** ✓

> 注：当前线上 m=7、K=3 时 `stateLen=9`，固定项 54272、行成本 5120，`ubFactor=(262144-512-54272)/5120=40`，单核 259584 B（余量 2560）——比改动后宽松，量级与原注释"max baseDim=512, ubFactor=30"一致（预算多扣了 UB_RESERVED/2，偏保守）。

---

## 5. TilingKey 0（CUTBSD，prefill dim≥3152）—— 不受影响

### 5.1 kernel 实际分配（`fn_cutbsd.h:81-94`）

| buffer | 大小（字节） |
|---|---|
| `inQueueX` | `2 × baseDim × dtypeSize` |
| `inQueueW` | `K × baseDim × 4` |
| `yBuf` | `K × baseDim × 4` |
| `xFp32Buf` | `baseDim × 4` |
| `convStatesQue` | `2 × baseDim × dtypeSize` |
| `outQueue` | `baseDim × dtypeSize` |

**没有任何 buffer 含 `stateLen`** —— CUTBSD 逐 token 处理，缓存行只用 `baseDim` 大小。**m 扩展对 CUTBSD 的 UB 完全零影响**。

### 5.2 tiling 预算（`tiling.cpp:550-559`）与数值

```
spaceWithUbfactor = 2·dtypeSize + K·4 + K·4 + 4 + 2·dtypeSize + dtypeSize
                  = 4 + 24 + 24 + 4 + 4 + 2 = 62（K=6）
baseDim = (262144 - 1024/2)/62 /16×16 = 261632/62/16×16 = 4219/16×16 = 4208
```

单核实际 = `62 × baseDim(=4208) = 260896 ≤ 261632` ✓。**与 m 无关，本次不涉及**。

---

## 6. maxDraftTokens 新机制如何影响 UB（tiling.cpp:276-278）

```cpp
int64_t draftStateLen = maxDraftTokens_ + windowSize_ - NUM_ONE;   // maxDT+K-1
int64_t effectiveStateLen = stateLen_ > draftStateLen ? stateLen_ : draftStateLen;
stateLen_ = effectiveStateLen;
```

- **不传/默认 maxDT=7**：`effectiveStateLen = max(stateLen, 9)`。对 m≤7 的旧调用，`stateLen ≤ 9` → `effectiveStateLen = 9` → **UB 与改动前完全一致**（兼容性保证）。
- **显式 maxDT=16**：`effectiveStateLen = max(stateLen, 21)`（K=6）。此时 **UB 按 maxDT=16 分配**——即按"最坏 m=16"分配，正是 §3.4 / §4.3 已验证的两条最坏路径，**均不溢出**。
- **层3 保护**：`m > maxDT` 的调用在 tiling 阶段被拒（`tiling.cpp:269-274`）——**不允许出现"UB 按小 maxDT 分配、实际 m 却更大"的越界组合**，从入口杜绝溢出。

三套机制（预算动态收缩 + 层3 拦截 + effectiveStateLen）**共同保证任何合法调用不溢出**。

---

## 7. 汇总表

| TilingKey | 模板 | m 依赖 | 最坏单核（m=16,K=6,D=16384） | 余量 | 溢出风险 |
|---|---|---|---|---|---|
| 0 | CUTBSD | **无** | 260896 B（K=6, baseDim=4208） | 736 B | 无（与 m 无关） |
| 1 | CUTBS | 强（cacheQue/cacheBuf） | 260608 B（ubFactor=27） | 1536 B | 无（ubFactor 逐行收缩） |
| 2 | Update | 强（convStatesQueue） | 260800 B（maxDim=2608） | 320 B | 无（maxDim 收缩） |

**结论**：910B UB=256KB，三模板最坏情况单核均 ≤ 260896 B，全部在 262144 内，且有 320~1536 B 的预算内余量（tiling 还额外扣了 UB_RESERVED=1024/512 B）。m 7→16 只是让 tiling 把 `baseDim`/`ubFactor`/`maxDim` 收缩得更紧，**溢出被预算公式设计排除**。

---

## 8. 上板验证计划（把"算出来的"变成"测出来的"）

预算论证是静态证据，真实 kernel 路径仍需上板确认：

1. **msprof 抓 UB 用量**：D=16384 × m=16 × W=6（最坏），确认实际占用 < 256KB 且无溢出报错
2. **边界回归**：m=7 不传 maxDT（应无性能劣化）、maxDT=16 + m=16（最坏）、maxDT=16 + m=7（UB 浪费但合法）
3. **预期**：tiling 日志里 `maxDim/ubFactor` 随 stateLen 收缩可观测（`tiling.cpp:775` OP_LOGD 已打 ubFactor）

---

## 附：预算公式 vs kernel InitBuffer 逐行对照表（核对依据）

| 模板 | tiling 预算项 | kernel InitBuffer | 系数一致 |
|---|---|---|---|
| Update | xQueueSpace=2·dtypeSize | xQueue_=2·curDim·dtypeSize | ✓（curDim=1 单位） |
| Update | convStateQueueSpace=stateLen·dtypeSize | convStatesQueue_=stateLen·curDim·dtypeSize | ✓ |
| Update | weightQueueSpace=K·4 | weightQueue_=K·curDim·4 | ✓ |
| Update | outputQueueSpace=dtypeSize | outputQueue_=curDim·dtypeSize | ✓ |
| Update | yBufSpace=K·4 | yBuf_=K·curDim·4 | ✓ |
| Update | xBufSpace=4 | xBuf_=curDim·4 | ✓ |
| CUTBS | cacheQue=2·stateLen·dtypeSize·512 | cacheQue=2·stateLen·baseDim·dtypeSize | ✓ |
| CUTBS | cacheBuf=stateLen·dtypeSize·512 | cacheBuf=stateLen·baseDim·dtypeSize | ✓ |
| CUTBS | inQueueX_=2·(K-1)·dtypeSize·512 + inQueueX 行成本 | inQueueX=2·(ubFactor+K-1)·baseDim·dtypeSize | ✓ |
| CUTBS | xFp32Buf_=(K-1)·4·512 + xFp32Buf 行成本 | xFp32=(ubFactor+K-1)·baseDim·4 | ✓ |
| CUTBSD | 6 项均无 stateLen | 6 个 InitBuffer 均无 stateLen | ✓ |
