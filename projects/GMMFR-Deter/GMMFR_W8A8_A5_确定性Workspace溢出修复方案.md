# GMMFR W8A8 A5 确定性 Workspace 溢出修复方案

> 对比 A3 确定性实现，分析当前代码 deterWorkspace 溢出风险
> 方案：将 FR flush 移入 ProcessSingleGroup while 循环，在 matmul 前判断
> 审查日期: 2026-05-30

---

## 1. 问题分析

### 1.1 当前架构 — Sliding Window

```
Workspace = userWorkspace (96MB/64MB, host tiling 决定)
windowSize = deterWorkspaceSize / (N * sizeof(CType))   ← 可容纳的 M 行数
lowBoundM  = windowStartM + windowSize                   ← 水位线
```

每个 tile 写入 workspace 的位置计算（`LinearWriteToWorkspace`）：

```
wsRow = groupAccumM_ + offsetM + i
wsIdx = wsRow * n_ + yOffset
```

其中 `offsetM` 已包含 `tileMOffset`（tile 在当前 group 内的 M 偏移），
`groupAccumM_ = cumulativeGroupM_ - windowStartM`。

### 1.2 溢出根因

**当前溢出检查在 group 边界**（`operator()` line 395）：

```cpp
if (isDeter_ && (cumulativeGroupM_ + groupM) > deterSync_.lowBoundM) {
    // FLUSH → windowStartM = cumulativeGroupM_
    //        lowBoundM = cumulativeGroupM_ + windowSize
}
ProcessSingleGroup(params, bs, groupIdx);  // 写入 groupM 行
```

**溢出场景**：当 `groupM > windowSize` 时：

```
cumulativeGroupM_=0, windowSize=6144, lowBoundM=6144
Group 0: M=15000

group 级检查: 0+15000 > 6144 → FLUSH
  curGroupM=0, totalM=0-0=0   ← 无效 flush（workspace 为空）
  windowStartM=0, lowBoundM=6144   ← 不变

ProcessSingleGroup:
  Tile 0~47: 写入 workspace[0..6143]     ✓ 安全
  Tile 48:   写入 workspace[6144..6271]  ✗ 溢出！
```

### 1.3 A3 vs A5 粒度对比

| | A3 (`grouped_matmul_finalize_routing.h`) | A5 (`kernel_..._deter.h`) |
|---|---|---|
| 溢出检查位置 | `VectorSync()` — 每个 tile 的 Cube 之后 | `operator()` — 每个 group 之前 |
| 检查条件 | `curBlockM > lowBoundM` | `cumulativeGroupM_ + groupM > lowBoundM` |
| 检查粒度 | tile 级 | group 级 |
| groupM > windowSize 时 | 安全（多 tile 过程分批 flush） | **溢出** |

A3 的 `VectorSync` 代码（line 622-649）：

```cpp
// A3: 每次 tile 后检查，可能多次 flush
while (mnConfig.curBlockM > syncConfig.lowBoundM) {
    // 推进到 group 边界或 block 边界
    while (syncConfig.curGroup < tiling->groupNum) {
        uint32_t mi = ...;
        if (syncConfig.curGroupM + mi <= syncConfig.lowBoundM) {
            syncConfig.curGroupM += mi;
            syncConfig.curGroup++;
        } else {
            syncConfig.curM += (syncConfig.lowBoundM - syncConfig.curM) / mnConfig.singleM * mnConfig.singleM;
            break;
        }
    }
    FRDeterministic(syncConfig);  // flush
    syncConfig.lowBoundM = syncConfig.curM + syncConfig.windowSize;
}
```

---

## 2. 修复方案

### 2.1 核心思路

将 overflow 检查从 group 边界**移入 `ProcessSingleGroup` 的 while 循环**，在 `mmadOp_()` 之前判断当前 tile 是否会写越界，越界则先 flush。

```
原: group 边界检查一次 → ProcessSingleGroup(全部 tile)
新: ProcessSingleGroup { for each tile: 检查 → matmul → write }
```

### 2.2 修改总览

| # | 文件 | 位置 | 操作 |
|---|------|------|------|
| 1 | `kernel_gmm_finalize_routing_pertoken_dequant_deter.h` | `ProcessSingleGroup` while 循环内，line 296 之后 | **新增** tile 级溢出检查 |
| 2 | 同上 | `ProcessSingleGroup` AIV 路径，line 312-313 | **不改**（负值抵消，确认安全） |
| 3 | 同上 | `operator()` for 循环内，lines 395-402 | **删除** group 级溢出检查 |

---

## 3. 修改 1 — tile 级溢出检查

### 3.1 代码

在 `ProcessSingleGroup` 的 while 循环内，`int64_t tileMOffset = y / n;` 之后插入：

```cpp
while (bs.GetTileIdx(tileIdx)) {
    BlockShape singleShape = bs.GetBlockShape(tileIdx);
    blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::PERTOKEN_MODE>(
        Get<DETER_IDX_M_TILEIDXS>(tileIdx), Get<DETER_IDX_N_TILEIDXS>(tileIdx),
        Get<DETER_IDX_M_TAIL_SPLIT_TILEIDXS>(singleShape),
        Get<DETER_IDX_N_TAIL_SPLIT_TILEIDXS>(singleShape));
    int64_t y = Get<DETER_IDX_C_OFFSETS>(blockOffset_);
    int64_t tileMOffset = y / n;

    // ===== [新增] tile 级溢出检查 =====
    if (isDeter_) {
        uint64_t tileEndAbsM = cumulativeGroupM_
                             + (uint64_t)tileMOffset
                             + (uint64_t)Get<MNK_M>(singleShape);
        if (tileEndAbsM > deterSync_.lowBoundM) {
            deterSync_.curGroupM = cumulativeGroupM_ + (uint64_t)tileMOffset;
            SyncAll<false>();
            if ASCEND_IS_AIV { FinalFlush(params); }
            deterSync_.windowStartM = deterSync_.curGroupM;
            deterSync_.lowBoundM = deterSync_.curGroupM + deterSync_.windowSize;
            SyncAll<false>();
        }
    }
    // ===== [新增结束] =====

    if ASCEND_IS_AIC {
        if (isVecSetSyncCom_) { WaitForVector(); }
        AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{
            Get<MNK_M>(singleShape), Get<MNK_N>(singleShape), k};
        mmadOp_(aGlobal_[Get<DETER_IDX_A_OFFSETS>(blockOffset_)],
                bGlobal_[Get<DETER_IDX_B_OFFSETS>(blockOffset_)],
                l0cOutUb_, mmSingleShape, transA, transB);
        NotifyVector();
    }
    // ... 后续不变
```

### 3.2 数据安全论证

**检查条件**：`tileEndAbsM = cumulativeGroupM_ + tileMOffset + tileM`

FLUSH 前，workspace 保存了 `[windowStartM, cumulativeGroupM_ + tileMOffset)` 范围内的所有行。因为该 tile 的 matmul 还没跑，这些行都是完整写入的。

**FLUSH 参数**：`curGroupM = cumulativeGroupM_ + tileMOffset`
- `totalM = curGroupM - windowStartM` = 当前 workspace 中完成的所有行数
- `FinalFlush` 从 workspace 位置 `[0, totalM)` 读出，通过 `GetRowIndex(windowStartM + mOffset)` 映射到输出行

**FLUSH 后重置**：
- `windowStartM = curGroupM` → 新的窗口起点
- `lowBoundM = curGroupM + windowSize` → 新水位线

### 3.3 同步正确性

- FLUSH 前 `SyncAll<false>()`：确保所有 core 的 AIC/AIV 已完成前一个 tile → workspace 数据完整
- FLUSH 后 `SyncAll<false>()`：确保所有 core 的 `windowStartM`/`lowBoundM` 一致后再继续
- FLUSH 发生在 AIC/AIV 分支之前，AIC 侧也在 SyncAll 等待

### 3.4 与 isVecSetSyncCom_ 的交互

```
Tile 0: isVecSetSyncCom_=false, 可能触发 flush
        flush 后进入 AIC 分支: WaitForVector() 被 if(isVecSetSyncCom_) 跳过 ✓
        AIC matmul, set isVecSetSyncCom_=true
        AIV WaitForCube + process

Tile 1: isVecSetSyncCom_=true, 可能触发 flush
        flush 后进入 AIC 分支: WaitForVector() 执行 ✓ (等前一个 tile 的 AIV)
```

flush 发生在 `isVecSetSyncCom_` 的更新之前，AIC 的 `WaitForVector()` 受 `isVecSetSyncCom_` 保护，逻辑一致。

---

## 4. 修改 2 — SetWorkspaceGroupOffset 不改

### 4.1 原结论回顾

`SetWorkspaceGroupOffset(cumulativeGroupM_ - windowStartM)` 在 flush 后产生负值，但 `offsetM` 中的 `tileMOffset` 恰好抵消，**无需修改**。

### 4.2 数学验证

```
wsRow = groupAccumM_ + offsetM + i
      = (cumulativeGroupM_ - windowStartM) + (tileMOffset + intraTileOffset) + i
      = cumulativeGroupM_ - windowStartM + tileMOffset + intraTileM

FLUSH 后: windowStartM = cumulativeGroupM_ + X    (X = flush 时的 tileMOffset)

wsRow = cumulativeGroupM_ - (cumulativeGroupM_ + X) + tileMOffset + intraTileM
      = tileMOffset - X + intraTileM

tile 按 tileMOffset 递增处理，flush 后的 tile 满足 tileMOffset ≥ X
所以 tileMOffset - X ≥ 0，wsRow ≥ 0 始终成立 ✓
```

### 4.3 数值例子

| 迭代 | cumM | windowStartM | tileMOffset | groupAccumM_ | wsRow 首行 | 安全? |
|------|------|-------------|-------------|-------------|-----------|-------|
| T47 | 0 | 0 | 6016 | 0 | 6016 | ✓ |
| **flush** | 0 | **6144** | -- | -- | -- | -- |
| T48 | 0 | 6144 | 6144 | -6144 | 0 | ✓ |
| T49 | 0 | 6144 | 6272 | -6144 | 128 | ✓ |
| **flush** | 0 | **12288** | -- | -- | -- | -- |
| T96 | 0 | 12288 | 12288 | -12288 | 0 | ✓ |
| T97 | 0 | 12288 | 12416 | -12288 | 128 | ✓ |

**结论：不改。**

---

## 5. 修改 3 — 移除 group 级溢出检查

### 5.1 删除代码

`operator()` 内 for 循环中，lines 395-402 整块删除：

```cpp
// ===== 删除以下 =====
if (isDeter_ && (cumulativeGroupM_ + (uint64_t)Get<MNK_M>(problemShape_)) > deterSync_.lowBoundM) {
    deterSync_.curGroupM = cumulativeGroupM_;
    SyncAll<false>();
    if ASCEND_IS_AIV { FinalFlush(params); }
    deterSync_.windowStartM = deterSync_.curGroupM;
    deterSync_.lowBoundM = deterSync_.curGroupM + deterSync_.windowSize;
    SyncAll<false>();
}
// ===== 删除结束 =====
```

删除后的 for 循环：

```cpp
for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
    if (!UpdateGroupParams(params, groupIdx)) {
        continue;
    }
    ProcessSingleGroup(params, bs, groupIdx);
    cumulativeGroupM_ += Get<MNK_M>(problemShape_);
    deterSync_.curGroupM = cumulativeGroupM_;
}

// 最终 FinalFlush 保留不变
if (isDeter_) {
    SyncAll<false>();
    if ASCEND_IS_AIV { FinalFlush(params); }
    deterSync_.windowStartM = deterSync_.curGroupM;
    SyncAll<false>();
}
```

### 5.2 删除理由

1. **覆盖性**：修改 1 的 tile 级检查覆盖所有溢出场景
2. **正确性 bug**：当 `groupM > windowSize` 时，该检查产生空 flush，flush 后 `lowBoundM` 不变，`ProcessSingleGroup` 仍会溢出
3. **减少开销**：省去每 group 一次的比较 + 可能的空 flush（2×SyncAll）

### 5.3 功能等价验证

场景：多个 group，总 M 超 windowSize

```
cumulativeGroupM_=4000, windowStartM=0, windowSize=6144, lowBoundM=6144
Group 1: M=3000
```

| | 旧（group 级检查） | 新（只有 tile 级检查） |
|---|---|---|
| group 前检查 | 4000+3000=7000>6144 → FLUSH(4000行) | -- |
| ProcessSingleGroup | T0: tileEndAbsM=4128≤12144 ✓ | T0: tileEndAbsM=4128≤6144 ✓ |
| | T15: tileEndAbsM=6048≤12144 ✓ | T15: tileEndAbsM=6048≤6144 ✓ |
| | T16: tileEndAbsM=6176≤12144 ✓ | **T16: tileEndAbsM=6176>6144 → FLUSH(6048行)** |
| | ... | lowBoundM=6048+6144=12192, T16: 6176≤12192 ✓ |

旧方案 2 次 flush（4000行 + 剩余），新方案也是 2 次 flush（6048行 + 剩余）。flush 行数不同但**最终输出正确**——因为被 flush 的行都是已完成且不会再被修改的。

---

## 6. Shape 偏移模拟

**模拟参数**：
```
N = 4096, sizeof(CType) = 4 (float)
deterWorkspaceSize = 96 MB = 100,663,296 B
windowSize = 100,663,296 / (4096 × 4) = 6,144 rows
baseM = 128 (per-tile M)
```

### 6.1 场景 A：正常（总 M < windowSize）

```
Group 0: M=1000, Group 1: M=2000, Group 2: M=500
总 M = 3500 < 6144
```

| 迭代 | cumM | tileMO | tileM | tileEndAbsM | lowBoundM | Flush? |
|------|------|--------|-------|-------------|-----------|--------|
| G0 T0 | 0 | 0 | 128 | 128 | 6144 | No |
| G0 T7 | 0 | 896 | 128 | 1024 | 6144 | No |
| G1 T0 | 1000 | 0 | 128 | 1128 | 6144 | No |
| G1 T15 | 1000 | 1920 | 128 | 3048 | 6144 | No |
| G2 T3 | 3000 | 384 | 128 | 3512 | 6144 | No |

→ 0 次中间 flush，1 次最终 flush ✓

### 6.2 场景 B：多 group 累积超 window

```
Group 0: M=4000, Group 1: M=3000
```

| 迭代 | cumM | tileMO | tileM | tileEndAbsM | lowBoundM | Flush? | wsRow首行 |
|------|------|--------|-------|-------------|-----------|--------|-----------|
| G0 T0 | 0 | 0 | 128 | 128 | 6144 | No | 0 |
| G0 T31 | 0 | 3968 | 128 | 4096 | 6144 | No | 3968 |
| G1 T0 | 4000 | 0 | 128 | 4128 | 6144 | No | 4000 |
| G1 T15 | 4000 | 1920 | 128 | 6048 | 6144 | No | 5920 |
| **G1 T16** | 4000 | 2048 | 128 | **6176** | 6144 | **FLUSH** | — |

FLUSH: curGroupM=6048, totalM=6048, windowStartM=6048, lowBoundM=12192

| G1 T16 继续 | 4000 | 2048 | 128 | 6176 | 12192 | No | 0 |
| G1 T17 | 4000 | 2176 | 128 | 6304 | 12192 | No | 128 |
| G1 T23 | 4000 | 2944 | 128 | 7072 | 12192 | No | 896 |

最终 flush: totalM = 7000 - 6048 = 952 行 ✓

### 6.3 场景 C：单 group M > windowSize（溢出修复验证）

```
Group 0: M=15000
```

| 迭代 | cumM | tileMO | tileM | tileEndAbsM | lowBoundM | Flush? |
|------|------|--------|-------|-------------|-----------|--------|
| T0 | 0 | 0 | 128 | 128 | 6144 | No |
| T47 | 0 | 6016 | 128 | 6144 | 6144 | No (=) |
| **T48** | 0 | 6144 | 128 | **6272** | 6144 | **FLUSH 1** |

FLUSH 1: curGroupM=6144, totalM=6144, windowStartM=6144, lowBoundM=12288

| T48 继续 | 0 | 6144 | 128 | 6272 | 12288 | No | 0 |
| T95 | 0 | 12160 | 128 | 12288 | 12288 | No (=) | 6016 |
| **T96** | 0 | 12288 | 128 | **12416** | 12288 | **FLUSH 2** |

FLUSH 2: curGroupM=12288, totalM=6144, windowStartM=12288, lowBoundM=18432

| T96 继续 | 0 | 12288 | 128 | 12416 | 18432 | No | 0 |
| T117 | 0 | 14976 | 32(tail) | 15008 | 18432 | No | 2688 |

最终 flush: totalM = 15000 - 12288 = 2712 行 ✓

→ 原代码在此场景 T48 时溢出；修复后安全

### 6.4 场景 D：多 group + 单 group M > windowSize

```
Group 0: M=2000, Group 1: M=12000, Group 2: M=1000
```

| 迭代 | cumM | tileMO | tileM | tileEndAbsM | lowBoundM | Flush? |
|------|------|--------|-------|-------------|-----------|--------|
| G0 all | ≤2000 | — | — | max=2000 | 6144 | 无 |
| G1 T0 | 2000 | 0 | 128 | 2128 | 6144 | No |
| G1 T31 | 2000 | 3968 | 128 | 6096 | 6144 | No |
| **G1 T32** | 2000 | 4096 | 128 | **6224** | 6144 | **FLUSH 1** |

FLUSH 1: curGroupM=6096, totalM=6096, windowStartM=6096, lowBoundM=12240

| G1 T32 继续 | 2000 | 4096 | 128 | 6224 | 12240 | No |
| G1 T79 | 2000 | 10112 | 128 | 12240 | 12240 | No (=) |
| **G1 T80** | 2000 | 10240 | 128 | **12368** | 12240 | **FLUSH 2** |

FLUSH 2: curGroupM=12240, totalM=6144, windowStartM=12240, lowBoundM=18384

| G1 T93 (tail) | 2000 | 11904 | 128(tail) | 14032 | 18384 | No |
| G2 T0 | 14000 | 0 | 128 | 14128 | 18384 | No |

最终 flush: totalM = 15000 - 12240 = 2760 行 ✓

→ 2 次中间 flush + 1 次最终 flush，无溢出

---

## 7. 修改风险与回滚

| 风险 | 等级 | 缓解 |
|------|------|------|
| tile 级检查引入额外 branch 开销 | 低 | 每次 tile 一次 uint64 加法 + 比较，相对 matmul 开销可忽略 |
| `tileMOffset` 依赖 scheduler 的顺序保证 | 低 | ASWT Pro Scheduler 保证 tile 按 M 递增顺序产出 |
| `cumulativeGroupM_` 在 group 中间不更新 | 无 | 设计意图，flush 时用 `cumulativeGroupM_ + tileMOffset` 计算 curGroupM |

**回滚**：恢复 `kernel_gmm_finalize_routing_pertoken_dequant_deter.h` 到修改前的版本即可。

---

## 8. 测试建议

1. **单 group M > windowSize 极端 case**：`N=4096, group=1, M=20000`，验证不溢出、输出正确
2. **边界 case**：`groupM = windowSize`, `groupM = windowSize - 1`, `groupM = windowSize + 1`
3. **多 group 混合**：小 group + 大 group + 小 group，验证 flush 在正确边界触发
4. **与非确定性版本对照**：同输入、确定性 flag=0 vs flag=1，输出值一致
5. **内存越界检测**：用 CANN 模拟器的 memory checker 确认无越界写入
