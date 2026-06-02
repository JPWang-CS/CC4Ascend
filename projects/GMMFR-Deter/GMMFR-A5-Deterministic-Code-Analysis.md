# GMMFR A5 确定性分支 — 逐句代码详解

> 两个新增文件，拷贝自原版后修改。仅讲解确定性新增/修改的逻辑代码，跳过重命名、注释格式差异。

---

## 文件 1: `kernel_gmm_finalize_routing_pertoken_dequant_deter.h`

Kernel 调度层，负责遍历所有 group 的所有 tile、管理 workspace 窗口、触发 flush。

---

### 1.1 新增结构体 `DeterSyncConfig`（:64-70）

```cpp
struct DeterSyncConfig {
    uint64_t curM = 0;           // 未使用，保留
    uint64_t curGroupM = 0;      // 当前 flush 边界的全局 M 行号
    uint64_t lowBoundM = 0;      // 窗口上界：累计 M 超过此值必须 flush
    uint64_t windowSize = 0;     // 窗口容量，单位是行数
    uint64_t windowStartM = 0;   // 当前窗口起始的全局 M 行号
};
```

**字段关系图**：
```
全局 M 轴:
 0        windowStartM        curGroupM         lowBoundM
 │◄─────────────────────────►│                    │
 │    workspace 已用行数       │◄──────────────────►│
 │                            │ windowSize 行空间   │
 │          ← workspace 存储范围 →                  │
```

- `windowSize` = workspace 能存多少行，计算公式：`deterWorkspaceSize / (N * sizeof(输出类型))`
- `windowStartM` = 当前窗口起始行
- `lowBoundM` = `windowStartM + windowSize`，超过则 flush
- `curGroupM` = flush 的切割点（上一次 flush 时的累计 M）

---

### 1.2 新增成员变量（:137-138）

```cpp
    DeterSyncConfig deterSync_;
    uint64_t cumulativeGroupM_ = 0;   // 所有已处理 group 的 M 行总数
```

- `cumulativeGroupM_` 随每个 group 处理完递增，用于判断窗口是否要溢出
- `deterSync_` 维护当前窗口的状态

---

### 1.3 GMMTiling 新增两个字段（:147-148）

```cpp
    struct GMMTiling {
        // ... 原有：groupNum, groupListType, baseM, baseN, baseK, hasBias ...
        uint32_t deterWorkspaceSize;   // 新增：workspace 总字节数（Host Tiling 传入）
        uint32_t coreNum;              // 新增：使用的 AI Core 数量
    };
```

构造函数对应新增（:153-156）：
```cpp
    GMMTiling(uint32_t groupNum_, ..., uint32_t deterWsSize_, uint32_t coreNum_)
        : ..., hasBias(hasBias_), deterWorkspaceSize(deterWsSize_), coreNum(coreNum_) {}
```

---

### 1.4 `operator()` — 主入口新增代码

#### 窗口容量初始化（:374-377）

```cpp
        deterSync_.windowSize =
            params.gmmParams.deterWorkspaceSize / (params.gmmParams.matmulTiling->N * sizeof(CType));
        deterSync_.lowBoundM = deterSync_.windowSize;
```

- `N * sizeof(CType)` = 每行输出占多少字节
- `windowSize` = workspace 能容纳的行数
- 初始时 `lowBoundM = windowSize`（从第 0 行开始）

**举例**（N=4096, CType=bfloat16_t(2B), workspace=93952409B）：
```
每行 = 4096 × 2 = 8192 字节
windowSize = 93952409 / 8192 = 11468 行
lowBoundM = 11468
```

#### Epilogue Init 多传 pipe（:381）

```cpp
            epilogueDequantOp_.Init(params.epilogueParams, GetTPipePtr());
```

原始版本只传 `params`。确定性版本多传 `GetTPipePtr()`，因为 Epilogue 内部需要用 pipe 来 `InitBuffer(deterDmaQue_)`。

#### 组循环内：窗口溢出检测 + flush（:395-404）

```cpp
            if ((cumulativeGroupM_ + (uint64_t)Get<MNK_M>(problemShape_)) > deterSync_.lowBoundM) {
```

判断条件：`已累计行数 + 当前组的行数 > 窗口上界` → 当前组放不进窗口了。

```cpp
                deterSync_.curGroupM = cumulativeGroupM_;
                // 记录 flush 点：当前累计 M

                SyncAll<false>();
                // 全核同步：确保之前所有 tile 的 Epilogue 写 workspace 完成

                if ASCEND_IS_AIV {
                    FRDeterministic(params);
                    // AIV 执行 flush：workspace → y
                }

                deterSync_.windowStartM = deterSync_.curGroupM;
                // 新窗口从此处开始

                deterSync_.lowBoundM = deterSync_.curGroupM + deterSync_.windowSize;
                // 新窗口上界

                SyncAll<false>();
                // 全核同步：flush 完成后再继续
            }
```

**举例**（windowSize=11468，3个 group 的 M 分别为 5000、4000、3000）：
```
group 0 (M=5000): cumulativeGroupM_=0,  0+5000=5000 ≤ 11468  → 不 flush
group 1 (M=4000): cumulativeGroupM_=5000, 5000+4000=9000 ≤ 11468 → 不 flush
group 2 (M=3000): cumulativeGroupM_=9000, 9000+3000=12000 > 11468 → ★ 触发 flush
  curGroupM = 9000
  FRDeterministic 处理 workspace[0..9000) → y
  windowStartM = 9000, lowBoundM = 9000 + 11468 = 20468
```

#### 组处理完后累计 M（:406-407）

```cpp
            cumulativeGroupM_ += Get<MNK_M>(problemShape_);
            deterSync_.curGroupM = cumulativeGroupM_;
```

#### 最终 flush（:410-415）

```cpp
        SyncAll<false>();
        if ASCEND_IS_AIV {
            FRDeterministic(params);           // 处理窗口中剩余的行
        }
        SyncAll<false>();
```

所有 group 遍历完后，窗口内可能还有未 flush 的数据，做最后一次。

---

### 1.5 `ProcessSingleGroup()` — 新增代码

#### 分核方式变更（:296）

```cpp
        while (bs.GetTileIdxRowMajor(tileIdx)) {
```

原始用 `GetTileIdx`。确定性改用 `GetTileIdxRowMajor`（行优先遍历），使同一 M 行的 tile 落在同一核，提高 workspace 写入的局部性。

#### tile 级窗口溢出检测（:301-315）

```cpp
            int64_t y = Get<DETER_IDX_C_OFFSETS>(blockOffset_);
            int64_t tileMOffset = y / n;               // tile 在当前 group 内的 M 行偏移
            int64_t singleM = Get<MNK_M>(singleShape); // tile 的 M 行数
            int64_t wsAccum   = static_cast<int64_t>(cumulativeGroupM_ - deterSync_.windowStartM);
            int64_t tileWsEnd = wsAccum + tileMOffset + singleM;
```

**逐句**：

- `y` = tile 在线性化输出中的偏移（元素单位）
- `tileMOffset = y / n` = tile 在当前 group 内从第几行开始
- `singleM` = 这个 tile 有多少行
- `wsAccum` = 窗口内已占用的行数 = `累计M - 窗口起始M`
- `tileWsEnd` = 这个 tile 写完后在窗口内的位置 = `窗口已占用 + tile偏移 + tile大小`

**举例**（cumulativeGroupM_=9000, windowStartM_=0, n=4096）：
```
tile A: y=8192000, tileMOffset=2000, singleM=256
  wsAccum = 9000 - 0 = 9000
  tileWsEnd = 9000 + 2000 + 256 = 11256 ≤ 11468 → 不 flush

tile B: y=9240576, tileMOffset=2256, singleM=256
  tileWsEnd = 9000 + 2256 + 256 = 11512 > 11468 → ★ tile 级 flush
```

```cpp
            if (tileWsEnd > static_cast<int64_t>(deterSync_.windowSize)) {
                deterSync_.curGroupM = cumulativeGroupM_ + static_cast<uint64_t>(tileMOffset);
                // flush 点 = 累计M + tile偏移（tile 之前的行）

                SyncAll<false>();
                if ASCEND_IS_AIV {
                    FRDeterministic(params);
                }
                deterSync_.windowStartM = deterSync_.curGroupM;
                deterSync_.lowBoundM  = deterSync_.curGroupM + deterSync_.windowSize;
                SyncAll<false>();
            }
```

逻辑与组级 flush 相同，只是触发粒度更细——在 tile 级别。

#### Epilogue 新增 `SetWorkspaceGroupOffset` 调用（:330-331）

```cpp
                epilogueDequantOp_.SetWorkspaceGroupOffset(cumulativeGroupM_ - deterSync_.windowStartM);
```

告诉 Epilogue：当前窗口内已累计了多少行。Epilogue 用这个值计算写入 workspace 的起始行号。

**举例**：
```
cumulativeGroupM_ = 9000, windowStartM_ = 0
→ groupAccumM_ = 9000 - 0 = 9000
→ Epilogue 从 workspace 第 9000 行开始写
```

---

### 1.6 `FRDeterministic()` — 全新函数（:346-360）逐句

```cpp
    __aicore__ inline void FRDeterministic(const Params &params)
    {
        uint64_t totalM = deterSync_.curGroupM - deterSync_.windowStartM;
        if (totalM == 0) {
            return;
        }
```

`totalM` = 本次 flush 需要处理的行数。为 0 直接返回。

```cpp
        uint64_t coreNumVec = params.gmmParams.coreNum * GetTaskRation();
```

- `GetTaskRation()` = 2（AIC:AIV = 1:2 配比）
- `coreNumVec` = Vector 核总数，如 36 × 2 = 72

```cpp
        uint64_t n = params.gmmParams.matmulTiling->N;
```

输出列数。

```cpp
        for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
            auto outRow = epilogueDequantOp_.GetRowIndex(deterSync_.windowStartM + mOffset);
```

- `mOffset` 遍历窗口内每一行
- `GetRowIndex(windowStartM + mOffset)` 从 `rowIndexGlobal_` 读取第 `windowStartM + mOffset` 个元素
- 返回值 `outRow` = 该行在最终输出 y 中的目标行号

```cpp
            if (outRow % coreNumVec != GetBlockIdx())
                continue;
```

**按行号取模分核**：每个 Vector 核只处理 `outRow % 72 == 自身blockIdx` 的行。避免多核写同一行 y。

```cpp
            epilogueDequantOp_.DeterministicFlushRow(mOffset, outRow, n);
```

执行单行 flush：从 workspace 读 → 原子加到 y。

**完整举例**（windowStartM=0, curGroupM=9000, coreNumVec=72, 核 5 的视角）：
```
mOffset=0:   outRow = rowIndex[0] = 5,    5 % 72 = 5  == 核5 → 处理
mOffset=1:   outRow = rowIndex[1] = 77,   77 % 72 = 5 == 核5 → 处理
mOffset=2:   outRow = rowIndex[2] = 100,  100 % 72 = 28 ≠ 5 → 跳过
...
mOffset=8999: outRow = rowIndex[8999] = 341, 341 % 72 = 53 ≠ 5 → 跳过

核5 总共处理约 9000/72 ≈ 125 行
```

---
---

## 文件 2: `block_epilogue_dequant_finalize_routing_deter.h`

Epilogue 计算层，负责反量化 + Logit 乘 + 写 workspace + flush。

---

### 2.1 新增常量（:31-41）

```cpp
constexpr uint32_t MAX_SINGLE_MNSD = 128 * 256U;        // 32768
constexpr uint32_t HALF_DB_MAX_SINGLE_MNSD = 32 * 256U; // 8192
constexpr uint32_t BLOCKS_BYTESD = 256U;
constexpr uint64_t MAX_OUTPUT_M_UBSD = 32UL;
constexpr uint64_t BLOCK_ELEMENTS_FP32SD = 8UL;
constexpr uint32_t DETER_UB_SIZED = 12 * 1024U;         // 12KB — L0C 输入 buffer + DMA 队列 buffer
```

`DETER_UB_SIZED` 是确定性新增的，其余值与原版相同只是后缀改为 `D`。

---

### 2.2 Params 新增字段（:83-84）

```cpp
    struct Params {
        // ... 原有：yGMAddr, x2ScaleGmAddr, x1ScaleGmAddr, biasGmAddr, logitGmAddr, rowIndexGmAddr, baseM, baseN
        GM_ADDR workspaceGM{nullptr};        // 新增：workspace 的 GM 地址
        uint32_t deterWorkspaceSize = 0;     // 新增：workspace 字节数
    };
```

---

### 2.3 新增成员变量（:137, 152, 165）

```cpp
    AscendC::GlobalTensor<DataTypeOut> mmQuantOutGm_;     // workspace 的 GlobalTensor
    AscendC::TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterDmaQue_;  // flush 用的 DMA 队列（12KB）
    uint64_t groupAccumM_ = 0;                             // 窗口内已累计行数
```

- `mmQuantOutGm_`：绑定到 workspace 地址，反量化结果写入此 GlobalTensor
- `deterDmaQue_`：flush 时从 workspace 读数据到 UB 的 DMA 中转队列
- `groupAccumM_`：由 `SetWorkspaceGroupOffset` 设置

---

### 2.4 新增三个方法声明（:104-106）

```cpp
    void DeterministicFlushRow(uint64_t mOffset, uint64_t outRow, uint64_t nSize);
    uint64_t GetRowIndex(uint64_t globalOffset);
    void SetWorkspaceGroupOffset(uint64_t groupAccumM);
```

---

### 2.5 `Init()` 新增代码（:220-223）逐句

```cpp
    pipe->InitBuffer(deterDmaQue_, 1, DETER_UB_SIZED);
```

- 初始化 DMA 队列，1 个 buffer，大小 12KB
- 用于 `DeterministicFlushRow` 中一次搬运一行数据

```cpp
    uint32_t wsElemCount = params_->deterWorkspaceSize / sizeof(DataTypeOut);
    mmQuantOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut *>(params_->workspaceGM), wsElemCount);
```

- `wsElemCount = 93952409 / 2 = 46976204`（BF16 场景）
- 将 workspace 地址绑定为 `mmQuantOutGm_`，后续所有 workspace 读写都通过此 GlobalTensor

```cpp
    rowIndexGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeRowIndex *>(params_->rowIndexGmAddr));
```

绑定行索引表。`FRDeterministic` 中通过 `rowIndex[i]` 查找该行在 y 中的目标行号。

---

### 2.6 `SetWorkspaceGroupOffset()`（:517-519）逐句

```cpp
    __aicore__ inline void SetWorkspaceGroupOffset(uint64_t groupAccumM)
    {
        groupAccumM_ = groupAccumM;
    }
```

由 Kernel 调度层调用，传入 `cumulativeGroupM_ - windowStartM`。保存后供 `LinearWriteToWorkspace` 使用。

---

### 2.7 `GetRowIndex()`（:509-513）逐句

```cpp
    __aicore__ inline uint64_t GetRowIndex(uint64_t globalOffset)
    {
        return static_cast<uint64_t>(rowIndexGlobal_.GetValue(globalOffset));
    }
```

从 GM 的 `rowIndexGlobal_` 读取第 `globalOffset` 个元素。返回该行在 y 中的目标行号。

由 `FRDeterministic` 调用：
```
outRow = GetRowIndex(windowStartM + mOffset)
```

---

### 2.8 `LinearWriteToWorkspace()`（:273-282）— 核心写入，逐句

原版此函数叫 `VectorAtomicProcess`，功能是原子加写 y。确定性版本改为**线性写 workspace**。

```cpp
    __aicore__ inline void LinearWriteToWorkspace(
        uint32_t curBaseN,            // N 列数
        uint32_t curVecBaseM,         // 当前行数
        uint64_t offsetM,             // 全局 M 偏移（传入的是 logitOffset + i*32）
        uint64_t yOffset,             // N 偏移（传入的是 blockCoord 的 Y 分量）
        LocalTensor<DataTypeOut> &yLocal)  // UB 上的输出数据（反量化+Logit乘后的结果）
    {
        DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
        // 每次 DataCopyPad 搬 1 行，宽度 = curBaseN × sizeof(BF16)
```

```cpp
        for (uint32_t i = 0; i < curVecBaseM; i++) {
            uint64_t wsRow = groupAccumM_ + offsetM + i;
```

**偏移计算第一步**：workspace 逻辑行号
- `groupAccumM_` = 窗口内已累计的行数（由 `SetWorkspaceGroupOffset` 设置）
- `offsetM` = 当前这批数据在全局中的 M 偏移
- `i` = 本批数据内第几行

```cpp
            uint64_t wsIdx = wsRow * n_ + yOffset;
```

**偏移计算第二步**：workspace 线性索引（元素单位）
- `n_` = 输出列数
- `yOffset` = N 方向偏移
- `wsIdx` = 该元素在 workspace GlobalTensor 中的下标

```cpp
            DataCopyPad(mmQuantOutGm_[wsIdx], yLocal[i * alignN_], paramsOut);
        }
    }
```

从 UB `yLocal[i * alignN_]` 搬 `curBaseN` 个元素到 GM `mmQuantOutGm_[wsIdx]`。

**完整偏移举例**（groupAccumM_=5000, offsetM=100, n_=4096, yOffset=0, curVecBaseM=32）：

```
i=0:  wsRow = 5000 + 100 + 0  = 5100
      wsIdx = 5100 × 4096 + 0 = 20889600
      DataCopyPad mmQuantOutGm_[20889600] ← yLocal[0]

i=1:  wsRow = 5101
      wsIdx = 5101 × 4096 = 20893696
      DataCopyPad mmQuantOutGm_[20893696] ← yLocal[alignN_]

...

i=31: wsRow = 5131
      wsIdx = 5131 × 4096 = 21016576
      DataCopyPad mmQuantOutGm_[21016576] ← yLocal[31 × alignN_]
```

**workspace 内存布局**（行优先，每行 N 个元素）：
```
mmQuantOutGm_:
  行 0:     [elem 0, 1, ..., N-1]         wsIdx 范围 [0, N)
  行 1:     [elem N, N+1, ..., 2N-1]       wsIdx 范围 [N, 2N)
  ...
  行 5100:  [elem 20889600, ...]            ← 本次写入起始
  行 5131:  [elem 21016576, ...]            ← 本次写入结束
  ...
  行 windowSize-1:                          ← workspace 上限
```

---

### 2.9 `DeterministicFlushRow()`（:523-541）— 逐行 flush，逐句

```cpp
    __aicore__ inline void DeterministicFlushRow(
        uint64_t mOffset,     // workspace 内的行偏移（相对窗口起始）
        uint64_t outRow,      // y 中的目标行号（由 GetRowIndex 得到）
        uint64_t nSize)       // N 列数
    {
        constexpr uint64_t baseN = DETER_UB_SIZED / sizeof(DataTypeOut);
        // baseN = 12288 / 2 = 6144 — 每次 DMA 最多搬 6144 个 BF16
```

```cpp
        for (uint64_t nOff = 0; nOff < nSize; nOff += baseN) {
            uint64_t curN = (nOff + baseN > nSize) ? (nSize - nOff) : baseN;
            // 最后一段可能不足 6144
```

按 `baseN` 分段搬运。一般 N=4096 < 6144，所以只循环一次。

```cpp
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curN * sizeof(DataTypeOut)), 0, 0, 0};
            auto local = deterDmaQue_.AllocTensor<DataTypeOut>();
```

从 DMA 队列分配 12KB UB buffer。

```cpp
            DataCopyPad(local, mmQuantOutGm_[mOffset * nSize + nOff], copyParams,
                        DataCopyPadExtParams<DataTypeOut>{});
```

**读 workspace → UB**：
- 源地址：`mmQuantOutGm_[mOffset * nSize + nOff]`
- 偏移公式：`mOffset × N + nOff`
  - `mOffset` = 行号（相对于窗口起始）
  - `nOff` = 该行内的 N 偏移

```cpp
            deterDmaQue_.EnQue(local);
            auto readLocal = deterDmaQue_.DeQue<DataTypeOut>();
```

队列入队/出队确保 DMA 传输完成。

```cpp
            SetAtomicAdd<float>();
            DataCopyPad(yGlobal_[outRow * nSize + nOff], readLocal, copyParams);
            SetAtomicNone();
```

**原子加到 y**：
- 目标地址：`yGlobal_[outRow * nSize + nOff]`
  - `outRow` = 目标行号（来自 rowIndex）
  - `nOff` = 该行内的 N 偏移
- `SetAtomicAdd` 开启浮点原子加：`y[outRow*N+nOff] += readLocal`
- `SetAtomicNone` 关闭

**为什么还要原子加？** 不同行的 rowIndex 可能映射到同一个 `outRow`（不是一一映射），多个核可能 flush 到同一目标行。原子加保证并发安全。但相比原始路径的 tile 级竞争，确定性路径只在整行完成后才写一次 y，竞争粒度大幅降低。

```cpp
            deterDmaQue_.FreeTensor(readLocal);
        }
    }
```

**完整 flush 举例**（mOffset=100, outRow=50, nSize=4096）：
```
baseN=6144 > nSize=4096 → 只循环一次, nOff=0, curN=4096

1. 分配 12KB UB: local
2. 读 workspace: mmQuantOutGm_[100 × 4096 + 0] = mmQuantOutGm_[409600] → local
3. 原子加到 y: yGlobal_[50 × 4096 + 0] = yGlobal_[204800] += local
```

---

### 2.10 `operator()` 中的差异

在 `operator()` 的尾部，输出逻辑从原始的 `VectorAtomicProcess` 改为 `LinearWriteToWorkspace`：

```cpp
        // 原始
        VectorAtomicProcess(singleN_, repeatTimesLine_, logitOffset + i * MAX_OUTPUT_M_UBS, yOffset, yLocal);
        // 内部：SetAtomicAdd → DataCopyPad yGlobal_[rowIndex[m] * N + n] → SetAtomicNone

        // 确定性
        LinearWriteToWorkspace(singleN_, repeatTimesLine_, logitOffset + i * MAX_OUTPUT_M_UBSD, yOffset, yLocal);
        // 内部：DataCopyPad mmQuantOutGm_[(groupAccumM_ + offsetM + i) * N + yOffset]
```

即：**原版直接写 y（原子加），确定性写 workspace（线性写）**。其余反量化、Logit 乘、ping-pong 等逻辑完全相同。

---

## 附：完整确定性数据流

```
                         ┌─── 每个 tile ───┐
                         │                  │
AIC:  x × w → L0C(INT32) ─── L0C2UB ──→ UB l0cOutUb_(INT32)
                                              │
AIV:  Cast INT32→FP32                         │
      × w_scale (UB)                          │
      × x_scale (UB, broadcast)               │
      + bias    (UB)                          │
      × logit   (UB)                          │
      = outUb (BF16)                          │
                         │                    │
                         ▼                    │
      LinearWriteToWorkspace():               │
        for each row i:                       │
          wsRow = groupAccumM_ + offsetM + i  │
          wsIdx = wsRow × N + yOffset         │
          mmQuantOutGm_[wsIdx] ← yLocal[i]   │
                                              │
      ──────── workspace 中 ────────          │
      行0: [N个BF16]                          │
      行1: [N个BF16]                          │
      ...                                     │
      行wsRow: [N个BF16]  ← 本次写入          │
                         │                    │
                         │  窗口满 or 所有 group 结束
                         ▼                    │
      FRDeterministic():                      │
        for mOffset in [0, totalM):           │
          outRow = rowIndex[windowStartM + mOffset]
          if outRow % coreNumVec != blockIdx: │
              skip                            │
          DeterministicFlushRow(mOffset, outRow, N):
            workspace[mOffset×N+nOff] → UB local
            SetAtomicAdd()
            y[outRow×N+nOff] += local
            SetAtomicNone()
```
