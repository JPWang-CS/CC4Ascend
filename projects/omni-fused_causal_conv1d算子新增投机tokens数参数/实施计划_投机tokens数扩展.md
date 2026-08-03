# fused_causal_conv1d 投机 tokens 数扩展 — 实施计划

> **范围仓**：`D:\Desktop\Code\omni-ops`（行号已 trace 实际代码）
> **绝对需求**：`projects/omni-fused_causal_conv1d算子新增投机tokens数参数/fused_causal_conv1d算子新增投机tokens数参数需求分析与设计说明书 - A3.md`
> **本文档定位**：实施叙事 + 全链路 + 代码走向模拟。算子原理/MTP 语义/参数关系链详讲在 `完整技术文档.md`，本文只引用。
> **日期**：2026-08-03
> **证据等级**：除标注 STRONG（已上板/已编译）外，所有结论基于源码静态 trace，属 WEAK。

---

## 一、实施总览

### 1.1 目标（一句话）

放宽 MTP 投机 token 上限 m：7→16（stateLen 9→18），并新增 `maxDraftTokens`（默认 7，[0,16]）约束 UB 上限，解耦 m 与 ACLGraph capture。

### 1.2 两条主线

| 主线 | 一句话 | 改动面 |
|---|---|---|
| **A** | 放宽 m 上限：`MAX_M=7→16` / `MAX_DECODE_LEN=8→17` | tiling.h 常量 + UT 阴性阈值 + UB 预算自适应（kernel 不改） |
| **B** | 新增 `maxDraftTokens`：默认 7，[0,16]，UB 按 `max(stateLen, maxDraftTokens+W-1)` 分配 | aclnn 签名 + l0op OP_ATTR 末尾追加 + torch binding + tiling 防御读 |

### 1.3 分层依赖图

`maxDraftTokens` 穿层走 **OP_ATTR 末尾追加**（不进 OpDef）：aclnn 收参 → l0op OP_ATTR 末尾追加 → tiling 防御读 → 放大 stateLen_ → set_stateLen 下发 kernel。kernel 零改动（读 effectiveStateLen）。

```
[签名层]  aclnn(A1-A3) ──→ torch binding(P1-P4) ──→ torch schema(P5)
              │
              └─→ l0op OP_ATTR 末尾追加 maxDraftTokens（index 8）

[tiling 层]  tiling.h 常量(H1-H2, H4)
             tiling 防御读 GetAttrs()->GetAttrPointer<int64_t>(IDX)(T1) ──→ 放大 stateLen_(C5 落点)
                                                                   │
                                                                   └─→ set_stateLen(T3) → TilingData(T4)

[验证层]  UT 阴性阈值(U1-U2) ← H1/H2
          UT 编译(U4) / atk json+executor(G1-G2) ← A1-A3 / P1-P5 全完成
          项目 golden(G3) ← 语义稳定后

[文档层]  M1-M3 ← 上述全部稳定后
```

### 1.4 阶段划分（详见 §四）

| 阶段 | 目标 | 检查点 |
|---|---|---|
| 1 | 签名层全栈通（A1-A3 / P1-P5） | 全仓编译通过 |
| 2 | tiling 层（H1-H2/H4 / T1-T4 / T6 / C5 落点） | UT 阳性 m∈{8,12,16} 通过、阴性 m=17 报错 |
| 3 | kernel 不溢出验证（不改代码） | 上板跑 D=16384 × m=16，msprof 看 UB |
| 4 | golden + atk 扩展（G1-G3） | 项目 golden N/N PASS、atk 全量回归 |
| 5 | 文档（M1-M3） | doc-sync 全仓扫无残留 |

---

## 二、全链路分析

### 2.1 完整调用链路图

```
Python:
  torch_npu.npu_ai_infra_fused_causal_conv1d(x, weight, conv_states, *, ..., maxDraftTokens=16)
        │  (kwargs 解析 → binding 函数)
        ▼
Torch binding:
  npu_ai_infra_fused_causal_conv1d_npu(...)        [npu_...cpp:19]
   ├─ value_or(7) 注入 max_draft_tokens_const
   └─ EXEC_NPU_CMD_V1(aclnnAiInfraFusedCausalConv1d, ..., max_draft_tokens_const, y)
        │  (PyTorch dispatch → aclnn C 接口)
        ▼
aclnn GetWorkspaceSize (host):
  aclnnAiInfraFusedCausalConv1dGetWorkspaceSize(...)  [aclnn_...cpp:136]
   ├─ L2_DFX_PHASE_1(...)                              [:145]   只跟踪 tensor，不跟踪 scalar
   └─ AiInfraFusedCausalConv1dCommonProcess(...)       [:29]    CreateView + Contiguous
        └─ AiInfraFusedCausalConv1dl0op::AiInfraFusedCausalConv1d(...)  [:105]
             │  (op_executor 收集 op；l0op OP_ATTR 末尾追加 maxDraftTokens，仿 matmul batchInvariant)
             ▼
l0op 注册:
  AiInfraFusedCausalConv1d(...)                       [ai_infra_...cpp:24]
   ├─ L0_DFX(...)                                      [:48]    只 tensor
   └─ ADD_TO_LAUNCHER_LIST_AICORE(                    [:52]
         OP_INPUT(...12 个...),
         OP_OUTPUT(convStatesRef, yOut))
        │  (executor 落 op → GE 下发 → tiling 期)
        ▼
Tiling (host, 编译期 + 运行期):
  AiInfraFusedCausalConv1dTiling::DoTiling(...)
   ├─ ParseAndCheck()                                 [tiling.cpp:644]
   │   ├─ GetNpuInfo() / GetOpName()
   │   ├─ GetAttrInfo()                               [:73]
   │   │   └─ 防御读 maxDraftTokens（GetAttrNum + null 检查）   ← OP_ATTR 超集
   │   ├─ CheckXValid() / Check*Valid()               [:135-497]
   │   ├─ GetTilingKey()                              [:499]    ← 控制流：选 3 模板
   │   └─ GenerateInfo()                              [:516]    ← UB 预算自适应 + C5 落点放大 stateLen_
   ├─ set_* 系列                                      [:682-718]  ← 写 TilingData
   └─ SaveToBuffer → RawTilingData 下发               [:719]
        │  (tiling data 序列化 → kernel 参数)
        ▼
Kernel (device, AICore):
  按 TilingKey 分发：
   ├─ TilingKey=0 → CUTBSD  (prefill, dim 大 / cuSeq 小)
   ├─ TilingKey=1 → CUTBS   (prefill, dim 小 / cuSeq 大)
   └─ TilingKey=2 → Update  (decode, MTP 走这里)
  InitBuffer 读取 tilingData_ 的 stateLen（=effectiveStateLen）/ baseDim → 分配 UB
  Process() 主循环按 seqLen (= m+1) 逐 token 卷积
        │
        ▼
Golden 验证:
  atk `causal_conv1d_golden` (executor_ai_infra_..._continue.py:73) 纯 PyTorch 参考实现
  项目 golden 套 bench_metrics 格式输出 PASS/FAIL + 全精度指标
```

### 2.2 每层职责 / 契约 / 控制流

| 层 | 职责（干什么 / 不干什么） | 接口契约（上下游怎么传） |
|---|---|---|
| **Torch binding** | 解析 Python kwargs，把 `optional` 字段转成 aclnn 需要的 C 类型（值/空指针）；shape 推导（meta）；EXEC_NPU_CMD_V1 分发到 aclnn | Python kwarg（camelCase 外观）↔ 内部 snake_case；`optional<int64_t> value_or(7)` 注入默认值 |
| **aclnn GetWorkspaceSize** | host 入口；非连续 tensor 的 CreateView；Contiguous 化；构造 op_executor；透传 maxDraftTokens 给 l0op；返回 workspace size | C 函数签名（inputs → attrs → yOut → workspaceSize → executor） |
| **l0op** | 把 op 加入 launcher list；OP_ATTR 末尾追加绑定 maxDraftTokens | OP_INPUT / OP_OUTPUT 顺序与现有保持一致；OP_ATTR 末尾追加 maxDraftTokens（index 8，不动原 8 个顺序） |
| **TilingInfoParser** | 读 op 属性（防御读）→ 校验 → 选 TilingKey → 算 UB 预算 → C5 落点放大 stateLen_ → 写 TilingData | `GetAttrs()->GetAttrPointer<int64_t>(IDX)` + GetAttrNum + null 检查 |
| **TilingData** | host↔kernel 序列化结构，kernel 用其字段算 buffer | stateLen 字段复用（写 effectiveStateLen），不新增字段（C5 路径 B） |
| **Kernel** | device 执行：搬数 → 卷积 → 写 cache / y。**不消费 m**，消费 `seqLen_ = m+1` 或 `query_start_loc` 差值 | 不改（buffer 全动态，自动跟随 tilingData 字段） |

### 2.3 runMode / TilingKey 控制流

源 `tiling.cpp:101` / `:499-514`：

```
runMode = (maxQueryLen > MAX_DECODE_LEN || maxQueryLen <= 0) ? 0 : 1   ← 主线 A：MAX_DECODE_LEN 8→17

runMode=0 (prefill):
  if cuSeqLength < coreNum || dimSize >= MAX_DIM_CUTBSD(3152):  TilingKey=0 (CUTBSD)
  else:                                                        TilingKey=1 (CUTBS)

runMode=1 (decode):
  if xDimSize == 2 || xDimSize == 3:                           TilingKey=2 (Update)
```

**主线 A 的关键语义**：MAX_DECODE_LEN=17 保证 m=16（`max_query_len = m+1 = 17`）仍走 decode 路径。若保持 8，m=8..16 会被误判为 prefill。

### 2.4 三个 kernel 模板对 m 的敏感度

| 模板 | TilingKey | m 敏感 buffer | 主线 A 影响 |
|---|---|---|---|
| CUTBSD | 0 | 无（`convStatesQue = 2 × baseDim × dtype`，与 stateLen 无关） | 不受影响 |
| CUTBS | 1 | `cacheQue (双缓冲) + cacheBuf`（与 stateLen 成正比） | UB +27KB（baseDim=512 估算），ubFactor 28→24，循环 +18% |
| Update | 2 | `convStatesQueue_`（与 stateLen 成正比） | maxDim -25.5%，dim 切分数增加 |

源：`op_kernel/ai_infra_fused_causal_conv1d_fn_cutbs.h:82-90` / `op_kernel/ai_infra_fused_causal_conv1d_update.h:98`。

---

## 三、代码走向模拟（核心增值）

### 3.1 场景 A：`maxDraftTokens=16` 新参数穿层（主线 B，OP_ATTR 末尾追加（B'））

从 Python kwarg 一路到 kernel buffer 分配的完整路径。

**A.1 Python kwarg → Torch binding 入口**

```python
torch_npu.npu_ai_infra_fused_causal_conv1d(
    x, weight, conv_states, *, ..., inplace=False, maxDraftTokens=16)
```

Python 端 kwarg 名 `maxDraftTokens`（需求文档 §2.4.1，camelCase 外观）。torch schema 字段名 snake_case `max_draft_tokens`（与现有 schema 一致，见 `ops_def_registration.cpp:63-70`），由 torch 自动映射 kwarg。

**A.2 Torch binding 函数签名加参数**

文件 `torch_ops_extension/.../npu_ai_infra_fused_causal_conv1d.cpp:19-27`：

```cpp
// 现状（已 trace）：
at::Tensor npu_ai_infra_fused_causal_conv1d_npu(
    at::Tensor &x, const at::Tensor &weight, at::Tensor &conv_states,
    ... 11 个 optional tensor ...,
    c10::optional<std::string_view> activation, int64_t pad_slot_id, int64_t run_mode,
    c10::optional<int64_t> max_query_len, int64_t residual_connection, int64_t block_size,
    int64_t conv_mode, bool inplace)
```

**改后**：在 `bool inplace` 之后追加 `c10::optional<int64_t> max_draft_tokens`。

**A.3 默认值注入（value_or(7)）**

参考 `:49` 现有 `max_query_len.value_or(-1)` 模式：

```cpp
int64_t max_draft_tokens_const = max_draft_tokens.value_or(7);  // DEFAULT_MAX_DRAFT_TOKENS
```

**A.4 EXEC_NPU_CMD_V1 透传**

`:52-73` 现有调用按 aclnn 签名顺序传参。在 `inplace` 之后、`y` 之前插入 `max_draft_tokens_const`。

**A.5 aclnn GetWorkspaceSize 签名加参数**

文件 `op_api/aclnn_ai_infra_fused_causal_conv1d.h:21-44` 与 `op_api/aclnn_ai_infra_fused_causal_conv1d.cpp:136-143`。实际代码现状（已 trace `:136-143`）：

```cpp
aclnnStatus aclnnAiInfraFusedCausalConv1dGetWorkspaceSize(
    aclTensor *x, ..., const aclTensor *initialStateIdx,
    int64_t activationMode, int64_t padSlotId, int64_t runMode, int64_t maxQueryLen,
    int64_t residualConnection, int64_t blockSize, int64_t convMode, bool inplace,
    const aclTensor *yOut, uint64_t *workspaceSize, aclOpExecutor **executor);
```

**改后**：在 `bool inplace` 之后、`const aclTensor *yOut` 之前插入 `int64_t maxDraftTokens,`（仿 matmul `batchInvariant` 位置 `aclnn_ai_infra_matmul.cpp:905` cubeMathType, batchInvariant, workspaceSize）。

**A.6 CommonProcess 透传**

`:29-35` 函数签名加 `int64_t maxDraftTokens`；`:105-126` 调用 l0op 处透传。**注意 `:145-158` 的 L2_DFX_PHASE_1 只 DFX_IN tensor、DFX_OUT tensor，不含 scalar**，无需改。

**A.7 l0op OP_ATTR 末尾追加 maxDraftTokens（B' 真实机制，不加 graph 类）**

fused_causal_conv1d 是 aclnn 直调 l0op（无 matmul 那种 MatmulGraphImpl 多态体系），所以 maxDraftTokens 直接在 l0op 的 OP_ATTR 列表末尾追加，不引入 graph setter / graph 类。OP_ATTR 是运行期 op 实参绑定，可超集 OpDef 静态注册（`GetAttrs()` 读的是 OP_ATTR 绑定的，不是 OpDef `.Attr()` 注册的）。

参考 matmul 真实实现（已 trace）：
- `aclnn_ai_infra_matmul.cpp:45`：`OP_ATTR(..., cubeMathType, batchInvariant, ...)`
- `aclnn_ai_infra_matmul.cpp:431/905/931`：aclnn 收 `bool batchInvariant` → 作 l0op 实参传入 `OP_ATTR`（`matmulGraph->SetBatchInvariant` 只是把值写进 opInfo 中间结构体，最终仍经 OP_ATTR 绑定，非旁路）
- `ai_infra_matmul_def.cpp`：无 `batch_invariant` 的 `.Attr()`（不进 OpDef）
- `ai_infra_matmul_base_tiling.cpp:65-66`：`BATCH_INVARIANT_ATTR_NUM=5, BATCH_INVARIANT_ATTR_INDEX=4`
- `ai_infra_matmul_base_tiling.cpp:230-233`：`if (GetAttrs()->GetAttrNum() >= 5) { GetAttrPointer<bool>(4) ... }` 防御读

**本算子改后**（在 l0op 函数 `ai_infra_fused_causal_conv1d.cpp:24-46` 的 OP_ATTR 末尾追加）：

```cpp
// ai_infra_fused_causal_conv1d.cpp 的 l0op 函数内，OP_ATTR 现有 8 个 attr 之后追加：
OP_ATTR(context, maxiNf, maxDraftTokens)   // index 8，不动原 8 个顺序
```

aclnn CommonProcess 把 `maxDraftTokens` 作 l0op 实参传入即可，无需 graph setter / 动态转换。此值经 OP_ATTR 绑定进 op 运行期属性，tiling 期经 `GetAttrs()->GetAttrPointer<int64_t>(8)` 读到。

**A.8 tiling.h 加 INDEX 常量**

`tiling.h:48-55` 已有：

```cpp
constexpr int64_t ACTIVATION_MODE_INDEX = 0;
...
constexpr int64_t INPLACE_INDEX = 7;   // 第 55 行
```

**改后**追加：

```cpp
constexpr int64_t MAX_DRAFT_TOKENS_ATTR_INDEX = 8;
constexpr int64_t DEFAULT_MAX_DRAFT_TOKENS = 7;
constexpr int64_t MIN_DRAFT_TOKENS = 0;
constexpr int64_t MAX_DRAFT_TOKENS = 16;
```

**A.9 tiling.cpp 防御读 op 属性（OP_ATTR 超集核心）**

仿 matmul `base_tiling.cpp:230-233` 的 `GetAttrNum + null 检查` 防御读模式（不依赖 OpDef 注册顺序，只看运行期是否下发了该属性）。在 `tiling.cpp:128-131` 现有 inplace 解析之后、`return SUCCESS`（:132）前新增：

```cpp
// 防御读：老调用方未传 maxDraftTokens 时（value_or(7)），用默认值
auto attrs = context_->GetAttrs();
int64_t maxDraftTokensVal = DEFAULT_MAX_DRAFT_TOKENS;
if (attrs != nullptr && attrs->GetAttrNum() > MAX_DRAFT_TOKENS_ATTR_INDEX) {
    auto max_draft_tokens_ptr = attrs->GetAttrPointer<int64_t>(MAX_DRAFT_TOKENS_ATTR_INDEX);
    if (max_draft_tokens_ptr != nullptr) {
        maxDraftTokensVal = *max_draft_tokens_ptr;
    }
}
bool isValidMdt = (maxDraftTokensVal >= MIN_DRAFT_TOKENS && maxDraftTokensVal <= MAX_DRAFT_TOKENS);
OP_CHECK_IF(!isValidMdt,
    OP_LOGE(context_->GetNodeName(),
        "maxDraftTokens must be in [%ld, %ld], but got %ld.",
        MIN_DRAFT_TOKENS, MAX_DRAFT_TOKENS, maxDraftTokensVal),
    return ge::GRAPH_FAILED);
maxDraftTokens_ = maxDraftTokensVal;
```

（class 成员需新增 `int64_t maxDraftTokens_ = DEFAULT_MAX_DRAFT_TOKENS;`）

**A.10 C5 落点：放大 stateLen_（只新增，不动原引用）**

C5 决策（路径 B）：`effectiveStateLen = max(stateLen_, maxDraftTokens_ + windowSize_ - 1)`，落点在 `tiling.cpp:245` `multiTokenNum_` 反推**之后**、`:560`/`:564`/`:590`/`:709` 等使用 stateLen_ 的位置**之前**。**不修改原 stateLen_ 引用**，新增放大写入（与 [[prefer-additive-changes-no-modify-original]] 一致）：

```cpp
// tiling.cpp:245 之后新增（只新增落点，不动 :237/:560/:564/:590/:709 原引用）
int64_t effectiveStateLen = std::max(stateLen_, maxDraftTokens_ + windowSize_ - NUM_ONE);
stateLen_ = effectiveStateLen;  // 后续 :560/:564/:590/:709 读到的是放大后的值
```

**A.11 set_stateLen 下发 kernel（路径已存在，无新代码）**

`:709` 现有 `tilingData_.set_stateLen(stateLen_)` 不动，写入的已是 effectiveStateLen。**kernel `InitBuffer` 读 `tilingData_->stateLen` 自动得到 effectiveStateLen，kernel 代码零改动**（Update `:98` / CUTBS `:82-90` 全动态）。

**A.12 ACLGraph capture/replay 行为（C5 收益）**

- capture 期固定 `maxDraftTokens=16` → effectiveStateLen 稳定 → UB 分配稳定
- replay 期 m 在 [0,16] 内变化 → stateLen 实际值变，但 effectiveStateLen 不变 → **不触发 DynamicCompileStatic 重 tiling**
- 代价：maxDraftTokens > 实际 m 时 UB 略浪费（约 `(maxDT - m)/(maxDT+W-1)` 比例）

### 3.2 场景 B：`m=16`（stateLen=18）改变 tiling 决策与 UB 预算（主线 A）

**B.1 m 从 shape 隐式推**

`tiling.cpp:159`（已 trace）：

```cpp
multiTokenNum_ = xShapePtr->GetOriginShape().GetDim(DIM_ONE) - NUM_ONE;  // x 3D 时 m = shape[1] - 1
```

`conv_states.shape[1] = 18` → `multiTokenNum_ = 16`（通过 x.shape[1]=17 反推）。

**B.2 校验**

`:162` `multiTokenNum_ <= MAX_M`（改后 16）通过；`:155` `maxQueryLen < MAX_DECODE_LEN+1`（改后 17）通过，确认走 decode 路径。

**B.3 UB 预算重算 — Update 路径（decode + MTP）**

源 `tiling.cpp:585-640`（已 trace）。核心公式：

```cpp
convStateQueueQueueSpace = stateLen_ * xDtypeSize_;            // :590
spaceWithDim = xQueue + weight + convStateQueue + output + yBuf + xBuf;  // :594-595
maxDim = ((ubSize - UB_RESERVED) / spaceWithDim / ALIGN_SIZE) * ALIGN_SIZE;  // :597
```

**stateLen 9→18 时**（W=3, BF16）：
- `convStateQueueQueueSpace`: 9×2=18B → 18×2=36B
- `spaceWithDim`: ~33B → ~51B（每 dim 元素）
- `maxDim` 约下降 35%（具体值依赖 ubSize 实测）

**关键机制**：`maxDim` 自适应收缩 → `:613-619` 的 `cD = (dimSize + maxDim - 1) / maxDim` 增加 → dim 切分数增加 → **UB 不溢出，但 batch 并行度降低，性能劣化**。

**B.4 UB 预算重算 — CUTBS 路径（prefill, dim 小）**

源 `tiling.cpp:555-583`（已 trace）。核心公式：

```cpp
cacheQue = DOUBLE_BUFFER * stateLen_ * xDtypeSize_ * MAX_DIM_CUTBS;  // :560
cacheBuf = stateLen_ * xDtypeSize_ * MAX_DIM_CUTBS;                  // :564
spaceWithUbfactor = ubSize - (UB_RESERVED/2 + inQueueX_ + inQueueW + cacheQue + yBuf + xFp32_ + weightBuf + cacheBuf);  // :570
ubFactor_ = spaceWithUbfactor / (inQueueX + outQueue + xFp32Buf);    // :572
```

**stateLen 9→18**（K=3, baseDim=512, BF16）：
- cacheQue: 2×9×2×512 = 18KB → 2×18×2×512 = 36KB（+18KB）
- cacheBuf: 9×2×512 = 9KB → 18×2×512 = 18KB（+9KB）
- 合计 +27KB
- `spaceWithUbfactor` 减小 → `ubFactor_` 28→24（预计）→ 单次迭代 token 数 ↓ → 循环 +18%

**kernel InitBuffer 验证**：`op_kernel/ai_infra_fused_causal_conv1d_fn_cutbs.h:82-90`（已 trace）`cacheQue/cacheBuf` 全部用 `tilingData_->stateLen` 动态算，**kernel 零改动**。

**B.5 边界场景：m=16 + D=16384 + W=6**

最坏 UB 占用（Update 路径）：
- stateLen = W-1+m = 5+16 = 21（maxDraftTokens=16 时 effectiveStateLen 与之一致）
- `convStateQueue = 21 × curDim × 2B`，若 curDim = maxDim（自适应后）≈ ~512，则 ~21KB
- 总 UB（含 xQueue/weightQueue/outputQueue/yBuf/xBuf）≈ 不溢出
- dim 切分数 = 16384 / 512 = 32 核，batch 并行度大幅下降

**这是 R1 高风险点**，必须上板验证（msprof 看 UB 占用）。

### 3.3 边界场景：ACLGraph capture/replay（路径 A vs B）

**路径 A（不改 buffer 公式）**：
- capture 期 `conv_states.shape[1]=18` 被 tiling 冻结 → UB 按实际 m=16 分配
- replay 期若 m 变化（如 m=8 → shape[1]=10）→ **触发 DynamicCompileStatic 重 tiling** → 破坏 graph 性能

**路径 B（C5 决策，已采纳）**：
- tiling 计算 `effectiveStateLen = max(stateLen_, maxDraftTokens_ + windowSize_ - 1)`
- capture 期固定 `maxDraftTokens=16`，replay 期 m 在 [0,16] 内变化 → stateLen 实际值变，但 effectiveStateLen 不变 → **UB 分配稳定，不触发重 tiling**
- 代价：maxDraftTokens > 实际 m 时 UB 略浪费（约 `(maxDT - m)/(maxDT+W-1)` 比例）
- 落点见 A.10

---

## 四、实施顺序与检查点

### 阶段 1 — 签名层全栈通

**目标**：A1-A3 / P1-P5 全部落地，全仓能编译通过。maxDraftTokens 不进 OpDef，走 OP_ATTR 末尾追加。

**改动点**：

| # | 文件:行 | 改动 |
|---|---|---|
| A1 | `op_api/aclnn_ai_infra_fused_causal_conv1d.h:21-44` | GetWorkspaceSize 签名加 `int64_t maxDraftTokens`（inplace 后、yOut 前），仿 matmul `aclnn_ai_infra_matmul.cpp:905` |
| A2 | `op_api/aclnn_ai_infra_fused_causal_conv1d.cpp:136-181` | 函数定义 + CommonProcess 签名加 `int64_t maxDraftTokens` + 透传 maxDraftTokens 给 l0op（OP_ATTR 末尾追加 index 8，仿 matmul `aclnn_ai_infra_matmul.cpp:930-931` 的 `batchInvariant` 实参透传） |
| A3 | `op_api/ai_infra_fused_causal_conv1d.{h,cpp}:24-46` | l0op 函数签名加 `int64_t maxDraftTokens` + OP_ATTR 末尾追加 maxDraftTokens（index 8，不动原 8 个顺序） |
| P1-P3 | `torch_ops_extension/.../npu_ai_infra_fused_causal_conv1d.cpp:19-104` | npu + meta 双签名加 `c10::optional<int64_t> max_draft_tokens`；value_or(7) 注入默认 |
| P4 | 同上 `:52-73` | EXEC_NPU_CMD_V1 透传 `max_draft_tokens_const`（inplace 后、y 前） |
| P5 | `torch_ops_extension/.../ops_def_registration.cpp:70` | `m.def` schema 末尾追加 `, int max_draft_tokens=7`（与 omni 仓主流 snake_case 一致，参考 `:30/49/62/156/181` 的 `batch_invariant=False`） |

**检查点**：全仓编译通过（`build.sh`，无符号 undefined / 签名不匹配）；op_api UT（U4）所有 `aclnn...GetWorkspaceSize(...)` 调用加参数后能编译。

### 阶段 2 — tiling 层

**目标**：H1/H2/H4 常量改、T1 防御读、T2/T3/T4 字段、T6 校验、C5 落点放大 stateLen_。UT 阳性 m∈{8,12,16} 通过、阴性 m=17 报错。

**改动点**：

| # | 文件:行 | 改动 |
|---|---|---|
| H1 | `op_host/ai_infra_fused_causal_conv1d_tiling.h:85` | `MAX_M = 7` → `16` |
| H2 | `op_host/ai_infra_fused_causal_conv1d_tiling.h:81` | `MAX_DECODE_LEN = 8` → `17`（语义 = MAX_M+1） |
| H4 | 同文件常量区 | `MAX_DRAFT_TOKENS_ATTR_INDEX=8` + `DEFAULT/MIN/MAX_DRAFT_TOKENS`（与 A.8 一致） |
| T1 | `op_host/ai_infra_fused_causal_conv1d_tiling.cpp:131` 后 | 防御读 maxDraftTokens（GetAttrNum + null 检查，仿 matmul `base_tiling.cpp:230-233`）+ 范围校验 |
| T2 | tiling.cpp 类定义 | 加成员 `int64_t maxDraftTokens_ = DEFAULT_MAX_DRAFT_TOKENS;` |
| **C5 落点** | `tiling.cpp:245` 后 | `effectiveStateLen = max(stateLen_, maxDraftTokens_ + windowSize_ - 1)`，写回 `stateLen_`（**只新增，不动 :237/:560/:564/:590/:709 原引用**） |
| T3 | `tiling.cpp:709` 附近 | 现有 `set_stateLen(stateLen_)` 不动（写入已是 effectiveStateLen） |
| T4 | `tiling.h:132` 后（END_TILING_DATA_DEF 前） | 不新增字段（C5 路径 B 复用 stateLen 字段，kernel 读 effectiveStateLen） |
| T6 | `tiling.cpp` `CheckNumAcceptedTokensValid` 附近 | 加 `accepted[i] ∈ [0, multiTokenNum_+1]` 且 `≤ maxDraftTokens_+1` |
| U1 | `tests/ut/op_host/test_ai_infra_fused_causal_conv1d_tiling.cpp:1931` | 阴性 `M=8` → `M=17` |
| U2 | 同上 `:2729` | `stateLen=10→M=8` 改为 `stateLen=19→M=17` |
| U3 | 同上 | 加阳性 m∈{8,12,16} 用例 |

**检查点**：UT 全绿（阳性 m=16 通过、阴性 m=17 报错）。

### 阶段 3 — kernel 不溢出验证（不改代码）

**目标**：上板确认大 m + 大 D UB 不溢出，性能劣化在可接受范围。

**kernel 验证点（不改代码）**：

| # | 文件:行 | 验证 |
|---|---|---|
| K1 | `op_kernel/ai_infra_fused_causal_conv1d_update.h:98` | `convStatesQueue_` stateLen 自动跟随（=effectiveStateLen），maxDim 自适应收缩 |
| K2-K3 | `op_kernel/ai_infra_fused_causal_conv1d_fn_cutbs.h:82-90` | cacheQue/cacheBuf stateLen 自动跟随，ubFactor 自适应 |
| K4 | `op_kernel/ai_infra_fused_causal_conv1d_fn_cutbsd.h` | 不依赖 stateLen，不受影响 |
| K5 | update.h `CalRunningCacheRWIdx`（:179+） | accepted 偏移在 m=16 范围内合法（由 T6 校验保证） |

**检查点（WEAK → STRONG 需上板）**：
- msprof 跑 D=16384 × m=16 × W=6（Update 路径最坏 UB），看 UB 占用 < UB_SIZE
- atk 全量 json 回归（m=8..16 decode 用例）通过
- 性能：m=7（默认 maxDraftTokens）无劣化；m=16 可接受劣化

### 阶段 4 — golden + atk 扩展

**目标**：atk 11 json + 11 executor 加字段；项目 golden 新建覆盖 m∈[0,16]×D×accepted×apc 矩阵。

**改动点**：

| # | 文件 | 改动 |
|---|---|---|
| G1 | `CustomOP/atk/AiInfraFusedCausalConv1d/*.json` (11 个) | 每个 case 加 `maxDraftTokens` 字段（默认 7）；新增 m∈{8,12,16} case |
| G2 | `CustomOP/atk/AiInfraFusedCausalConv1d/executor_*.py` (11 个) | executor 调用 npu 处加 `maxDraftTokens=`；golden oracle 逻辑不改（已动态） |
| G3 | `projects/omni-fused_causal_conv1d算子新增投机tokens数参数/golden/golden_fused_causal_conv1d_mtp.py` | 新建。继承 atk `causal_conv1d_golden` oracle，套 bench_metrics 输出 |

**caseData 矩阵**（G3）：
- m ∈ {0, 7, 8, 12, 16}（边界 + 中间档）
- D ∈ {64, 1024, 8192, 16384}（小/中/大/极限）
- accepted ∈ {0, 1, m/2, m}（per-batch 变化）
- apc ∈ {on, off}

**判据**：双标杆（有 GPU）`max_re_ratio ≤ 5, avg_re_ratio ≤ 1.5, rms_ratio ≤ 1.5`；无 GPU 降级 flat isclose + err_ratio。**输出两份都要比**（y + 写回的 conv_states），cache 写错不会反映在 y 上是常见 false-pass。

**检查点**：项目 golden N/N PASS；atk 全量回归通过。

### 阶段 5 — 文档

**目标**：M1-M3 同步，doc-sync 全仓扫无残留。

| # | 文件 | 改动 |
|---|---|---|
| M1 | `docs/aclnnAiInfraFusedCausalConv1d.md`（omni 仓） | 签名 + m 范围 7→16 + stateLen 9→18 + maxDraftTokens 参数行 + accepted 值域 |
| M2 | `docs/npu_ai_infra_fused_causal_conv1d.md`（omni 仓） | torch 接口同步 |
| M3 | `D:\Desktop\Code\ops-transformer_AI\attention\fused_causal_conv1d\docs\aclnnFusedCausalConv1d.md`（上游） | 视交付要求，可后置 |

**检查点**：`ascendc-doc-sync` skill 扫 `MAX_M=7` / `[0,7]` / `multiTokenNum=7` 字样无残留。

---

## 五、剩余风险

仅 **R1** 待上板验证。

| 项 | 影响 | 处理 |
|---|---|---|
| **R1 大 m + 大 D UB 溢出** | 阶段 3 验证门槛 | 上板跑 D=16384 × m=16，msprof 看 UB（WEAK → STRONG 需上板）。UB 自适应（`spaceWithDim` 动态 `maxDim`，源 `tiling.cpp:590-597`）保证不溢出，但性能劣化需量化 |

---

## 六、golden 实施要点

### 6.1 oracle 继承

atk `executor_ai_infra_fused_causal_conv1d_continue.py:73` 的 `causal_conv1d_golden` 是纯 PyTorch 实现，**已按 shape 动态处理任意 m**（`padded_input` 长度自动伸缩）。项目 golden G3 直接 import 即可，无需重写 oracle 逻辑。

### 6.2 caseData 矩阵思路

| 维度 | 取值 | 目的 |
|---|---|---|
| m | {0, 7, 8, 12, 16} | 边界回归 + 新增中间档 |
| D | {64, 1024, 8192, 16384} | 小/中/大/极限 UB 压力 |
| accepted | {0, 1, m/2, m} | per-batch 变化、MTP offset 边界 |
| apc | {on, off} | APC 跨多 block 写 cache |

### 6.3 可证伪设计

- **阳性**：m=16, accepted=8，y 与 conv_states 都对齐 golden
- **阴性 A**：m=16, accepted=17（越界），期望 T6 校验报错
- **阴性 B**：m=16 但 `conv_states.shape[1] != W-1+m`，期望 CheckConvStatesValid 报错
- **回归**：m=0/1（原 decode 单 token）不破

### 6.4 判据

参考 atk json 的 `cv_fused_double_benchmark`：max_re_ratio ≤ 5, avg_re_ratio ≤ 1.5, rms_ratio ≤ 1.5（双标杆，需 GPU）。无 GPU 时降级 flat isclose + err_ratio。

---

## 附：命名分层表（C3 决策）

`maxDraftTokens` 各层命名遵循本算子现有 8 attr 的分层惯例（已 trace `maxQueryLen` 的分布：aclnn `:33/141` camelCase / TilingData `tiling.h:124` camelCase / 成员 `tiling.h:211 maxQueryLen_` / setter `tiling.cpp:713 set_maxQueryLen`）：

| 层 | 命名 | 风格 | 参考 |
|---|---|---|---|
| Python kwarg | `maxDraftTokens` | camelCase | 需求 §2.4.1 |
| torch schema | `max_draft_tokens` | snake_case | `ops_def_registration.cpp:70`（同仓 `:30/49/62/156/181` 的 `batch_invariant`） |
| aclnn C 签名 | `maxDraftTokens` | camelCase | `aclnn_...cpp:136` 现有 `maxQueryLen` |
| l0op 函数参数 | `maxDraftTokens` | camelCase | l0op 现有 `maxQueryLen` |
| TilingData 字段 | 不新增（C5 复用 stateLen） | — | — |
| INDEX 常量 | `MAX_DRAFT_TOKENS_ATTR_INDEX` | UPPER_SNAKE | `tiling.h:48-55` 现有 `INPLACE_INDEX` 等 |
| tiling 局部变量 | `max_draft_tokens_ptr` / `maxDraftTokensVal` | snake / camel 混用（局部） | 仿 `attrsPtr->GetInt(...)_ptr` 模式 |
| tiling 类成员 | `maxDraftTokens_` | camelCase + 尾下划线 | `tiling.h:211 maxQueryLen_` |
| OpDef attr | **不涉及**（不进 OpDef，走 OP_ATTR 超集） | — | — |

---

## 附：关键文件路径速查

```
omni-ops 主仓:
  D:\Desktop\Code\omni-ops\inference\ascendc\src\ops-transformer\attention\ai_infra_fused_causal_conv1d\
    op_host\ai_infra_fused_causal_conv1d_def.cpp                    # OpDef（不改，不进 OpDef）
    op_host\ai_infra_fused_causal_conv1d_tiling.{h,cpp}             # 常量 + 防御读 + UB + C5 落点（H1-H4, T1-T6）
    op_kernel\ai_infra_fused_causal_conv1d_update.h                 # Update kernel（不改，验证）
    op_kernel\ai_infra_fused_causal_conv1d_fn_cutbs.h               # CUTBS kernel（不改，验证）
    op_kernel\ai_infra_fused_causal_conv1d_fn_cutbsd.h              # CUTBSD kernel（不受影响）
    op_api\aclnn_ai_infra_fused_causal_conv1d.{h,cpp}               # aclnn + l0op OP_ATTR 末尾追加（A1-A3）
    op_api\ai_infra_fused_causal_conv1d.{h,cpp}                     # l0op（A3，OP_ATTR 末尾追加 maxDraftTokens）
    docs\aclnnAiInfraFusedCausalConv1d.md / npu_...md               # 文档（M1-M2）
    tests\ut\op_host\test_..._tiling.cpp / tests\ut\op_api\test_aclnn_...cpp  # UT

  D:\Desktop\Code\omni-ops\inference\ascendc\torch_ops_extension\omni_custom_ops\
    csrc_base\ops_def_registration.cpp                               # torch schema（P5）
    ops_transformer\attention\ai_infra_fused_causal_conv1d\csrc\npu_ai_infra_fused_causal_conv1d.cpp  # binding（P1-P4）

matmul OP_ATTR 超集参考实现（已 trace）:
  aclnn_ai_infra_matmul.cpp:45                                        # OP_ATTR(..., batchInvariant) 绑定（行号证实 OP_ATTR 路径）
  aclnn_ai_infra_matmul.cpp:431/905/931                               # aclnn 收 batchInvariant → SetBatchInvariant 写 opInfo（中间层），最终作 l0op 实参进 OP_ATTR
  ai_infra_matmul_base_tiling.cpp:65-66/230-233                     # ATTR_NUM/INDEX 常量 + 防御读
  ai_infra_matmul_def.cpp                                           # 无 batch_invariant 的 .Attr()（不进 OpDef）

atk / golden:
  D:\Desktop\Code\CustomOP\atk\AiInfraFusedCausalConv1d\*.json / executor_*.py  # G1-G2
  D:\Desktop\Code\CC4Ascend\projects\omni-fused_causal_conv1d算子新增投机tokens数参数\golden\golden_fused_causal_conv1d_mtp.py  # G3 新建
```
