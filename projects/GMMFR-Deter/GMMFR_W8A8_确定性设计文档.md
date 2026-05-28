日期	修订记录	作者
（留空）	初版	（留空）


# 1 算子需求

## 1.1 设计约束

运行环境：A5（Ascend 950）
调用方式：单算子
适用算子：GroupedMatmulFinalizeRouting（GMMFR），pertoken 量化路径（W8A8）

## 1.2 需求描述

GMMFR W8A8 确定性计算。

### 1.2.1 功能描述

GroupedMatmulFinalizeRouting 算子为 GroupedMatmul 和 MoE FinalizeRouting 的融合算子，
其 FinalizeRouting 阶段通过 rowIndex 将各 expert 的 token 结果 scatter-add 到输出矩阵
对应行。非确定性路径中，多核通过 AtomicAdd 并发写同行不同列块时，浮点累加顺序不固定
（(a+b)+c ≠ a+(b+c)），且核间调度顺序不固定，导致每次运行数值不一致。

确定性子特性新增确定性计算模式，通过控制累加操作顺序保证每次计算结果完全一致。

### 1.2.2 适用范围

仅支持 W8A8 pertoken 量化场景：
- 输入 x（左矩阵）：INT8，ND
- 输入 w（右矩阵）：INT8，NZ
- scale 量化参数：FLOAT32
- pertokenScale：FLOAT32
- 输出 y：FLOAT32

其他数据类型组合（INT4 weight、MX 量化等）不开启确定性模式。

### 1.2.3 精度要求

精度继承原算子标准，满足双千分之一。
开启确定性后，同一输入多次运行结果完全一致。


# 2 算子设计

## 2.1 总体设计

核心思想：将非确定性路径的"即时 AtomicAdd 散写"改为"延迟写回 + 滑窗批量同步"，
通过控制累加顺序保证确定性。

确定性机制包含五个要素：

| 要素 | 说明 |
|---|---|
| ① 工作空间分配 | 从 L2 中划分确定性 workspace（mmQuantOutGm），用于暂存线性写入结果 |
| ② 行优先调度 | N 先变→M 前进，zigzag N 方向，保证写入顺序与 token 行号对应 |
| ③ 线性写入 | AIV 反量化后按 token 顺序写入 mmQuantOutGm，每条结果写完整一行 |
| ④ 滑窗触发 | 累计 M 超窗口上限时 SyncAll，调用 FRDeterministic 批量归约 |
| ⑤ 分核归约 | 回写行按 outRow % coreNum == blockIdx 分配，每行仅单核处理 |

确定性判定逻辑（DeterministicTilingProcess）：
- context->GetDeterministic() != 0
- aDtype == DT_INT8 且 bDtype == DT_INT8
同时满足时 deterministicFlag = 1。

## 2.2 计算流程

### 2.2.1 非确定性 vs 确定性数据流对比

非确定性路径：
AIC: MatMul(int8×int8→int32) → NotifyVector
AIV: WaitForCube → Dequant(x2scale, x1scale, bias) → LogitMuls
     → AtomicAdd → yGm[rowIndex[i]]

确定性路径：
AIC: MatMul(int8×int8→int32) → NotifyVector
AIV: WaitForCube → Dequant(x2scale, x1scale, bias) → LogitMuls
     → LinearWriteToWorkspace(mmQuantOutGm)

滑窗满时：
     SyncAll → FRDeterministic:
       for each row: outRow = rowIndex[m]
         if outRow % coreNum == blockIdx: AtomicAdd → yGm[outRow]
     → SyncAll → 窗口前移

### 2.2.2 滑窗生命周期

1. Init：根据 deterWorkspaceSize 计算 windowSize = workspaceSize / (N * sizeof(float))
2. 累积：每个 group 完成后 cumulativeGroupM += curGroupM
3. 触发：cumulativeGroupM > lowBoundM 时触发
4. FR：SyncAll → FRDeterministic 分核归约 → SyncAll
5. 前移：windowStartM = curGroupM，lowBoundM = curGroupM + windowSize
6. 收尾：所有 group 处理完后，最后再 SyncAll → FRDeterministic → SyncAll

### 2.2.3 同步点汇总

算子执行过程中出现 4 处 SyncAll<false>()：
- 入口：prologue 初始化完毕后
- 滑窗触发：FRDeterministic 前后各一次（窗口满时）
- 出口：最终收尾 FRDeterministic 前后各一次

## 2.3 程序架构设计

### 2.3.1 新增文件

| 文件 | 行数 | 功能 |
|---|---|---|
| block_scheduler_gmm_aswt_with_tail_split_pro.h | 265 | 行优先 ASWT 调度器，N 先变→M 前进，窗间 zigzag N 反转 |
| kernel_gmm_finalize_routing_pertoken_dequant_deter.h | 396 | 确定性 Kernel 主循环，滑窗检测 + FRDeterministic |
| block_epilogue_dequant_finalize_routing_deter.h | 599 | 确定性 Epilogue，线性写入 + DeterministicFlushRow |

### 2.3.2 修改文件

| 文件 | 变更 |
|---|---|
| grouped_matmul_finalize_routing_tiling_data.h | TilingData 新增 deterministicFlag、deterWorkspaceSize 字段 |
| grouped_matmul_finalize_routing_quant_tiling.h | 新增 DeterministicTilingProcess()、GetWorkspaceSize() |
| grouped_matmul_finalize_routing_quant_tiling.cpp | 实现确定性判定 + workspace 大小计算 |
| grouped_matmul_finalize_routing_pertoken_dequant.h | 入口新增 deterministicFlag 分支路由到 deter 函数 |
| block_scheduler_policy.h | 新增 GroupedMatmulAswtWithTailSplitProScheduler tag |

### 2.3.3 入口分支

grouped_matmul_finalize_routing_pertoken_dequant() 读取 TilingData 后检查
deterministicFlag。当 deterministicFlag == 1 时路由到 deter 变体函数，替换三组件：

| 组件 | 非确定性 | 确定性 |
|---|---|---|
| Scheduler | GroupedMatmulAswtWithTailSplitScheduler | GroupedMatmulAswtWithTailSplitProScheduler |
| Epilogue | BlockEpilogueDequantFinalizeRouting | BlockEpilogueDequantFinalizeRoutingDeter |
| Kernel | KernelGmmFinalizeRoutingPertokenDequant | KernelGmmFinalizeRoutingPertokenDequantDeter |

## 2.4 Tiling 设计

### 2.4.1 TilingData 变更

GMMFinalizeRoutingDataParams 新增两个字段，由 Host 侧 Tiling 阶段填充后传递给 Kernel：

| 字段 | 类型 | 含义 |
|---|---|---|
| deterministicFlag | uint32_t | 确定性模式开关，1=开启 |
| deterWorkspaceSize | uint32_t | 确定性 workspace 大小（字节），由 DeterministicTilingProcess 计算 |

### 2.4.2 确定性 workspace 分配

DeterministicTilingProcess() 负责确定性子特性的 Tiling 阶段处理：

1. 判定条件检查：context->GetDeterministic() == 0 或非 W8A8 时，deterministicFlag = 0 并返回
2. 计算 workspace 大小：
   - SYS_WORKSPACE_SIZE = 16MB（系统预留）
   - deterWorkspaceSize = (L2_SIZE - 16MB) × 80%
   - DETER_WORKSPACE_RATIO = 0.8

GetWorkspaceSize() 在确定性开启时返回 SYS_WORKSPACE_SIZE + deterWorkspaceSize，
确定性关闭时退化为原基类逻辑。

## 2.5 Scheduler 设计

### 2.5.1 列优先 vs 行优先

原 ASWT Scheduler 为列优先（M 先变），确定性 Pro 版本改为行优先（N 先变）。

**为什么原来用列优先**：非确定性路径 AIV 直接 AtomicAdd 散写到 yGm，写入目标由
rowIndex 决定，与 tile 发射顺序无关。列优先下 M 先变有利于 Cube 侧 weight 矩阵的
L2 复用（同一 weight tile 被连续的 M-tile 共用），减少重复加载 weight 开销。

**为什么确定性需要行优先**：确定性路径 AIV 将结果按 token 顺序线性写入 mmQuantOutGm
（wsRow = groupAccumM + offsetM + i），行优先确保同一 M 窗口内所有 N-tile 连续发射，
同一 token 行的 N 方向结果连续写入 workspace，无需跳转地址。列优先则会让 M 频繁变化，
workspace 写入分散，与 FR 阶段逐行归约不匹配。

**代码差异**：两者 GetTileIdx() 中仅 index→(M,N) 映射公式不同，其余分核逻辑一致。

原版列优先（M 先变）：
```
// block_scheduler_gmm_aswt_with_tail_split.h
M = rowIdx * mainMWindow_ + index % mainMWindow_;
N = (index / mainMWindow_) % nCnt_;
```
index 递增时 M 先变（`index % mainMWindow_` 先增长）。

Pro 版行优先（N 先变）：
```
// block_scheduler_gmm_aswt_with_tail_split_pro.h
M = rowIdx * mainMWindow_ + (index / nCnt_) % mainMWindow_;
N = index % nCnt_;
```
index 递增时 N 先变（`index % nCnt_` 先增长）。

### 2.5.2 zigzag N 方向反转

相邻 M 窗口间 N 方向反转，提升 L2 复用：

示例（mCnt=8, nCnt=4, WINDOW_LEN=4）：

Window 0 (M=0~3), N 正向：
  M=0: (0,0)(0,1)(0,2)(0,3) → M=1: (1,0)(1,1)(1,2)(1,3)
  M=2: (2,0)(2,1)(2,2)(2,3) → M=3: (3,0)(3,1)(3,2)(3,3)

Window 1 (M=4~7), N 反向：
  M=4: (4,3)(4,2)(4,1)(4,0) → M=5: (5,3)(5,2)(5,1)(5,0)
  M=6: (6,3)(6,2)(6,1)(6,0) → M=7: (7,3)(7,2)(7,1)(7,0)

## 2.6 Kernel 设计

### 2.6.1 窗口状态机

Kernel 使用 DeterSyncConfig 结构体维护滑窗状态：

| 字段 | 含义 |
|---|---|
| curM | 当前窗口内已累积的 M 总数 |
| curGroupM | 当前 group 结束后的累积 M |
| lowBoundM | 窗口下界，cumulativeGroupM 超过此值触发 FR |
| windowSize | 窗口容量 = deterWorkspaceSize / (N * sizeof(CType)) |
| windowStartM | 当前窗口起始 M 偏移 |

初始化时计算 windowSize 和初始 lowBoundM = windowSize。每个 group 完成后更新
cumulativeGroupM，若超过 lowBoundM 则触发滑窗。

### 2.6.2 ProcessSingleGroup

每个 group 内的 tile 循环处理逻辑与原 Kernel 一致（AIC MatMul → NotifyVector，
AIV Dequant → NotifyCube），区别在于 AIV 侧写回目标为 mmQuantOutGm 而非 yGm。

AIV 侧调用 SetWorkspaceGroupOffset() 记录当前 group 在窗口内的起始偏移，
Epilogue 的 LinearWriteToWorkspace() 据此计算 workspace 写入位置：

  wsRow = groupAccumM + offsetM + i
  mmQuantOutGm[wsRow * n_ + yOffset] = result[i]

### 2.6.3 FRDeterministic

由 AIV 侧执行，将当前窗口内暂存的 token 结果按 rowIndex 归约到 yGm。

流程：
1. 计算窗口内总行数 totalM = curGroupM - windowStartM
2. 遍历窗口内每行 mOffset：
   a. 通过 rowIndex 获取目标输出行 outRow
   b. 分核过滤：outRow % (coreNum * TaskRation) == GetBlockIdx()，不满足则跳过
   c. 满足条件则调用 DeterministicFlushRow(mOffset, outRow, N)
3. 分核保证每行仅有一个 vector 核执行 AtomicAdd，累加顺序固定

分核过滤公式中 coreNum 为 AIC 核数（即 usedCoreNum），乘以 GetTaskRation() 得到
vector 核总数（A5 上每个 AIC 对应 2 个 AIV）。

### 2.6.4 主循环

确定性 Kernel operator() 主循环流程：

1. prologue 初始化（AIV 归零 yGm + sharedInput）
2. mmadOp 初始化（AIC）
3. 计算 windowSize，lowBoundM = windowSize
4. SyncAll<false>()
5. Epilogue 初始化（AIV）
6. for each group:
     - 更新 group 参数，跳过空 group
     - 若 cumulativeGroupM + curM > lowBoundM:
         SyncAll → FRDeterministic → SyncAll
         窗口前移: windowStartM = curGroupM, lowBoundM = curGroupM + windowSize
     - ProcessSingleGroup
     - cumulativeGroupM += curM
7. SyncAll → FRDeterministic → SyncAll（最终收尾）

## 2.7 Epilogue 设计

确定性 Epilogue（BlockEpilogueDequantFinalizeRoutingDeter）运行在 AIV 侧，分两个阶段：

**计算阶段**（operator()）：继承原 Epilogue 的 Dequant + LogitMuls，写回目标改为
mmQuantOutGm（线性写入 workspace）。

**FR 阶段**（DeterministicFlushRow）：由 Kernel 的 FRDeterministic 调用，将 workspace
中暂存结果按行归约到 yGm。

### 2.7.1 计算阶段（operator()）

计算阶段保持与原 Epilogue 相同的 Ping-Pong 流水，整体流程：

1. CopyInLogit：从 GM 加载 logit 到 UB（double buffer）
2. CopyX2ScaleFromGm2Ub：加载 per-channel scale
3. CopyX1ScaleFromGm2Ub（可选）：加载 per-token scale
4. CopyBiasFromGm2Ub（可选）：加载 bias
5. WaitFlag<MTE2_V>：等待 MTE2 搬运完成
6. VFDoDequantWithX1X2Scale：反量化（int32→fp32 × x2scale × x1scale + bias）
7. LogitPingPong 切换
8. VFDoLogitMuls：logit × 反量化结果 → outUb（Ping-Pong，MTE3 流水）
9. LinearWriteToWorkspace：将 outUb 中的结果按 token 行顺序写入 mmQuantOutGm

第 7-9 步使用 V_MTE3 流水：VF 计算结果→SetFlag<V_MTE3>→MTE3 搬运→WaitFlag<V_MTE3>
→下一轮 VF 计算，实现计算与搬运重叠。

### 2.7.2 线性写入（LinearWriteToWorkspace）

与原 Epilogue 的核心区别。原 Epilogue 在反量化+LogitMuls 后直接 AtomicAdd 散写到
yGm[rowIndex[i]]，确定性 Epilogue 改为按 token 顺序写入 mmQuantOutGm：

  for each row i in [0, curVecBaseM):
    wsRow = groupAccumM + offsetM + i
    mmQuantOutGm[wsRow * N + yOffset] = yLocal[i * alignN]

groupAccumM 由 Kernel 侧的 SetWorkspaceGroupOffset() 在每组开始前设置，记录当前
group 在窗口内的起始 M 偏移。同一行所有 N 方向的 tile 写入 workspace 同一行不同列。

### 2.7.3 FR 回写（DeterministicFlushRow）

由 Kernel 的 FRDeterministic 调用，将 workspace 中暂存的一行结果归约到 yGm：

1. 按 baseN（= 12KB / sizeof(DataTypeOut)）分块遍历 N 方向
2. 每个分块：
   - DataCopyPad 从 mmQuantOutGm[mOffset * N + nOff] 搬到 UB（DMA）
   - DMA EnQue → DeQue 等待
   - SetAtomicAdd<float>()
   - DataCopyPad 写回 yGm[outRow * N + nOff]
   - SetAtomicNone()

每次回写涉及 AtomicAdd，用于与 sharedInput 归约。因分核过滤保证每行仅单核处理，
AtomicAdd 仅为核内安全操作，不引入核间竞争。

## 2.8 Buffer/UB 设计

### 2.8.1 Workspace 使用

确定性新增 GM workspace：

| 缓冲区 | 位置 | 大小 | 用途 |
|---|---|---|---|
| mmQuantOutGm | L2 | deterWorkspaceSize = (L2_SIZE - 16MB) × 80% | 暂存 token 反量化结果，供 FR 阶段分核归约 |

### 2.8.2 UB 使用

确定性 Epilogue 新增 UB：

| 缓冲区 | 大小 | 用途 |
|---|---|---|
| deterDmaQue_ | 12KB | FR 阶段 DMA 搬运队列（TQueBind） |

继承原 Epilogue 的 Ping-Pong 双缓冲：

| 缓冲区 | 大小 | 用途 |
|---|---|---|
| l0cOutUb | MAX_SINGLE_MNSD = 128×256 × sizeof(int32) | AIC→AIV 跨核共享 Cube 输出 |
| logitUb (Ping/Pong) | BLOCK_BYTESD = 256B × 2 | Logit 数据加载 |
| x2ScaleUb (Ping/Pong) | BLOCK_BYTESD × 2 | Per-channel scale |
| x1ScaleUb (Ping/Pong) | BLOCK_BYTESD × 2 | Per-token scale |
| biasUb (Ping/Pong) | BLOCK_BYTESD × 2 | Bias |
| outUb (Ping/Pong) | HALF_DB_MAX_SINGLE_MNSD = 32×256 × sizeof(float) × 2 | LogitMuls 输出 |

### 2.8.3 Workspace 累计分析

确定性模式总 workspace = SYS_WORKSPACE_SIZE(16MB) + deterWorkspaceSize。

以 L2 = 128MB 为例：
- 确定性 workspace = (128 - 16) × 0.8 ≈ 89.6MB
- 总 workspace ≈ 105.6MB

mmQuantOutGm 的行容量：windowSize = deterWorkspaceSize / (N × sizeof(float))
以 N=128 为例：windowSize ≈ 89.6MB / 512B ≈ 183K 行

### 2.8.4 UB 使用汇总

A5 Vector UB = 248KB。确定性 Epilogue UB 布局如下：

VECIN 使用量：
- l0cOutUb/Float: 128KB（双视图重叠占用同一空间）
- logitUbPing/Pong: 2KB (256×float × 2)
- x2ScaleUbPing/Pong: 2KB (256×float × 2)
- x1ScaleUbPing/Pong: 2KB (256×float × 2)
- biasUbPing/Pong: 1KB (256×bf16 × 2)
- 预留偏移: 12KB
- VECIN 合计 = 12KB + 128KB + 7KB = 147KB

VECOUT 使用量：
- outUbPing: 32KB (32×256×float)
- outUbPong: 32KB
- VECOUT 合计 = 64KB

deterDmaQue_：Pipe 独立分配 12KB。

VECIN + VECOUT 合计 = 147KB + 64KB = 211KB < 248KB，不溢出。


# 3 测试设计

## 3.1 确定性功能验证

通过多次调用算子并比对计算结果验证确定性：
- 关闭确定性开关（deterministicFlag=0），多次运行结果不一致，复现非确定性现象
- 开启确定性开关（deterministicFlag=1），多次运行结果逐 bit 一致

## 3.2 泛化与精度测试

- 通过 aclnn_fuzz 泛化用例覆盖所有 W8A8 输入组合
- 精度继承原算子标准，满足双千分之一
- 单用例循环 5 次，结果保持一致
- 边界测试覆盖大 shape 和小 shape 场景
