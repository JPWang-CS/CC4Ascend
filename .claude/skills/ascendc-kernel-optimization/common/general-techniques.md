# 通用 Kernel 优化技巧

跨算子类的通用优化技巧归纳。各技巧均从真实代码 trace 得出，附首次出现的算子范例。

## 1. 同地址冲突规避（A2A3）→ 同地址并行（A5）

- **A2A3**：多核写同一 GM 地址会冲突，需**错位/对角线分核**规避（attention 四种分核、gmm `MNBlockIdxCompute` + `thresholdDimM=5`）
- **A5**：硬件支持同地址并行，可简化为**规则分核**（顺序/对称/TND/SWAT），减无效偏移
- 迁移要点：A2A3→A5 时先功能等价保留 tile，再放开分核约束，profiling 看 MAC/MTE2/L2 命中率

## 2. 双缓冲 Ping/Pong 流水掩盖

- Cube↔Vector 交替的算子（attention/matmul）用 **Ping/Pong TBuf** 让第 n 轮搬出与第 n+1 轮搬入 overlap
- 同步靠 `HardEvent::MTE3_MTE2` SetFlag/WaitFlag 配对（attention `:787-957`）
- 加深：`BigDoubleBuffer` TilingKey 开关增加流水深度（UB 足够时）

## 3. 多级流水滑动窗口（MSD / 三阶）

- 量化算子（gmm swiglu_quant / ffn antiquant_msd）用 **PreProcess(n+1)/MidProcess(n)/PostProcess(n-1)** 滑动窗口
- weight 反量化(AIV) → matmul(AIC) → dequant+activation+requant(AIV) 三级重叠
- 适用：A8W4/A16W4 等需 weight 反量化的量化路径

## 4. 量化分档（noquant / fullquant / fullquant_mx）

- A5 attention 在 common 提供 **noquant / fullquant_gqa / fullquant_mx** 三档共享模块
- mc2 matmul_all_reduce tiling 分 **quant / unquant / weight_quant** 三档
- 分档决定 UB scale 预算与流水结构

## 5. GEMV 阈值切换

- 瘦长矩阵（M ≤ 阈值）切 GEMV 路径，避免 Normal MM 的 padding 浪费
- gmm `GEMV_THRESHOLD = 8`，独立 `*_gemv.h` 实现（逐行 LoadData + Mmad + Dequant）
- 由 TilingKey 切换（`TILING_KEY_GEMV`）

## 6. Regbase（A5）vs Membase（A2A3）

- A2A3：`LocalTensor` + TQue/TBuf + 显式 Alloc/EnQue/DeQue/Free
- A5：`RegTensor` + MicroAPI + `MaskReg` + `LoadDist/StoreDist`，寄存器级计算无 UB 中间缓冲
- RoPE/elementwise 类（posembedding）A5 用 `DINTLV_B32`(De-Interleave) / `DIST_UNPACK_B16`(BF16 解包) / `DIST_PACK4_B32`(int8 打包)

## 7. CrossCore 同步严格匹配（A5）

- A5 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 必须**一一匹配**，无 HWTS 兜底，不匹配必死锁
- 排查重点：异常分支提前 return 导致 Set/Wait 不配、多 stage 复用 flagId 串扰、循环边界不对齐
- A2A3 有兜底可掩盖这些问题，A5 直接暴露

## 8. broadcast 后缀编码（cos/sin / scale 对齐）

- posembedding RoPE 用 tiling 后缀 `_ab/_aba/_ba/_bab` 编码 cos/sin 的 B 维 broadcast
- 对应不同 layout（BNSD/BSND/SBND/TND）的对齐方式

## 9. 变体按场景拆文件（避免 padding/无效计算）

- moe routing：normal / small_active_row / fullload 三变体（激活行数不同）
- attention 分核：S1/S2/Batch/SameAB 四类（shape 不同）
- posembedding：按 broadcast 后缀拆文件
- 原则：host tiling 据 shape/场景选变体，避免统一路径的 padding 浪费

## 10. 模块化共享基础设施

- attention `common/op_kernel/arch35/`：fia_block_cube/vec + fia_kernel 共享（50+ 文件）
- gmm `gmm_infra/`：CUTLASS 式 gemm/epilogue/layout 模板 + `arch/gmm_arch.hpp` 架构抽象
- mc2 `3rd/`：mat_mul_v3/batch_mat_mul_v3/rms_norm 共享计算库
- 趋势：新算子倾向模块化模板 + 架构抽象层，而非 arch22/arch35 硬拆