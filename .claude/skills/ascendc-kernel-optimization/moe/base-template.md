# MoE 类 base 模板（深度版）

以 `moe_init_routing` 为基准。trace 自 `moe/moe_init_routing/op_kernel/arch35/moe_gather_out.h`。

## 数学本质（优化出发点）

MoE routing = 按 expert 路由表把 token 从原位置 `gather` 到 expanded 位置（供后续专家计算），算完再 `scatter` 回去。核心操作是**按索引的离散数据搬运**，非矩阵乘。

核心矛盾：routing 表的索引不连续、各 expert token 数不均，导致**离散访存 + 负载不均**。base 优化的目标是让 gather/scatter 的 UB 分块高效、按激活行数选变体避免空转。

## base 实现：MoeGatherOut（已 trace moe_gather_out.h Process）

### 三层分块循环
```cpp
loops = (coreRows + perLoopRows - 1) / perLoopRows;      // 行分块
colsLoops = Ceil(cols, maxColsOneLoop);                   // 列分块（token 维度）
kLoops = Ceil(coreK, perLoopK);                           // K（隐藏维）分块
for (kLoop) {
  kTileLength = (尾块) ? tailK : perLoopK;
  for (loop < loops-1) {
    for (colsLoop < colsLoops-1) {
      CopyInIndices(loop, kLoop);              // 读 routing 索引
      indicesLocal = DeQue;
      UpdataOffset(loop, colsLoop);            // 算源/目偏移
      CopyIn(loop, colsLoop);                  // 按索引 gather 源数据
      CopyOut(loop, colsLoop, indicesLocal);   // 写到 expanded 目标位置
    }
  }
}
```

### 为什么三层分块
- **K 分块**：隐藏维 K 大时一次放不下 UB，分块搬
- **行分块**：token 行数（coreRows）按核分配后再按 UB 容量切 perLoopRows
- **列分块**：maxColsOneLoop 受 UB 约束，超过则分批

每层都有尾块处理（tailK / tailCols），保证非整数倍 shape 正确。

### 为什么先 CopyInIndices 再 CopyIn
索引决定源地址。先搬索引到 UB（indicesLocal），再用它算每个 token 的源偏移（UpdataOffset），最后按偏移 gather。若索引留 GM 每次随机读，性能差；搬进 UB 后连续消费。

## 变体选择（已 trace 类名）

| 变体 | 适用 | 原因 |
|---|---|---|
| MoeGatherOut | 标准 | 激活行数正常 |
| MoeGatherOutSmallActiveRow | 激活行少 | 标准 perLoopRows 对小激活行空转浪费，用更小分块 |
| MoeInitRoutingFullload | 数据量小 | 全量 load 进 UB，省分块调度开销 |
| MoeMrgsort | 需排序 | routing 索引归并排序保证顺序 |

## base 设计理由总结
routing 类算子瓶颈在**离散访存带宽**而非算力。优化方向：
1. UB 分块（三层）让 gather 连续化
2. 索引预搬进 UB
3. 按激活行数选变体避免空转
4. expert token 不均由 host tiling 的 ComputeExpertParallNum（见 ffn）处理负载

## 相关
专家矩阵乘本身在 `gmm/`（见 gmm base），MoE 类只管 routing/permute。