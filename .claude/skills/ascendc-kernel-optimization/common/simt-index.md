# SIMT / Index 离散访存范式

ops-nn index 类（80 算子，embedding/gather/scatter/bucketize 等）。内容经真实代码 trace + ascendc-development SIMT 章节。源：`ops-nn/index/`。

## 范式特征

index 类是**离散访存主导**：按索引从 GM 收集/分散数据，地址不连续。
- arch 分布：159 arch35 / 7 arch32 / 4 arch22（已验证，有 arch32 独有）
- 算子：embedding(+bag) / gather_elements(+v2) / gather_nd / scatter / bucketize(+v2) / apply_top_k_top_p / index_add ...

## SIMD vs SIMT 选路（已验证 ascendc-development）

离散访存用 **SIMT**（线程级并行），连续访存用 SIMD：
- **尾轴 ≤ 2048** → SIMT（多线程各自处理离散小块）
- **尾轴 > 2048** → SIMD（向量化连续搬运）

SIMT 核心（gather_v2 范例）：
```cpp
__simt_vf__ LAUNCH_BOUND(2048) void GatherSimt(...) {
    for (idx = GetThreadIdx(); idx < elements; idx += GetThreadNum()) {
        // 每线程独立算索引、直接访问 GM
        y[idx] = outOfBound ? 0 : x[xIndex];
    }
}
```
- 无需显式 UB buffer（线程直接 `__gm__` 访问）
- 线程间隐式同步

## base 设计理由
- 离散访存无法用 DataCopy 连续搬运 → SIMT 线程各自取
- embedding/gather 的瓶颈在随机访存带宽，非算力
- 索引规整（分桶/排序）可减随机访存

## 通用优化
- **上游索引规整**：scatter/gather 前做索引分桶/排序，提高访存连续性
- **边界与主路径解耦**：热点循环不引入分支
- **SIMD+SIMT 混合**：按数据分布选路径，非固定单一
- **线程负载均衡**：任务切分匹配稀疏性，避免线程饿死

## base 流水（SIMD 连续路径，elementwise 相似）
```
CopyIn（按索引批量搬） → Compute（gather/scatter 拼装） → CopyOut
```
SIMD 路径用 TQue + DataCopyPad，SIMT 路径用 `__simt_vf__` 直接 GM。

## index 类算子（已 trace ops-nn/index/）

离散访存算子清单（已验证）：`embedding` / `embedding_bag` / `embedding_dense_grad`(+v2) / `inplace_index_add` / `gather_elements`(+v2) / `gather_nd` / `bucketize`(+v2) / `apply_top_k_top_p_with_sorted` / `scatter` 系列。

arch 分布：159 arch35 / 7 arch32 / 4 arch22（index 类含 arch32，离散访存在 310P 也有实现）。

### embedding / index_add 范式
embedding 按 indices 从权重表 gather 行；index_add 按 indices scatter-add 到输出。两者都是**索引驱动的离散搬运**：
- 连续路径（SIMD）：DataCopyPad 按索引批量搬，TQue 管理
- 离散路径（SIMT）：`__simt_vf__` 线程各自取一个索引访问 GM

选路规则同 gather_v2：尾轴 ≤2048 用 SIMT（多线程离散高效），>2048 用 SIMD（向量化连续高效）。