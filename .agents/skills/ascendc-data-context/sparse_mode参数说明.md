# sparseMode 稀疏模式

sparseMode 指 Attention 计算中的稀疏 mask 模式，用于遮蔽 QKᵀ 矩阵的特定位置。

## 模式速查表

| sparseMode | 名称 | 说明 | 特殊约束 |
|:---:|---|------|------|
| 0 | defaultMask | 根据 preTokens/nextTokens 自动决定 mask | 默认模式 |
| 1 | allMask | 全量 attention mask 矩阵 | 忽略 preTokens/nextTokens |
| 2 | leftUpCausal | 左上角下三角 causal mask | 忽略 preTokens/nextTokens |
| 3 | rightDownCausal | 右下角下三角 causal mask | 忽略 preTokens/nextTokens |
| 4 | band | preTokens/nextTokens 之间的带区域 | 起点右下角，两者需有交集 |
| 5 | prefix(非压缩) | rightDownCausal + 左侧 prefix 矩形 | varlen 不支持 |
| 6 | prefix(压缩) | 压缩下三角 + 矩形 mask (3072×2048) | — |
| 7 | varlen rightDownCausal | 长序列外切 + rightDownCausal | 仅 varlen |
| 8 | varlen leftUpCausal | 长序列外切 + leftUpCausal | 仅 varlen |
| 9 | treeMask | 推测解码树形注意力掩码 | GQA/MLA(非量化), MLA only(全量化) |

## 关键模式详解

### sparseMode=0 (defaultMask)

最灵活的模式，通过 preTokens/nextTokens 控制：
- **不传 mask** → 无 mask
- **nextTokens=0, preTokens≥Sq** → causal 下三角 mask
- **preTokens<Sq, nextTokens>0** → band mask
- **nextTokens<0** → preTokens 必须 ≥ |nextTokens|

### sparseMode=5/6 (prefix)

prefix 场景：在 causal 基础上左侧 + N 列 prefix 可见区域
- sparseMode=5: 非压缩矩阵 (BNSS 或 B1SS)
- sparseMode=6: 压缩矩阵 (3072×2048)

### sparseMode=7/8 (varlen 外切)

长序列跨卡切分场景：
- sparseMode=7: 最后一卡变 band，其余 rightDownCausal
- sparseMode=8: 第一卡变 band，其余 leftUpCausal

**sparseMode=7 band 参数约束：**
- preTokens ≥ last_Skv
- last_Sq - last_Skv ≤ nextTokens ≤ 0
- 非 band batch: Sq ≤ Skv

### sparseMode=9 (treeMask)

推测解码场景，支持 BSND/BNSD (3D) 或 TND (紧凑 1D) 格式：
- 对角线 = 0（关注自身）
- 上三角 = 1（不关注未来）
- 下三角 = 0/1（树结构决定）
- 每个 batch: Q_S ≤ KV_S

## 来源
- `ops-transformer_AI/docs/zh/context/sparse_mode参数说明.md`
