# MHC（多头组合 / Sinkhorn）类算子范式

真实源：`ops-transformer_AI/mhc/`。共 **8 个算子 + mhc_sinkhorn_common**。arch 分布已全量验证。

## 算子清单与 arch 分布（已验证）

| 算子 | arch22 | arch35 | 说明 |
|---|:---:|:---:|---|
| `mhc_sinkhorn` | | ✓ | Sinkhorn-Knopp 迭代→双随机矩阵投影 |
| `mhc_sinkhorn_backward` | | ✓ | 反向 |
| `mhc_pre` | | ✓ | H_res/H_post 投影 + h_in |
| `mhc_pre_backward` | | ✓ | 反向 |
| `mhc_pre_sinkhorn` | | ✓ | Pre + Sinkhorn 融合 |
| `mhc_pre_sinkhorn_backward` | **✓** | ✓ | 反向（双架构） |
| `mhc_post` | **✓** | ✓ | Post Mapping + Res Mapping + 残差（双架构） |
| `mhc_post_backward` | **✓** | ✓ | 反向（双架构） |

> 旧 skill 误记"唯一双架构是 mhc_post_backward"，实际 **3 个算子双架构**（mhc_post / mhc_post_backward / mhc_pre_sinkhorn_backward）。

## 特征

- **几乎全 A5**：8 个算子中 5 个 arch35 only
- **核心算法**：Sinkhorn-Knopp 迭代将混合矩阵投影到双随机流形，稳定深度网络信号传播
- `mhc_sinkhorn_common/op_host/arch35/`：共享头文件

## 来源
- `ops-transformer_AI/mhc/`（find arch* 全量验证）