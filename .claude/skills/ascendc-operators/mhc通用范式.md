# MHC（多头组合 / Sinkhorn）类算子通用范式

## 算子列表（9 个）

| 算子 | 功能 | arch |
|------|------|:---:|
| `mhc_sinkhorn` | Sinkhorn-Knopp 迭代→双随机矩阵投影 | arch35 |
| `mhc_sinkhorn_backward` | 反向 | arch35 |
| `mhc_pre` | 计算 H_res/H_post 投影矩阵 + h_in | arch35 |
| `mhc_pre_backward` | 反向 | arch35 |
| `mhc_pre_sinkhorn` | Pre + Sinkhorn 融合 | arch35 |
| `mhc_pre_sinkhorn_backward` | 反向 | arch35 |
| `mhc_post` | Post Mapping + Res Mapping + 残差 | arch35 |
| `mhc_post_backward` | 反向 | arch22 + arch35 |
| `mhc_sinkhorn_common` | 公共头文件 | 无 op_kernel |

## 特征

- **唯一同时有 arch22+arch35 的算子**：`mhc_post_backward`
- **几乎全 A5**：9 个中有 8 个 arch35
- **Sinkhorn-Knopp 是核心算法**：迭代将混合矩阵投影到双随机流形
- 用于稳定深度网络信号传播、解决梯度消失/爆炸

## 来源
- Agent 分析 `mhc/` 全部 9 个算子目录
