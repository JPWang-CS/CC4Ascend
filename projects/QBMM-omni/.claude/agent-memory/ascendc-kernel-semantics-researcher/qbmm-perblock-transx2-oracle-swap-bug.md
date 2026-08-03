---
name: qbmm-perblock-transx2-oracle-swap-bug
description: QBMM-omni perblock oracle trans_x2=True 时多 swap x2f/x2sf 致数值错(K//gK==N//gN 方阵也不崩但数值转置); 已修(删两处 swap)
metadata:
  type: feedback
---

QBMM-omni `tests/golden.py` 的 `_compute_perblock_gb` 与 `benchmark_ref` perblock 分支 在 `trans_x2=True` 时错误 swap x2f/x2sf, 已于 2026-07-30 修复(删两处 `if c.trans_x2:` swap 块, 保留 trans_x1 swap)。

**Why(语义):** gen_data 把 scale 存成逻辑布局 `[K//gK,N//gN]`、x2 存逻辑 `[K,N]`; call_npu 才在喂 NPU 前 `.transpose(-1,-2)` 转成转置存储态(aclnn doc L659 契约: x2 shape (n,k)、x2Scale (ceil(n/128),ceil(k/128)) = NPU 期望转置存储)。oracle 读 gen_data 已是逻辑布局, 不该再 swap。对照 ops-nn `_compute_per_tile_int8`(:323-419 注释 L341-354): ops-nn host 约定是"scale 与 input 同转置态存储", 故 oracle 先 swap 回逻辑; 项目 gen_data 约定相反, swap 是抄来的残留错误。

**关键纠正(前一 agent 判断错):** 前一 agent memory 原写"方阵单例 swap 是 no-op、现有方阵仍 pass"——**错**。严格 numpy 验证: 方阵 M=128,N=512,K=512 时 K//gK==N//gN==4, swap 后 shape `[4,4]` 不崩(shape 兼容), 但 x2sf 随机生成不对称, swap 致数值转置, oracle 算成 `x1@x2^T`(转置乘, 完全错), max|Δ| vs 正确参考 = 6.46。即 shape-no-op ≠ value-no-op。故现网"perblock 方阵单例已通"若指真上板对拍 PASS, 必是误报(修前 oracle 错、NPU 对、bf16 dbr_max 门≈0.039 必破); 实际多半是 golden-only(只验 finite 不验数值) 或未真跑。修复让方阵 oracle 也变正确(max|Δ| vs 朴素参考 = 0.0)。

**How to apply:**
- 修法(已落地): 删 `_compute_perblock_gb` 与 `benchmark_ref` perblock 分支各一处 `if c.trans_x2: x2f/x2sf swap`(2 行 × 2 处)。保留 trans_x1 swap(perblock 恒 trans_x1=False, 无害; aclnn V5 G-B 强制 trans_x2=true, 见 perblock_tiling.cpp:103-114)。
- 修复点 file:line(修后): `_compute_perblock_gb` 注释 + `benchmark_ref` perblock 分支注释。
- 非方阵修前必崩: M=32,N=256,K=512 → `ValueError: operands could not be broadcast (32,1) (0,512)`(swap 后 x2sf[2,4]→repeat→[2,512], K-tile kt=2,3 越界取空)。修后 shape=[32,256] 正确。
- perblock groupSizeM 硬定死=1(perblock_tiling.cpp:91 + aclnn doc L659 双约束), 不存在 gM=128 合法 case; x1Scale shape=`[M, ceil(K/128)]` 与 M 无关(非 M/gM, 前一 agent 担心 ceil(M/gM) 是混淆 pergroup/MX 别处契约)。gen_data x1Scale=[M,K//gK] 正确匹配。
- pergroup `_compute_pergroup_kg` 注释"无 swap"正确, 不受此 bug 影响。
- 这是 oracle bug 不是 kernel bug: kernel-expert 已确认 v4_perblock.h AddBias 正确, golden `out+=bias` 语义正确, 问题只在 trans_x2 swap。

关联 [[qbmm-pergroup-int4-kg-semantics]]。独立 numpy 验证脚本: D:/Desktop/TMP/verify_perblock_oracle_fix.py + check_sym.py。
