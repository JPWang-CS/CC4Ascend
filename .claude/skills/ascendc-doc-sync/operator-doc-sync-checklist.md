# 算子变更跨仓文档同步排查清单

算子的语义/规格/数据布局/量化模式变更（如量化 scale 加 batch 维、新增 dtype、改 shape 约束、改转置规则）后，**同一份变更会散射到多个仓的多个独立文档产物**。本清单按仓逐项列出需排查的文档/注释产物，供变更落地后逐项核对。

> 来源：QBMM-Batch（quantBatchMatmul MX scale 加 batch 维）项目教训——该项目 aclnn .md 已于 2026-06-30 同步，但 `op_graph/*_proto.h` 注释漏到 2026-07-11；排查时还发现 ops-tensor kernel doc、op-plugin 对外 doc、examples 数据构造注释等都是独立产物线。

## 核心原则（先记这三条，免漏）

1. **各产物独立**：`proto.h` 注册注释、`aclnn*.md`、`README.md`、ops-tensor kernel doc、op-plugin torch_npu doc 是各自独立的文档产物，**改一处不会自动同步其余**。变更后必须逐项排查。
2. **一个 op_dir 多份 aclnn doc**：ops-nn 一个算子目录下 `docs/` 可含多份 aclnn API 文档（如 `quant_batch_matmul_v3/docs/` 同时有 `aclnnQuantMatmulV3.md`/`aclnnQuantMatmulV4.md`/`aclnnQuantMatmulWeightNz.md`；`quant_batch_matmul_v4/docs/` 有 `aclnnQuantMatmulV5.md`）。aclnn API 入口与 proto 文件非 1:1（两级 dispatch，见 agent memory `qbmm-v3-v4-directory-split.md`）。**遍历该 dir 下所有 `aclnn*.md`**，别只改命中的那份。
3. **兄弟仓易漏**：ops-tensor（kernel/tensor-api）与 op-plugin（PyTorch 对外）是 ops-nn 之外的独立仓，排查时别只盯 ops-nn。

## ① ops-nn（算子定义仓，主战场）

算子目录 `<repo>/matmul/<op>/`（或 `<repo>/<类>/<op>/`）下：

| 产物 | 路径模式 | 何时需同步 |
|---|---|---|
| GE 算子注册注释（规格本体） | `op_graph/*_proto.h`（REG_OP 上方 `@par Inputs/Attributes/Outputs/Constraints`、dtype 组合表、shape 表） | 任何输入/属性/输出/shape/dtype/约束/量化语义变更 |
| aclnn API 文档 | `docs/aclnn*.md`（**遍历该 dir 全部**；含参数表、计算公式、约束说明、dtype/shape 组合表、示例代码） | 同上 |
| README | `README.md` | 接口能力概述、支持矩阵变更 |
| aclnn 适配 / checker 源码注释 | `op_api/*.cpp/.h`（含 `*_common.h` 里的常量/上限、checker 校验规则注释） | 常量/上限/校验规则变更（注释里解释魔法数） |
| tiling / checker / infershape 源码注释 | `op_host/**`（`arch20`/`arch35` checker、tiling、`*_infershape.cpp`、`*_def.cpp`） | tiling/checker/infer 逻辑或约束变更 |
| kernel 源码注释 | `op_kernel/arch*/**` | kernel 数据流/布局/同步变更 |
| 示例代码（数据构造注释） | `examples/test_aclnn_*.cpp`（多变体） | 输入数据布局/shape 变更（如 scale 加 batch，示例的 scale 构造也要跟） |
| 融合 pass / 框架对接 | `op_graph/fusion_pass/*`、`framework/*onnx_plugin.cpp` | 融合/对接逻辑或匹配条件变更 |

## ② ops-tensor（kernel / tensor-api 仓，兄弟仓）

| 产物 | 路径模式 | 何时需同步 |
|---|---|---|
| kernel 设计文档 | `docs/API/**/kernel_*.md`、`block/*.md`（如 `kernel_qbmm_mx.md`、`block_mmad_qbmm_mx.md`、`block_scheduler_qbmm_mx.md`） | kernel 数据流/batch 维处理/量化通路变更 |
| kernel 源码注释 | `include/blaze/**/kernel_*.h`、`block_*.h` | 同上 |

> 注：ops-tensor 的 kernel doc 可能含 `_without_batch` 等后缀变体（如 `kernel_qbmm_mx_without_batch.md`），batch 相关变更要核对是否波及这些变体。

## ③ op-plugin（PyTorch 对外仓）

| 产物 | 路径 | 决策 |
|---|---|---|
| 对外用户 API 文档 | `docs/zh/custom_APIs/torch_npu/torch_npu-<op>*.md`（含融合变体，如 `npu_quant_matmul.md` + `_gelu`/`_reduce_sum`/`add_quant_matmul` 等，多个） | **按对外文档策略单独定**：对外 doc 可能刻意维持 legacy、不补 MX/内部特性（见 agent memory `opplugin-doc-not-touched-mx.md`）。决策要显式记录，别默认跟着 ops-nn 改 |

## ④ ops-transformer（官方规格仓）

| 产物 | 路径 | 何时需同步 |
|---|---|---|
| 官方 AscendC 规格 / context 文档 | `docs/zh/**`（量化介绍、数据类型、非连续的Tensor、数据格式 等） | 规格/术语本身演化时（罕见；通常算子项目不动这里） |

## ⑤ 项目工作目录 `projects/<project>/`

| 产物 | 注意 |
|---|---|
| `需求.md` / `需求分析.md` / `实施方案.md` / `代码修改说明.md` / `代码评审意见.md` / `施工进度.md` | **易 stale**：方向/状态描述常滞后于代码，别盲信其描述的当前状态（见 agent memory `qbmm-batch-status.md` 的 "Doc trap"）。变更落地后同步更新；当代码与文档冲突时以代码为准 |

## 典型漏改模式（前车之鉴）

- **proto.h 注释 vs aclnn .md 不同步**：QBMM 项目 aclnn .md（2026-06-30）已补 MX scale-batch，`op_graph/*_proto.h` 注释漏到 2026-07-11——两者是独立产物线。
- **一个 dir 多份 aclnn doc 只改一份**：`quant_batch_matmul_v3/docs/` 有 V3/V4/WeightNz 三份，易只改命中的那份。
- **ops-tensor kernel doc 在兄弟仓**：`kernel_qbmm_mx.md` / `kernel_qbmm_mx_without_batch.md` 在 ops-tensor，排查别只盯 ops-nn。
- **examples 数据构造注释**：scale 布局变更后，`examples/test_aclnn_*.cpp` 里 scale 的构造逻辑也要跟，否则示例与文档不符。
- **op-plugin 对外 doc 决策未显式记录**：改了 ops-nn 就顺手改 op-plugin，可能违反"对外维持 legacy"策略；反之漏改也可能。决策要写进记忆。

## 使用方式

变更落地后、提交前，按 ①→⑤ 逐仓过表，对每类产物判断"本次变更是否波及"：波及则改、不波及则跳过并在 PR/记忆里说明跳过原因（尤其 op-plugin 对外 doc 的"不改"决策）。
