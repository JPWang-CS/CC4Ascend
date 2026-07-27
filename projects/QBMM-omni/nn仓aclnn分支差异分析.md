# nn 仓 aclnn 分支差异分析

> 目的：系统对照 ops-nn 仓 aclnn 文档（V3/V4/V5/WeightNz 四份）描述的分支 / 量化类型 / 入参组合，与 omni-ops 已迁内容的差异，找出未迁移分支或语义不一致点。
> 方法：直接读盘 nn 仓 docs/*.md + omni-ops 源码 + 项目 golden.py，逐项核对。
> 最后更新：2026-07-27
> 本文档是分析报告，不是设计文档；设计在 `迁移计划.md` / `修改清单.md`。

## 0. 文档清单与口径

nn 仓 aclnn 文档四份（**仅这四份**，仓库无更高级别 docs/ 汇总）：

| 文档 | 路径 | 行数 | 产品支持段 |
|------|------|------|------------|
| aclnnQuantMatmulV3 | `ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV3.md` | ~1439 | 950PR/950DT / A3 / A2 / 200I500(×) / 推理系列(✓) / 训练系列(×) |
| aclnnQuantMatmulV4 | `ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md` | ~1531 | 同上 |
| aclnnQuantMatmulWeightNz | `ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md` | ~1541 | 同上 |
| aclnnQuantMatmulV5 | `ops-nn/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md` | ~1811 | 同上 |

**口径**：本项目 scope 是 **A2A3 only**（910B/910C, DAV_2201）。下文"已迁/未迁"判定基准 = A2A3 路径 + omni-ops 实际源码 + golden.py 实际验证。

**重要前置**：aclnn V3/V4 文档头部均标注 **"该接口后续版本会废弃，请使用最新 aclnnQuantMatmulV5 接口"**。omni-ops 仍迁了 V3/V4/WeightNz aclnn 入口（照 ai_infra_matmul 模式 + 历史兼容），这是设计决策不是遗漏。

---

## 1. A2A3 路径支持的量化模式（V5 doc 为权威）

V5 doc 把产品分四档：950PR/950DT / A3 / A2 / 推理系列。本项目 A2A3 = **A2 档**（含 A3，因 A3 共用 DAV_2201 路径）。

A2 + A3 档支持的量化模式（V5.md:19-29 / 616-743）：

| 模式 | 公式（A2/A3） | dtype 组合要点 |
|------|---------------|----------------|
| **K-G**（int8×int32 + fp32 x1Scale + uint64/int64 x2Scale + fp32 yOffset） | `out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale` | int8×int32, A8W4-int(MSD) |
| **K-G**（int4×int4 + fp32 scales + fp16 x2Offset, pertoken-pergroup 非对称） | `out = x1Scale * x2Scale * (x1 @ x2 - x1 @ x2Offset)` | int4×int4, K-G pergroup 非对称 |
| **K-C / K-T**（pertoken） | 多 bias 组合（int32 / bf16 / fp16 / fp32） | int8×int8 / int4×int4 |
| **T-C / T-T**（pertensor / perchannel） | 多 bias 组合 | int8×int8 / int4×int4 / int8×int32 |
| **G-B**（pergroup-perblock, int8×int8 + fp32 scales + fp32 bias） | `out = (x1 @ x2) * x1Scale * x2Scale + bias` | int8×int8, groupSize=[1,128,128] |
| **int32 out**（scale 不参与） | `out = x1@x2 (+ bias)` | int8×int8, bias int32/null |

A2/A3 **明确不支持**（V5 doc 标 950-only）：
- B-B（fp8/hifloat8 perblock）
- MX（e8m0 scale + fp8/fp4）
- T-CG（fp8×fp4 with yScale）
- fp8/hifloat8/fp4 dtype 输入
- FLOAT8_E8M0 scale

---

## 2. omni-ops 已迁内容对照（A2A3 视角）

### 2.1 aclnn 入口层（已迁，强证据）

| 入口 | omni-ops 位置 | nn doc 对应 | 状态 |
|------|---------------|-------------|------|
| `aclnnAiInfraQuantMatmulV5`（ND 统一入口） | `ai_infra_quant_batch_matmul_v4/op_api/aclnn_ai_infra_quant_matmul_v5.{h,cpp}` | V5.md | 已迁 |
| `aclnnAiInfraQuantMatmulWeightNz`（NZ 入口，含 V3/V4 分派） | `ai_infra_quant_batch_matmul_v3/op_api/aclnn_ai_infra_quant_matmul_v4.{h,cpp}` | WeightNz.md | 已迁 |
| `aclnnAiInfraQuantMatmulV3` / `V4`（旧接口兼容） | `ai_infra_quant_batch_matmul_v3/op_api/aclnn_ai_infra_quant_matmul_v3.h` / `v4.h` | V3.md / V4.md | 头文件已迁（impl 在 v4.cpp 共用） |

### 2.2 l0op + kernel 层（已迁，强证据）

| l0op | 落核 | omni-ops 位置 | 状态 |
|------|------|---------------|------|
| `l0op::AiInfraQuantBatchMatmulV3` | V3 kernel | `ai_infra_quant_batch_matmul_v3/op_kernel/` | 全套已迁 |
| `l0op::AiInfraQuantBatchMatmulV4` | V4 kernel | `ai_infra_quant_batch_matmul_v4/op_kernel/`（含 msd/perblock/pergroup） | 全套已迁 |

### 2.3 量化模式 × golden 覆盖矩阵（关键差异在此）

| 模式 | nn doc（A2A3）声明 | omni-ops kernel 已迁 | golden 已覆盖 | 差异 |
|------|---------------------|----------------------|---------------|------|
| pertensor（T-C） | ✓ | ✓ | ✓（ND dyn/stat + NZ dyn/stat/bf16-scale + int8 out） | 无差异 |
| perchannel（T-T） | ✓ | ✓ | ✓（ND dyn/stat + NZ dyn/stat/bf16-scale + tx1/tx2/tx1x2 + ibias + int8/int32 out） | 无差异 |
| pertoken（K-C/K-T） | ✓ | ✓ | ✓（ND + NZ + tx2 + ibias + fp16/bf16 out） | 无差异 |
| int32 out（raw matmul） | ✓ | ✓ | ✓（ND + NZ + K=1024 + ibias） | 无差异 |
| int4×int4 对称 | ✓ | ✓ | ✓（ND + NZ, int32-packed） | 无差异 |
| **G-B perblock**（int8 + fp32 block scales） | ✓（V5.md:645-665） | ✓（v4 perblock.h） | **单例**（M=128,N=512,K=512,gK=gN=128, ND, bf16 out） | **未泛化**：单 shape、单 groupSize、无 NZ、无 bias 变体 |
| **K-G pergroup 非对称**（int4×int4 + fp32 scales + fp16 offset） | ✓（V5.md:715-743） | ✓（v4 pergroup.h） | **零覆盖** | **未验证**：kernel 已迁但无 golden case |
| **K-G A8W4-int**（int8×int32 + fp32 x1Scale + uint64 x2Scale + fp32 yOffset） | ✓（V5.md:676 / K-G table） | ✓（v4 msd.h） | **disabled**（EH0012 yOffset） | **aclnn 约束阻塞** |
| **V4 核 NZ 路径**（WeightNz → a8w4 NZ） | ✓（WeightNz.md） | ✓（aclnn v4.cpp 含 WeightNz） | **零覆盖** | **被 a8w4 disabled 阻塞** |
| **fbias**（fp32 scale + fp32 bias + perchannel/pertoken） | ✓（V5.md:680-686 列出 bias BFLOAT16/FLOAT16/FLOAT32 组合） | ✓（V3 kernel 支持） | **disabled**（EZ0020 trans_quant_param 转 int64 后强制 bias=INT32） | **aclnn 约束阻塞** |

---

## 3. 未迁移分支清单（任务 B 核心产出）

按"是否真的没迁"vs"迁了但没验证"区分：

### 3.1 真正"未验证"（kernel 已迁，golden 零或弱覆盖）

| # | 分支 | 影响 | 优先级 |
|---|------|------|--------|
| U1 | **K-G pergroup 非对称**（int4×int4 + fp32 scales + fp16 offset, out=bf16/fp16） | kernel 已迁、aclnn 文档明确 A2A3 支持，零验证 = 沉默风险（kernel 可能有死代码 / TilingKey 路径未触发） | **高** |
| U2 | **G-B perblock 泛化**（不同 M/groupSize/NZ/bias 变体） | 单例通过≠perblock 通，verifier 会判弱证据 | **高** |
| U3 | **V4 核 NZ 路径**（a8w4 NZ） | 被 U4 (a8w4 disabled) 阻塞，解阻塞前无法验证 | **中（阻塞）** |

### 3.2 aclnn 约束阻塞（kernel 已迁，aclnn 入口被拒）

| # | 分支 | 阻塞码 | 解阻塞路径 |
|---|------|--------|------------|
| B1 | **a8w4-int**（含 ND + NZ） | EH0012 yOffset 当前版本不支持 | 待办-3：host + semantics 协作排查 |
| B2 | **fbias**（fp32 bias + fp32 scale） | EZ0020 bias=INT32（trans_quant_param 转 int64 后） | 待办-3：同上 |

### 3.3 真正"未迁"分支（A2A3 范围内）

**无。** A2A3 范围内 nn doc 声明支持的所有量化模式，omni-ops kernel 层都已迁移。差异全在验证层。

### 3.4 明确 scope-out（不属于"未迁"，是设计剔除）

| 分支 | 剔除原因 |
|------|----------|
| B-B（fp8/hifloat8 perblock） | 950-only（V5.md 标 950PR/950DT） |
| MX（e8m0 + fp8/fp4） | 950-only |
| T-CG（fp8×fp4 + yScale） | 950-only |
| fp8/hifloat8/fp4 dtype 输入 | 950-only |
| FLOAT8_E8M0 scale | 950-only |
| Atlas 推理系列产品路径（310P III 等） | A2A3-only scope，且 op-plugin/torch 无 310P 路径 |

---

## 4. 语义级差异高亮（"迁了但语义不一致"排查）

逐一核对，**未发现 omni-ops 已迁代码与 nn doc 在 A2A3 范围内有语义级不一致**。具体核对点：

| 核对点 | nn doc | omni-ops | 一致性 |
|--------|--------|----------|--------|
| V5 入口分派谓词（isA8W4 / IsA8W8Perblock / IsA4W4PergroupNonSym） | V5.md:676-743 | aclnn_ai_infra_quant_matmul_v5.cpp（已迁，谓词照搬） | 一致（设计文档 §3.1） |
| WeightNz 入口分派（isA8W4F / isA8W4I） | WeightNz.md | aclnn_ai_infra_quant_matmul_v4.cpp（已迁） | 一致 |
| pertoken_scale 约束（DT_FLOAT + 1D[M]） | V5.md（EZ0020/EZ0013） | checker 已保留 [[qbmm-v5-pertoken-scale-fp32]] | 一致 |
| data↔scale 转置一致 | V5.md:873 / WeightNz.md | checker `is_x_scale_same_transpose` 已保留 [[qbmm-data-scale-transpose-must-match]] | 一致 |
| group 推断 3 层（op-plugin 打包 / aclnn InferGroupSize / host AnalyzeGroupInfo） | V5.md:766-773 | 三层都已迁 [[qbmm-mx-groupsize-inference-fix]] | 一致 |
| int32-packed（int4 打 8 个 / int32 容器） | V5.md:690-700 | golden.py pack_int4_lastdim 照此 | 一致 |
| scale_generate（fp32 高 13/16 位掩码） | nn golden.py | golden.py `_scale_generate` 照搬 | 一致 |

**潜在风险点（非语义不一致，是验证缺口）**：

- pergroup 非对称的 x2Offset（FLOAT16）构造：golden.py 无对应构造逻辑，需 kernel-semantics-researcher 确认 oracle 公式（V5.md:727 给出 `out = x1Scale * x2Scale * (x1 @ x2 - x1 @ x2Offset)`）。
- A8W4-int 的 yOffset 数学语义：V5.md:709 注"值要求为 8*x2*x2Scale 并在第1维累加"，golden.py a8w4 分支已按此构造（line 395），但 disabled 状态下未上板验证。

---

## 5. 结论与待办接入

### 5.1 结论

- **A2A3 范围内，omni-ops 已迁内容与 nn aclnn 文档语义一致，无遗漏分支**。
- **真正缺口在验证层**：U1（K-G pergroup 零验证）/ U2（perblock 单例未泛化）/ U3（V4 核 NZ 阻塞）。
- **aclnn 约束阻塞**：B1（a8w4 EH0012）/ B2（fbias EZ0020）需待办-3 排查。
- **950-only 分支**全部 scope-out，设计阶段已剔除，非遗漏。

### 5.2 接入施工进度待办

差异清单已并入 `施工进度.md` 第 5 节：
- U1 → 待办"[任务 B 新发现] K-G pergroup 未迁移分支"
- U2 → 待办-4 perblock 泛化
- U3 → 待办"[任务 B 新发现] V4 核 NZ 路径"（被待办-3 阻塞）
- B1/B2 → 待办-3

### 5.3 优先级建议（重申）

1. U2 perblock 泛化（投入小、收益直接、verifier 可升证据等级）
2. U1 K-G pergroup golden 补全（kernel 已迁、文档明确、零验证 = 沉默风险）
3. B1/B2 aclnn 约束排查（投入大、需 host+semantics）
4. U3 V4 核 NZ（被 B1 阻塞）

---

## 附录：文档路径与行号锚点

| 锚点 | 路径 | 行号 |
|------|------|------|
| V5 A2/A3 量化模式声明 | `ops-nn/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md` | 19-29 |
| V5 K-G 量化公式（int8×int32 / int4×int4） | 同上 | 33-47 / 715-743 |
| V5 K-C/K-T 公式 | 同上 | 49-71 |
| V5 T-C/T-T 公式 | 同上 | 73-95 |
| V5 G-B 公式 | 同上 | 99-107 / 645-665 |
| V5 产品支持矩阵 | 同上 | 7-14 |
| V5 A2/A3 公共约束 | 同上 | 616-641 |
| V5 A2/A3 T-C/T-T/K-C/K-T dtype 组合表 | 同上 | 672-711 |
| V5 A2/A3 K-G dtype 组合表 | 同上 | 718-742 |
| V3 文档（即将废弃） | `ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV3.md` | 全文 |
| V4 文档（即将废弃） | `ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md` | 全文 |
| WeightNz 文档 | `ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md` | 全文 |
