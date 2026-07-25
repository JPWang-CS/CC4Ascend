# 自定义AiInfraMatmul算子新增batch_invariant参数详细设计说明书

<center><strong>修订记录</strong></center>

<div>

| 日期 | 修订版本 | 修改描述 | 作者 |
| :---: | :---: | :---: | :---: |
| 2026-07-21 | 1.0 | 详细设计初稿 | yiqiao/w00939120 |

</div>

> 文档说明：本文档为详细设计，撰写责任人：算子MDE。需求编号为示例占位，待替换为需求管理平台实际编号。

# 1 关联需求

> **章节说明**：
> 附上该需求关联需求管理平台的需求信息和概要设计的链接。

| **需求编号** | 需求标题 | 需求链接 | 概要设计链接 | 修订版本 |
| ---------------- | ------------------------------------ | ------------------------------------------------------------ | ------------ | -------- |
| US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 | https://clouddragon.huawei.com/cloudreq-obp/third/TBD | xxx | 1.0 |

# 2 接口实现设计

> **章节说明**：接口实现的设计，按需填写，新特性引入的接口变化部分需说明关联特性。

## 2.1 PTA接口实现

> **章节说明**：说明实现逻辑，非完整代码实现。如PTA接口有新增修改，需在此处说明。

在既有 PTA 接口 `npu_ai_infra_matmul` 上新增可选参数 `batch_invariant`（默认 `False`）；**不新增独立 `_v2` op**，现有调用方不传该参数时行为不变。实现内通过运行期符号探测判定 V2 aclnn 接口可用性：V2 可用时调用 V2 aclnn 并将 `batch_invariant` 透传至算子属性；V2 不可用时回退至 V1 aclnn（此时 `batch_invariant` 被丢弃，attr 取默认 `false`）。该符号探测方式与仓内 `aclnnAiInfraMhcSandwichNormPostPreonlyV2` 的 V2 增量方式一致。

**函数原型**

```
npu_ai_infra_matmul(Tensor self, Tensor mat2, *, int? cube_math_type=0, bool batch_invariant=False) -> Tensor
```

> 不新增 `_v2` op；`batch_invariant` 作为既有 op 的可选 kwargs，默认 `False`，schema 向后兼容。

**参数说明**

- **self**（`Tensor`）：必选参数，matmul 输入左矩阵，2D。
- **mat2**（`Tensor`）：必选参数，matmul 输入右矩阵，ND 格式仅支持 2 维，NZ 格式仅支持 4 维。
- **cube_math_type**（`int?`）：可选参数，Cube 计算逻辑，当前仅支持 0。
- **batch_invariant**（`bool`）：可选参数，批量不变（确定性）开关，默认 `False`。取值为 `True` 时强制 `SpecialOpti=BASE`（关闭 K 轴错峰），保证 batch 间及多次运行结果逐字节一致；取值为 `False` 时行为与不传一致。

**返回值说明**

- **Out**（`Tensor`）：计算结果，支持 `float16`、`bfloat16`、`float32`，数据类型与 `self` 一致，ND 格式，不支持非连续 Tensor。

**PTA接口实现变更，需体现变化部分：**

| **需求编号** | 需求标题 | 变化点 |
| ---------------- | ------------------------------------ | ---------- |
| US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 | 既有 `npu_ai_infra_matmul` 新增 `batch_invariant` 可选形参（不新增 `_v2` op），binding 内运行期探测 V2 aclnn 可用性并路由 |

## 2.2 aclnn接口实现

> **章节说明**：说明实现逻辑，非完整代码实现。新特性引入的接口变化部分需说明关联特性。

V1 aclnn 接口（`aclnnAiInfraMatmul`、`aclnnAiInfraMatmulWeightNz`）签名保持不变；新增 V2 接口（`aclnnAiInfraMatmulV2`、`aclnnAiInfraMatmulWeightNzV2`），在 V1 入参基础上增加 `batchInvariant` 入参。V2 实现复用 V1 构图逻辑，增加 `SetBatchInvariant` 调用。

### 2.2.1 aclnnAiInfraMatmulV2接口

**函数原型**

```cpp
aclnnStatus aclnnAiInfraMatmulV2GetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat2,
  aclTensor       *out,
  int8_t           cubeMathType,
  bool             batchInvariant,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)

aclnnStatus aclnnAiInfraMatmulV2(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

**参数说明**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| self | 输入 | matmul 左矩阵 | 数据类型需与 mat2 满足推导规则 | BFLOAT16、FLOAT16、FLOAT32 | ND | 2 | √ |
| mat2 | 输入 | matmul 右矩阵 | reduce 维度需与 self 相等 | BFLOAT16、FLOAT16、FLOAT32 | ND | 2 | √ |
| out | 输出 | matmul 输出矩阵 | 数据类型与推导结果一致 | BFLOAT16、FLOAT16、FLOAT32 | ND | 2 | - |
| cubeMathType | 输入 | Cube 计算逻辑 | 当前仅支持 0（KEEP_DTYPE）| INT8 | - | - | - |
| batchInvariant | 输入 | 批量不变（确定性）开关 | `True` 强制关闭 K 轴错峰（SpecialOpti=BASE）；`False` 行为同 V1 | BOOL | - | - | - |
| workspaceSize | 输出 | Device 侧 workspace 大小 | - | - | - | - | - |
| executor | 输出 | op 执行器 | - | - | - | - | - |

**实现逻辑：**

1. 内部通过 `MatmulCommonProcess` 路由至自定义 AiInfraMatmul 算子，路由逻辑与 V1 一致。
2. 相比 V1 增加 `matmulGraph->SetBatchInvariant(batchInvariant)` 调用，将 `batchInvariant` 写入算子属性（index 4），由 op_host `DoTilingKey` 决定是否关闭 K_SHIFT。
3. V2 为 V1 的超集：`batchInvariant=false` 时数值结果与 V1 完全一致。

> **ABI 说明**：V1 `aclnnAiInfraMatmul` 签名未变，V2 为纯新增符号，现有 V1 调用方无需修改。

### 2.2.2 aclnnAiInfraMatmulWeightNzV2接口

**函数原型**

```cpp
aclnnStatus aclnnAiInfraMatmulWeightNzV2GetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat2,
  aclTensor       *out,
  int8_t           cubeMathType,
  bool             batchInvariant,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)

aclnnStatus aclnnAiInfraMatmulWeightNzV2(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

**参数说明**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续tensor |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| self | 输入 | matmul 左矩阵 | self 不转置为（m，k），转置为（k，m） | BFLOAT16、FLOAT16 | ND | 2 | √ |
| mat2 | 输入 | matmul 右矩阵（NZ）| 不转置为（n1，k1，k0，n0），k0=16，n0=16；转置为（k1，n1，n0，k0）| BFLOAT16、FLOAT16 | NZ | 4 | - |
| cubeMathType | 输入 | Cube 计算逻辑 | 当前仅支持 0（KEEP_DTYPE）| INT8 | - | - | - |
| batchInvariant | 输入 | 批量不变（确定性）开关 | `True` 强制关闭 K 轴错峰；`False` 行为同 V1 | BOOL | - | - | - |
| out | 输出 | matmul 输出矩阵 | （m，n），n 与 mat2 的 n1、n0 满足 ceil(n/n0)=n1 | BFLOAT16、FLOAT16 | ND | 2 | - |
| workspaceSize | 输出 | Device 侧 workspace 大小 | - | - | - | - | - |
| executor | 输出 | op 执行器 | - | - | - | - | - |

> **数据类型约束**：WeightNz（NZ）通路仅支持 BFLOAT16、FLOAT16，不支持 FLOAT32。FLOAT32 输入在 PTA 层即被引导至 ND 通路（`aclnnAiInfraMatmul`）执行。

**实现逻辑：** 与 2.2.1 一致，经 `BuildMatMulWeightNzGraph → ExecMmOpWithBias` 将 `batchInvariant` 透传至算子属性。

**aclnn接口实现变更，需体现变化部分：**

| **需求编号** | 需求标题 | 变化点 |
| ---------------- | ------------------------------------ | ------------------------------ |
| US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 | V1 `aclnnAiInfraMatmul`、`aclnnAiInfraMatmulWeightNz` 保持不变；新增 `aclnnAiInfraMatmulV2`、`aclnnAiInfraMatmulWeightNzV2` 接口，增加 `batchInvariant` 入参 |

## 2.3 HCCL C++接口实现

不涉及。本算子非 HCCL 算子。

## 2.4 算子信息库

不涉及。本算子仅支持 AclGraph、不走 GE 图模式（见 §2.5），`batch_invariant` 由 aclnn launcher 的 `OP_ATTR` 动态下发，op_host 运行期读取，OpDef 无需新增属性。

## 2.5 图模式设计

本算子不支持 GE 图模式，仅支持 AclGraph 图模式。V2 接口同样仅支持 AclGraph，通过 aclnn 两段式接口调用，算子内部不做额外图模式适配。

**实现变更，需体现变化部分:**

| **需求编号** | 需求标题 | 变化点 |
| ---------------- | ------------------------------------ | ------ |
| US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 | V2 接口仅支持 AclGraph，不支持 GE 入图（同 V1）|

# 3 总体设计

## 3.1 交付方式

| 类型         | 描述 | 备注 |
| ------------ | ---- | ---- |
| 交付内容     | 自定义算子包（增量特性）| omni-ops 仓，推理场景 |
| 代码承载     | omni-ops 仓 `inference/ascendc/src/ops-nn/matmul/ai_infra_matmul`、`inference/ascendc/src/ops-nn/matmul/common/op_host/op_api`、`torch_ops_extension/omni_custom_ops` | 在既有 AiInfraMatmul 上增量 |
| 期望交付时间 | | |

## 3.2 交付件汇总

| **序号** | **交付件**                   | **是否需要** | **涉及变动** | **备注**               |
| -------- | ---------------------------- | ------------ | ------------ | ---------------------- |
| 01       | pta接口适配                  | 是 | 是 | 既有 `npu_ai_infra_matmul` 新增 `batch_invariant` 形参（不新增 `_v2` op） |
| 02       | pta接口文档                  | 是 | 是 | 更新 `npu_ai_infra_matmul` 文档，新增 `batch_invariant` 参数说明 |
| 03       | aclnn接口适配  | 是 | 是 | 新增 V2 两个 aclnn 接口，V1 保持不变 |
| 04       | aclnn接口文档  | 是 | 是 | 新增 V2 aclnn 文档 |
| 05       | HCCL CANN接口适配 | 不涉及 | 不涉及 | 非HCCL算子 |
| 06       | HCCL CANN接口文档 | 不涉及 | 不涉及 | 非HCCL算子 |
| 07       | GE图模式适配                 | 不涉及 | 不涉及 | 不支持GE入图 |
| 08       | AclGraph图模式适配           | 是 | 是 | 仅支持AclGraph |
| 09       | 算子原型                     | 不涉及 | 不涉及 | OpDef 无需新增（AclGraph-only，attr 由 launcher 动态下发）|
| 10       | OpDef定义（信息库）          | 不涉及 | 不涉及 | 同上，OpDef 无需新增属性 |
| 11       | 算子tiling函数               | 是 | 是 | `DoTilingKey` 增加 batch_invariant 覆盖逻辑 |
| 12       | 算子kernel实现               | 不涉及 | 不涉及 | kernel 无改动 |
| 13       | 算子二进制配置               | 不涉及 | 不涉及 | 无新增模板 |
| 14       | 算子inferShape/inferDataType | 是 | 是 | 删除遗留 input_size/hidden_size 分支 |
| 15       | 图融合pass                   | 不涉及 | 不涉及 | |

## 3.3 特性策略与差异点

### 3.3.1 增量策略

aclnn 层采用 V2 增量：V1 aclnn 接口（`aclnnAiInfraMatmul`、`aclnnAiInfraMatmulWeightNz`）签名与实现保持不变，新增 V2 aclnn 接口承载 `batch_invariant`，V2 接口置于独立 `aclnn_ai_infra_matmul_v2.h/.cpp`（与仓内 `aclnnAiInfraFusedInferAttentionSinkV2` 等 V2 算子惯例一致）。torch 层共用既有 `npu_ai_infra_matmul` op（不新增 `_v2` op），仅新增可选形参 `batch_invariant`（默认 `False`，schema 向后兼容）；binding 内通过运行期符号探测判定 V2 aclnn 可用性：V2 可用时调用 V2 aclnn 并透传 `batch_invariant`；V2 不可用时回退至 V1 aclnn。

### 3.3.2 与 NN 仓 matmulV3 的差异点

| 差异项 | NN 仓 matmulV3 | omni-ops AiInfraMatmul（本特性）| 原因 |
| ------ | -------------- | ------------------------------ | ---- |
| batch 确定性开关 | 无 `batch_invariant` | 新增 `batch_invariant`（V2 op 与 aclnn）| omni 专属特性，关闭 K_SHIFT 以保证 batch 间结果一致 |
| 确定性机制 | GE 层 `GetDeterministic()==1` 路由至 `DETERMINISTIC_SPLIT_K`（split-K 归约维度）| 算子属性 `batch_invariant` 路由至 `SpecialOpti=BASE`（K_SHIFT 错峰维度）| 两者为相互独立的确定性维度；omni 采用算子属性方式，避免依赖 GE deterministic |
| K_SHIFT 错峰来源 | AOE 离线调优知识库按 shape 决定（`CheckAoeTilingEnable` 解析 tilingEnable 万位）| 机制同 NN 仓；`batch_invariant=True` 时强制覆盖为 BASE | 见 §3.4 |
| aclnn 接口范围 | 仅 V1 | V1 保持不变并新增 V2 | 保持 V1 ABI 兼容 |

### 3.3.3 torch `_v2` op 路由实现说明

`npu_ai_infra_matmul` 的 NPU 实现中，通过 `GetOpApiFuncAddr`（`EXEC_NPU_CMD_V1` 内部所用符号探测函数，定义于 `csrc_base/ops_common.h`）按 ND 与 NZ 两个 aclnn 入口分别探测 V2 符号是否存在：存在则调用 `aclnnAiInfraMatmulV2`（或 `aclnnAiInfraMatmulWeightNzV2`）并透传 `batch_invariant`；不存在则回退至对应 V1 aclnn。`batch_invariant=false` 时数值结果与不传一致。不可使用 omni 未导出的 `check_aclnn_kernel_available`。

## 3.4 TilingKey设计

> **章节说明**：说明TilingKey的设计原则，使用表格形式给出各个TilingKey的含义或者各个字段的含义。

TilingKey 通过 6 个模板参数组合编码（字段定义不变）：

| 字段 | 位宽 | 取值范围 | 含义 |
| ---- | ---- | -------- | ---- |
| LOADMODE | 4bit | 0=BASE_FULLLOAD, 1=AL1_FULLLOAD, 2=BL1_FULLLOAD | L1加载策略 |
| SPLITCOREMODE | 8bit | 0=BASE_SPLIT_K（omni 仅使能此项）| 核间切分策略 |
| FIXOPTI | 4bit | 0=BASE_FIXOPTI, 1=BASE_ENABLE_ALIGNOUT, 2=VEC_NZ2ND_UNALIGNOUT | 输出优化策略 |
| MIXND2NZ | 4bit | 0=MIXND2NZ_TRUE, 1=MIXND2NZ_FALSE, 2=MIXND2NZ_TRUE_PARALLEL | ND转NZ策略 |
| SPECIALOPT | 4bit | 0=K_NOT_SHIFT, 1=K_SHIFT | 特殊优化（K轴错峰）|
| FP32ADDMM | 4bit | 0=DISABLE, 1=ENABLE | FP32 addmm内存优化 |

> **本特性变更**：op_host `DoTilingKey` 中，当 `batch_invariant=true` 时强制置 `tilingEnableSpecialOpti=BASE`（即 SPECIALOPT 恒为 K_NOT_SHIFT），覆盖任何来源（AOE 知识库等）的 K_SHIFT 决策。覆盖点位于 tilingKey 组装的最终步骤，保证单点生效。

## 3.5 模板列表

> `batch_invariant=true` 时 SPECIALOPT 强制为 K_NOT_SHIFT，K_SHIFT 模板不可达；`batch_invariant=false` 时模板列表与 V1 一致。

| 规格 | 应用场景 | 模板 | 核间切分 | 核内切分 | 备注 |
| ---- | -------- | ---- | -------- | -------- | ---- |
| ND-FP16/BF16/FP32 | batch_invariant=true | Base FullLoad + NoPreload（强制）| BASE_SPLIT_K | MatmulBaseBlock + MM_CFG_NO_PRELOAD | K_SHIFT 关闭 |
| NZ（mat2 为 NZ）| batch_invariant=true | Base FullLoad + NoPreload（强制）| BASE_SPLIT_K | MatmulBaseBlock + MM_CFG_NO_PRELOAD | K_SHIFT 关闭，仅 FP16/BF16 |
| ND/NZ | batch_invariant=false | 同 V1 模板列表 | BASE_SPLIT_K | 见 V1 设计 | K_SHIFT 仍可由 AOE 知识库触发 |

## 3.6 代码结构设计

> **章节说明**：host/kernel/common等代码目录与文件结构设计。需体现新特性引入的代码变化部分。

| 目录/文件 | 内容 | 本特性变化点 | **需求编号** | 需求标题 |
| --------- | ---- | ------------ | ------------ | -------- |
| `op_api/aclnn_ai_infra_matmul.h` | aclnn V1 接口声明 | V1（`aclnnAiInfraMatmul`、`aclnnAiInfraMatmulWeightNz`）声明保持不变 | - | - |
| `op_api/aclnn_ai_infra_matmul.cpp` | aclnn V1 接口实现 | V1 实现保持不变 | - | - |
| `op_api/aclnn_ai_infra_matmul_v2.h` | aclnn V2 接口声明 | 新增 V2（`aclnnAiInfraMatmulV2`、`aclnnAiInfraMatmulWeightNzV2`）声明，增加 `batchInvariant` 入参 | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `op_api/aclnn_ai_infra_matmul_v2.cpp` | aclnn V2 接口实现 | 新增 V2 实现（复用 V1 构图逻辑，ND 路径 `SetBatchInvariant`、WeightNz 路径 `BuildMatMulWeightNzGraph` 透传 `batchInvariant`）| US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `op_host/ai_infra_matmul_def.cpp` | OpDef 定义 | 不涉及改动（AclGraph-only，`batch_invariant` 由 launcher `OP_ATTR` 动态下发，无需 OpDef 声明）| - | - |
| `op_host/ai_infra_matmul_common.h` | 公共定义 | `AiInfraMatmulArgs` 增加 `bool batchInvariant` 字段及索引常量 | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `op_host/ai_infra_matmul_base_tiling.cpp` | 基础 tiling 实现 | `GetDtype` 读取 attr[4]（带 attrNum 守卫）；`DoTilingKey` 增加 `batch_invariant` 覆盖 | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `op_host/ai_infra_matmul_infershape.cpp` | inferShape | 删除遗留 input_size/hidden_size 分支（新增第 5 属性会激活该死分支，导致 x2 的 K 轴被错误改写）| US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `common/op_host/op_api/matmul.cpp` | mm 计算图 | `AiInfraMatmulCommon` 的 `OP_ATTR` 始终将 `batchInvariant` 作为第 5 个属性（index 4）写入 | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `common/op_host/op_api/matmul.h` | mm 计算图声明 | 4 个 `AiInfraMatmul*` 变体增加 bool 入参 | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `common/op_host/op_api/matmul_util.cpp` | mm util | 读取 `MmOpInfo.batchInvariant`；`ExecMmOpWithBias`、`GetMatMulOp` 透传 | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `common/op_host/op_api/matmul_util.h` | mm util 声明 | `MmOpInfo` 增加字段；新增 `MatmulGraphImpl::SetBatchInvariant`；`ExecMmOpWithBias` 签名增加入参 | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `common/op_host/op_api/batch_matmul_util.cpp` | BatchMatmul | `TransBmm2Mm` 固定不传递 batch_invariant | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `torch_ops_extension/.../csrc_base/ops_def_registration.cpp` | torch schema | 既有 `npu_ai_infra_matmul` 的 `m.def` 新增 `bool batch_invariant=False` 形参（不新增 `_v2` op） | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `torch_ops_extension/.../ai_infra_matmul/csrc/npu_ai_infra_matmul.cpp` | torch binding | 既有 `npu_ai_infra_matmul` 实现新增 `batch_invariant` 形参；内部用 `GetOpApiFuncAddr` 探测 V2/V1 aclnn 符号按 ND/NZ 路由（注册名不变） | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |
| `op_kernel/*` | kernel | 无修改 | - | - |
| `tests/` | UT/ST | 新增 V2 用例与 batch 一致性 ST | US-TBD-1 | AiInfraMatmul 新增 batch_invariant 参数 |

# 4 模板设计

> 本特性不新增模板，仅改变模板选择行为。原模板的完整计算流程、Buffer 设计、Kernel 伪代码参见 AiInfraMatmul 既有设计与 NN 仓 matmulV3 详细设计文档。

## 4.1 batch_invariant 对模板选择的影响

Host 侧 tiling 流程不变（GetPlatformInfo → GetShapeAttrsInfo → DoOpTiling → DoLibApiTiling → DoTilingKey）。本特性唯一的行为改动位于 `DoTilingKey`（tilingKey 组装的最终步骤）：

```cpp
void AiInfraMatmulBaseTiling::DoTilingKey() {
    uint64_t disableMixNd2nz = static_cast<uint64_t>(MixNd2NzType::NO_ND2NZ);
    if (args_.batchInvariant) {
        tilingEnable_.tilingEnableSpecialOpti = TilingEnableSpecialOpti::BASE;  // 强制 K_NOT_SHIFT
    }
    tilingKey_ = GET_TPL_TILING_KEY(tilingEnable_.tilingEnableFullLoad,
                                    tilingEnable_.tilingEnableSplitCore,
                                    tilingEnable_.tilingEnableFixOpti, disableMixNd2nz,
                                    tilingEnable_.tilingEnableSpecialOpti, TilingEnableFp32Addmm::FALSE);
}
```

- `batch_invariant=false`：不触发覆盖，SPECIALOPT 由既有逻辑（AOE 知识库等）决定，行为同 V1。
- `batch_invariant=true`：SPECIALOPT 恒为 K_NOT_SHIFT，编译期模板必走 `MatmulBaseKernel + MM_CFG_NO_PRELOAD`，K_SHIFT 错峰（各核不同 K 起点）不生效，batch 间结果逐字节一致。

`batch_invariant` 读取带 `GetAttrNum()>=5` 守卫，V1（4 attrs）与 V2（5 attrs）均安全。

### 4.1.1 Tiling实现设计

`MatmulTilingData` 结构体不变，无新增字段。`batch_invariant` 仅作为 host 侧 tilingKey 覆盖标志，不进入 tilingData 二进制布局，kernel 侧无改动。

### 4.1.2 Buffer设计 / Kernel设计 / 异常场景

均与 V1 一致，本特性不改变 Buffer、Kernel 及异常处理逻辑。

### 4.1.3 支持确定性计算设计

`batch_invariant=true` 即为本算子的确定性开关：强制关闭 K_SHIFT 错峰，消除各核 K 方向累加顺序差异；BASE_SPLIT_K 模式下 batch 间及多次运行结果 MD5 逐字节一致。

### 4.1.4 精度分析及设计

`batch_invariant` 不改变算子数学正确性，精度继承原算子标准。`batch_invariant=true` 与 `false` 在相同输入下结果差异不超过 2 ULP，仅 batch 一致性增强。

### 4.1.5 性能分析及设计

`batch_invariant=true` 在 K_SHIFT 触发 shape 下放弃 K_SHIFT 带宽收益以换取确定性；非 K_SHIFT shape 性能不受影响。

# 5 HCCL算子详细实现

不涉及。本算子非 HCCL 算子。

# 6 维测设计

**可测试性**：
- UT：op_host tiling UT 覆盖 `batch_invariant=true` 与 `false` 两路径（V1 为 4 attrs、V2 为 5 attrs），校验 `DoTilingKey` 输出的 SPECIALOPT 位；op_api UT 覆盖 V2 两个 aclnn 入口；infershape UT 覆盖删除遗留分支后的行为。
- ST：Python 层 ST 通过 `npu_ai_infra_matmul(…, batch_invariant=True)` 验证 `batch_invariant=true` 在 K_SHIFT 触发 shape（如 768×768×768）下 batch 间及多次运行结果 MD5 逐字节一致；`false` 与不传一致；V2 aclnn 符号未导出时 binding 回退至 V1 aclnn。
- 测试脚本 `test_matmul_golden.py` 已支持双路径（每用例执行 BI=true 与 false）；V2 aclnn 符号存在时 true/false 均经 V2 aclnn，仅当 V2 符号缺失时才回退 V1 aclnn。

**可维护性**：V1 全链路保持不变，V2 为纯增量，后续 NN 仓 matmulV3 更新不冲突。

# 7 资料设计

| 资料 | 变化 | 备注 |
| ---- | ---- | ---- |
| aclnnAiInfraMatmulV2 接口文档 | 新增 | docs/aclnnAiInfraMatmulV2.md |
| aclnnAiInfraMatmulWeightNzV2 接口文档 | 新增 | docs/aclnnAiInfraMatmulWeightNzV2.md |
| aclnnAiInfraMatmul、aclnnAiInfraMatmulWeightNz 文档 | 不变 | V1 aclnn 文档保持不变 |
| npu_ai_infra_matmul 文档 | 更新 | docs/npu_ai_infra_matmul.md 新增 `batch_invariant` 参数说明（不新增 `_v2` 文档）|
