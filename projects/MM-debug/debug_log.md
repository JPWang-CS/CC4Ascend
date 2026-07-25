# AiInfraMatmul fp32+NZ 问题排查与修复

## 问题描述
omni 仓 AiInfraMatmul 算子在左右矩阵为 fp32、右矩阵为 FRACTAL_NZ 格式时，被人为屏蔽。
需要找出屏蔽原因、定位根因并修复，使 fp32+NZ 在 A2A3 平台正常工作。

## 全链路追踪 (fp32+NZ, M=128, K=8192, N=128, cubeMathType=0)

### torch 入口
`npu_ai_infra_matmul.cpp:98` → 检测 mat2 为 NZ → `aclnnAiInfraMatmulWeightNz`

### aclnn 层
```
CheckWeightNzParam → 通过
BuildMatMulWeightNzGraph:
  → GetWeightNzShape(fp32, c0=8): [16,512,16,8]
  → SetTensorToNZFormat: storageFormat=NZ
  → ExecMmOpWithBias
```

### ExecMmOpWithBias (common/matmul_util.cpp)
```
GetMatmulOpInfo:
  SetMmSupportDType: fp32+fp32+KEEP_DTYPE → 无变化, 全保持 DT_FLOAT
  SetMmSupportFormat: self=ND, mat2=NZ → support_info: self=ND, mat2=NZ, out=ND
  SetMatmulOpSupportInfo: isNdNzIn=true → mat2_format=NZ, output_dtype=DT_FLOAT
  
ContiguousAndCast(mat2): Cast(DT_FLOAT→DT_FLOAT) = no-op, format 不变
TransData(mat2, NZ): NZ→NZ = no-op

GetMatMulOp:
  fp16/bf16→fp32分支: false (self_dtype=DT_FLOAT)
  CheckSupportInfoFormatNzNzNd: false (self_format=ND≠NZ)
  CheckSupportInfoFormatNdNzNd: true ← Fix 2 新增
  → ReFormat x1→ND, x2→NZ (no-op)
  → AiInfraMatmulNd(FORMAT_ND, FORMAT_ND 作 output format)
    → simplified_key: diy,2/29/2/2/0/0/0/0 匹配 NZ binary
```

### tiling 层
```
DoOpTiling:
  InitTilingData → mm_.GetTiling() (matmul_tiling API)
  SetRunInfo
  SelectNZTiling: bFormat=NZ → tilingSelect_=BASE ← Fix 3 修改(移除 bType!=DT_FLOAT)
  DoBasicTiling:
    DoSelectTiling: tilingSelect_==BASE → 只做 DoL2CacheTiling
    tilingEnable_ 全为 BASE 值

DoLibApiTiling:
  DoTilingKey: tilingEnable_ 全 BASE → key = (0,0,0,1,0,0) = BASE AIC_ONLY
```

### kernel 层
```
binary: AiInfraMatmul_ND_NZ_ND_ND_FP32_FP32_FP32_FP32
构建系统传: -DFORMAT_X2=29 -DFORMAT_FRACTAL_NZ=29
→ format_x2 = CubeFormat::NZ ✅
→ IS_ND_NZ_FP32: 取决于 ORIG_DTYPE_X2 是否被定义
  如定义: IS_ND_NZ_FP32=1 → #else 通用模板(所有 key 可用)
  未定义: IS_ND_NZ_FP32=0 → #elif NZ 限制模板(仅 BASE+K_SHIFT)

无论哪种, tiling key=BASE 都有对应模板 ✅
```

## 修复记录

### Fix 1: 放开 aclnn 层 fp32+NZ 限制 ✅
- 文件: `ai_infra_matmul/op_api/aclnn_ai_infra_matmul.cpp`
- 删除 CheckWeightNzDtype 中 fp32 无条件拒绝
- 添加 CheckWeightNzDtypeValid 中 arch 判断(仅 A2A3 允许)

### Fix 2: 添加 NDNZ dispatch 路径 ✅
- 文件: `common/op_host/op_api/matmul_util.cpp`
- 添加 `CheckSupportInfoFormatNdNzNd` 函数
- 在 GetMatMulOp fallback 前插入 NDNZ 分支

### Fix 3: tiling 层 fp32+NZ 强制 BASE ✅
- 文件: `ai_infra_matmul/op_host/ai_infra_matmul_base_tiling.cpp`
- SelectNZTiling: 移除 `bType != DT_FLOAT` 条件
- 效果: fp32+NZ 也走 tilingSelect_=BASE, 避免 key 不匹配

## 待确认
- 第二次上板日志确认三处修改全部生效，但结果仍然完全错误
- 日志关键发现: **isNzB=0**，kernel 不知道 B 是 NZ 格式

### Fix 4: GetFormat 中设置 isNzB ✅
- 文件: `ai_infra_matmul/op_host/ai_infra_matmul_base_tiling.cpp`
- 根因: `GetFormat` 解析了 `bFormat=NZ` 但没设 `args_.isNzB=true`
  nn 仓在 `GetMoreMultiCoreSplitKArgs` 中有条件设置 isNzB，
  但更根本的是 omni 仓完全遗漏了从 format 到 isNzB 的映射
- 改动: `GetFormat` 末尾添加 `args.isNzA/isNzB = (format == NZ)`
- 这样 kernel 通过 `matmulRunInfo.isNzB=1` 知道 B 是 NZ 格式
