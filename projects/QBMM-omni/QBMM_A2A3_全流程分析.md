# A2A3 QBMM 全流程分析 · 速查手册（torch → aclnn → l0op → tiling → kernel）

> **范围**：仅 A2A3（910B `ascend910b` / 910C `ascend910_93`，`DAV_2201`，`__CCE_AICORE__==220`）。arch35(A5/950)、arch20(310P) 标"非 A2A3,不走"。纯静态代码核对,未上板(存疑见附录)。源仓 `ops-nn`、`op-plugin`。
> **命名约定(勿混)**："v3/v4 核" = l0op 算子 `QuantBatchMatmulV3/V4` + kernel 目录 `quant_batch_matmul_v3//v4/`(分派目标);"V5/WeightNz" = aclnn 接口入口。二者正交——pertoken 由 aclnn V5 传入,走 **v3 核**。

## 速查导航
| § | 查什么 |
|---|--------|
| [1 全景图](#1-全景图) | 一眼看清 torch→kernel 整条链 |
| [2 分派:模式→落核](#2-分派量化模式--落核) | 我的量化模式进 v3 核还是 v4 核、走哪个 aclnn 入口 |
| [3 支持矩阵](#3-支持矩阵dtype--format) | dtype×format 组合是否支持、ND/NZ 约束、NZ 分形 shape |
| [4 V3 kernel 路径](#4-v3-kernel-路径tiling--tilingkey--kernel) | 标准量化/pertoken 选哪个 tiling 模板、TilingKey、kernel 类 |
| [5 V4 kernel 路径](#5-v4-kernel-路径a8w4--perblock--pergroup) | A8W4/perblock/pergroup 的核与约束 |
| [6 端到端走线](#6-端到端走线) | 两条完整实例 |
| [附录](#附录) | 文件路径索引 + 证据强度/存疑 |

## 关键结论
1. **ND→V5、NZ→WeightNz**(两份独立主体 `v5.cpp:1062` / `v4.cpp:1756`);**Perblock/A4W4 非对称只有 V5(ND) 能命中**,WeightNz 只判 A8W4。
2. **scale=FLOAT/BF16 且 pertoken_scale=null → T-T(逐通道),非 pertoken**;pertoken 必须带 pertoken_scale。
3. **NZ 支持 transB=false**(`v4.cpp:1233-1243`);NZ 分形侧维 INT8=32、INT4/INT32=64。
4. **v4 的 `QuantType::MX(=4)` 是 TilingKey 槽位复用,非真量化**;**A2A3 无 MX、无 A8W4-Float(FP8×FP4)、无 FP8-E8M0**。
5. **TilingKey 位序(高→低)= `OPTION_ATTRS|PERTOKEN|KTT|TRANS`**(7bit);pertoken-bf16-x2NZ = **84**。
6. **offset**:A8W4-int→yOffset;A4W4 非对称→x2Offset(FLOAT16);其余→x2Offset。

---

## 1. 全景图
```
torch npu_quant_matmul  (op-plugin QuantMatmulKernelNpuOpApi.cpp:136)
  │ is_nz_format(x2)                                    EXEC 实证 :291/294/299/302(2格式×2 scale形态)
  ├─ x2=ND ─→ aclnnQuantMatmulV5       (v4目录 :1183) ─ CommonProcess:1062 ─ 三谓词分派
  └─ x2=NZ ─→ aclnnQuantMatmulWeightNz (v3目录 :2110) ─ CommonProcess:1756 ─ 只判 A8W4
                                    │
        isA8W4 ‖ Perblock ‖ A4W4非对称(仅 V5)  ┌─→ l0op::QuantBatchMatmulV4 ─→ v4 kernel {MSD(A8W4)/Perblock/Pergroup}
        else ──────────────────────────────────┴─→ l0op::QuantBatchMatmulV3 ─→ v3 kernel
                                                      │ tiling: BasicTiling(prio0,条件) / V3Tiling(prio2,兜底)   [PpMatmul=310P]
                                                      │ TilingKey = OPTION_ATTRS|PERTOKEN|KTT|TRANS
                                                      └ kernel: {TBE/BASIC/OPT}×{pertoken?} → BmmDequant*
```

---

## 2. 分派：量化模式 → 落核

### 2.1 aclnn 入口（ND→V5 / NZ→WeightNz）
torch 只一个函数,由 **x2 存储格式**二选一(`op-plugin:143-155`;`is_nz_format`=`:76-83`,判 `FRACTAL_NZ/NZ_C0_4/NZ_C0_16`)。两入口是**两份独立 CommonProcess**,分派谓词不同:

| 入口 | 触发 | 主体文件 : CommonProcess | 分派谓词 |
|------|------|--------------------------|---------|
| `aclnnQuantMatmulV5` | x2=ND | `quant_batch_matmul_v4/.../aclnn_quant_matmul_v5.cpp` : `1062` | **三谓词**(`:1158`) |
| `aclnnQuantMatmulWeightNz` | x2=NZ | `quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.cpp` : `1756` | **只判 A8W4**(`:1838`) |

> **Perblock/A4W4 非对称只有 V5(ND) 能命中**;WeightNz 只判 A8W4。

### 2.2 落核判定表（综合 ND/NZ）
V5(ND) 分派 `v5.cpp:1158`:`if (isA8W4 ‖ isA8W8Perblock ‖ isA4W4PergroupNonSymmetric) → V4 核; else → V3 核`。NZ 分派 `v4.cpp:1838`:`if (isA8W4Float ‖ isA8W4Int) → V4 核; else → V3 核`。

| 量化模式 | x1 / x2 | x1Scale(pertoken)/x2Scale | offset | 命中谓词 | 落核 |
|---|---|---|---|---|---|
| W8A8 pertensor | INT8/INT8 | null / UINT64(quant_param) | null/x2Off | 无 | **v3** |
| W8A8 perchannel(T-T) | INT8/INT8 | null / UINT64‖FLOAT‖BF16 | 可选 x2Off | 无 | **v3** |
| W8A8 pertoken(out=BF16) | INT8/INT8 | FLOAT[M] / BF16‖FLOAT[N] | null | Perblock=F(x2Scale≠FLOAT 或 dim 不匹配) | **v3** |
| W8A8 pertoken(out=INT32/FP16) | INT8/INT8 | FLOAT[M] / FLOAT‖BF16[N] | null | 无 | **v3** |
| INT4×INT4 对称 | INT4/INT4 | null / UINT64 | null | 无 | **v3** |
| **W8A8 perblock** | INT8/INT8 | FLOAT(≥2D)/FLOAT(≥2D,dim 同 x) | null | **Perblock=T** | **v4** |
| **INT4×INT4 pergroup 非对称** | INT4/INT4 | FLOAT[M,1]/FLOAT[⌈k/g⌉,N] | **x2Offset=FLOAT16** | **A4W4NonSym=T** | **v4** |
| **A8W4-int** | INT8/INT32→INT4 | FLOAT / FLOAT‖UINT64 | **yOffset(FLOAT)** | **isA8W4(int)=T** | **v4** |

**三谓词精确定义**：
- `isA8W4` = `isA8W4Float(:69)` ‖ `isA8W4Int(:91)`;分派处用 `isA8W4IntAfterPre`(INT8×INT4,`:1140`)。float(x1=FLOAT8_E4M3FN&x2=FLOAT4_E2M1)=**950**(`CheckParamsA8W4Float→DAV_3510,:823`);A2A3 仅 int(x1=INT8&x2=INT32,`CheckParamsA8W4Int→910B/910_93,:837-841`)。
- `IsA8W8Perblock`(`:165-175`) = x1Scale≠null & x1=x2=INT8 & x1Scale/x2Scale **均 DT_FLOAT** & 两 scale 维数分别==x1/x2 维数。
- `IsA4W4PergroupNonSymmetric`(`quant_matmul_checker.cpp:354`) = x1Scale/x2Scale/x2Offset 均非 null & x1Scale/x2Scale=FLOAT 且 dim>1 & **x2Offset=FLOAT16 且 dim>1** & bias=null & INT4×INT4 & k%1024==0 & n%256==0。

### 2.3 op-plugin 前处理速查（`QuantMatmulKernelNpuOpApi.cpp`）
- **A8W4 识别**:`is_a8W4_int = x1==kChar&&x2==kInt`(INT8×INT32,`:156`);`is_a8W4_float`(FP8×FP4)=950,A2A3 不产。
- **offset 路由**:A8W4-int→`y_offset`,其余→`x2_offset`(`:245-249`)。
- **group 打包**:`check_and_get_groups`(`:110-134`)要求 3 元素(M/N/K),打包 `(M<<32)|(N<<16)|K`(`:131`);只打包不推断。
- **transpose 一致性**:pertoken 存在时 `is_x_scale_same_transpose`(`:58-75`,比 stride)强制 x1↔pertoken_scale、x2↔scale 转置态一致,否则报 `transpose are not same`;A8W4 跳过(`:263`)。
- **`npu_trans_quant_param`**:`scale==FLOAT && !pertoken && out∉{BF16*,INT32}`(`:286`)时把 FLOAT scale 预转量化参数(`:289`);pertoken 不走此通道。

---

## 3. 支持矩阵（dtype × format）

### 3.1 A2A3 有效 dtype 组合
def 实证 + aclnn 文档交叉。**pertoken 组合须带 pertoken_scale**(K-C/K-T);同 dtype 无 pertoken_scale 时是 T-T 逐通道。

| # | x1 | x2 | scale | offset | bias | pertoken_scale | out | 模式 | 出处 |
|---|----|----|-------|--------|------|----------------|-----|------|------|
| 1 | INT8 | INT8 | UINT64/INT64 | null | null/INT32 | null | FP16 | T-C/T-T | V3.md:351,V4.md:424 |
| 2 | INT8 | INT8 | UINT64/INT64 | null/FLOAT32 | null/INT32 | null | INT8 | T-C/T-T | V3.md:352,V4.md:425 |
| 3 | INT8 | INT8 | FLOAT32/BF16 | null | null/INT32/BF16/FLOAT32 | null | BF16 | **T-C/T-T(非 pertoken)** | V3.md:353 |
| 4 | INT8 | INT8 | FLOAT32/BF16 | null | null/INT32 | null | INT32 | T-C/T-T | V3.md:355,V4.md:429 |
| 5 | INT4/INT32 | INT4/INT32 | UINT64/INT64 | null | null/INT32 | null | FP16 | INT4(T-C/T-T) | V3.md:354,V4.md:428 |
| 6 | INT8 | INT8 | FLOAT32/BF16 | null | null/INT32/BF16/FLOAT32 | **FLOAT32** | BF16 | **K-C/K-T pertoken** | V4.md:426 |
| 7 | INT8 | INT8 | FLOAT32 | null | null/INT32/FP16/FLOAT32 | **FLOAT32** | FP16 | **K-C/K-T pertoken** | V4.md:427 |
| 8 | INT4/INT32 | INT4/INT32 | FLOAT32/BF16 | null | null/INT32/BF16/FLOAT32 | **FLOAT32** | BF16 | INT4+pertoken | V4.md:430 |
| 9 | INT4/INT32 | INT4/INT32 | FLOAT32 | null | null/INT32/FP16/FLOAT32 | **FLOAT32** | FP16 | INT4+pertoken | V4.md:431 |

**补充约束**(def `quant_batch_matmul_v3_def.cpp` 顶层 `:23-250`,910b/910_93 复用 `:266-267`):
- x1/x2 仅 INT8/INT32/INT4(INT32=1 打 8 个 INT4 的承载);pertoken_scale 仅 FLOAT32(`:193-200`)。
- offset 仅 FLOAT32 且**仅 out=INT8 时可存在**(`:134-141`);bias 为 INT4/INT32 权重时只支持 1 维 `(n,)`。
- **950-only,A2A3 无**:`FLOAT8_E4M3FN/E5M2、HIFLOAT8、FLOAT4_E2M1`(x1/x2)、`FLOAT8_E8M0`(scale)——仅 `config950`(`:319+`);G-B/B-B/MX/T-CG/K-G 仅 aclnn V5 新增。

### 3.2 ND vs NZ（WeightNz）对照
x1 恒 ND;NZ 仅作用于 x2。

| 维度 | ND (x2) | NZ/WeightNz (x2) | 出处 |
|------|---------|-------------------|------|
| 格式 | `FORMAT_ND` | `FORMAT_FRACTAL_NZ` | def.cpp:72-85 |
| transB=true | ✓ `(batch,n,k)` | ✓ | V3.md:169/172 |
| **transB=false** | ✓ `(batch,k,n)` | **✓ 支持** | v4.cpp:1233-1243(+全 shape 校验双向) |
| transA | INT8 可 T/F;INT32/INT4 仅 F | 同左 | V3.md:225,V4.md:313 |
| NZ 分形(transB=true) | — | `(batch,k1,n1,n0,k0)`,n0=16;**INT8 k0=32 / INT4·INT32 k0=64** | common_check.h:52-55 |
| NZ 分形(transB=false) | — | `(batch,n1,k1,k0,n0)`,k0=16;**INT8 n0=32 / INT4·INT32 n0=64** | 同上 |
| 维度范围 | **2-6** | **4-8** | V4.md:167;op_api `MIN/MAX_DIM_NUM_NZ=4/8` |
| scale/bias 维度 | scale 1D `(t,)` t∈{1,n};bias `(n,)`‖`(batch,1,n)` | 同 ND(scale/bias 恒 ND) | V3.md:127/147 |

NZ 分形常量:`NZ_STORAGE_PENULTIMATE_DIM=16`、`NZ_K0_VALUE_INT8_TRANS=32`、`NZ_K0_VALUE_INT4_TRANS=64`(`common_check.h:52-55`,组装 `.cpp:47-75`)。transB=false 归一化:x2 已 NZ→`:1234` 直接放行;x2=ND 且 `!transposeX2`→`:1237-1242` 物理转置+翻转再转 NZ。

---

## 4. V3 kernel 路径（tiling → TilingKey → kernel）

### 4.1 tiling 模板优先级（A2A3）
入口 `quant_batch_matmul_v3_tiling.cpp:1560-1576` 按 arch 分流;A2A3(`npuArch=DAV_2201`)走 `TilingRegistry::DoTilingImpl({0,1,2})`,按 0→1→2 试 `IsCapable`:

| prio | 模板类 | A2A3 生效性 |
|---|---|---|
| 0 | `QuantBatchMatmulV3BasicTiling` | `IsCapable=true`(`basic_tiling.cpp:95`),真门控 `CheckUseBasicTiling()`(`:287`),不满足回退(`:117`) |
| 1 | `PpMatmulInt8Tiling` | 要求 `ASCEND310P`(`arch20/...:80-82`)——**恒假,不生效** |
| 2 | `QuantBatchMatmulV3Tiling` | 基类 `IsCapable=true`(`tiling_base.cpp:814`)——**兜底** |

即实际两模板:BasicTiling(0,条件) + V3Tiling(2,兜底)。

### 4.2 输入特征 → kernelTemplateType
`kernelTemplateType`(2bit) = `(isBf16Opt_<<1) | isBasicTiling`(`tiling.cpp:979`);枚举 **TBE=0/BASIC=1/OPT=2/PPMATMUL=3**(`tiling_key.h:26-29`,A2A3 只产 0/1/2)。

- **`isBf16Opt_`**(`ProcessMSmall`,`tiling.cpp:666-685`) = `(isDecode&&(isAllMix‖isPertoken)) ‖ IsKCNetDecode()` `‖ CheckSupportConditionQbmm(Perchannel)` `‖ (CheckSupportConditionQbmm(Pertoken)&&isPertoken)`,末 `&& !isTilingOut_`(MC2 不走 OPT)。(isDecode=`mSize<=baseM`;isPertoken=`needWorkspace<50M && isPertoken`;IsKCNetDecode=`:626-664` int8+pertoken+batchC==1+m≤256+无bias+K/N 64 对齐)
- **`isBasicTiling`** = 走 BasicTiling(prio0) 且 `cDtype==BF16 && IsPertokenBasicSwitchCondition()`(`basic_tiling.cpp:1381-1387`),否则委托 V3Tiling(`:1389`);走兜底(prio2)=0。
- **`IsPertokenBasicSwitchCondition`**(`basic_tiling.cpp:176-199`,并设 `isAicAiv1_2`) = `[(kSize≤1152&&baseBlock≥682)‖(1152<kSize≤1536&&mSize≥12544&&nSize≥1280)] && aFormat==ND && bFormat==NZ && isPertoken && !transA && !transB`。

| 输入特征 | → kernelTemplateType |
|---|---|
| pertoken 小 m(isDecode) 或 Perchannel/KN 满足,非 MC2 | **OPT=2** |
| pertoken-bf16 + x1=ND/x2=NZ + `IsPertokenBasicSwitchCondition` 满足 | **BASIC=1**(+IS_AICAIV_1_2) |
| int8 基本块(CheckUseBasicTiling 真,非上一行),isBf16Opt_=0 | **BASIC=1** |
| 其余标准量化/pertoken 走兜底 | **TBE=0** |

### 4.3 TilingKey 位域 + 算例
`GET_TPL_TILING_KEY(trans, kernelTemplateType, isPertoken, optionAttrs)`(`tiling.cpp:984`),参数序=`ASCENDC_TPL_ARGS_DECL` 声明序 `TRANS(2)→KTT(2)→PERTOKEN(1)→OPTION_ATTRS(2)`(`tiling_key.h:56-70`),**7bit**,第一参数打最低位。源码注释 `:74` "from high to low: needClean, pertoken, opt, basic, transX1, transX2" 印证:

| 位段 | [6:5] | [4] | [3:2] | [1:0] |
|------|-------|-----|-------|-------|
| 字段 | OPTION_ATTRS | PERTOKEN | KTT | TRANS |
| 取值 | 0 NONE/1 NEED_ATOMICLEAN/2 IS_AICAIV_1_2 | 0/1 | 0 TBE/1 BASIC/2 OPT/3 PPMATMUL | 0 NOT/1 B/2 A/3 ALL |

`trans=(transA<<1)|transB`;标准量化 `optionAttrs=NeedAtomiClean()`(bf16 恒 0);pertoken-basic `optionAttrs=(isAicAiv1_2<<1)`。
- **锚点** `GET_TPL_TILING_KEY(1,PPMATMUL=3,0,0)`=`0b0001101`=**13**(`pp_matmul_int8_tiling.cpp:99` 注释印证低位=TRANS)。
- **算例1** pertoken-bf16-x2NZ:trans=0/KTT=BASIC=1/pert=1/opt=IS_AICAIV_1_2=2 → `OPT(10)PERT(1)KTT(01)TRANS(00)`=`0b1010100`=**84**。
- **算例2** 标准量化 fp16 transB:trans=1/KTT=0/pert=0/opt=0(或 NeedAtomiClean) → `0b0000001`=**1**。

### 4.4 TilingKey → kernel 类
kernel 入口 `quant_batch_matmul_v3<TRANS,KTT,PERTOKEN,OPTIONATTR>`(`quant_batch_matmul_v3.cpp:212`)`if constexpr` 分派(A2A3 `__CCE_AICORE__==220`):

| KTT | PERT | 条件 | kernel 类 / 头文件 |
|---|---|---|---|
| TBE | 0 | 标准量化 int8→fp16/int8/int32 | `QuantBatchMatmulV3BaseKernel`‖`BmmDequant` / `_base.h`,`_v3.h` |
| TBE | 0 | int32 + x1ND/x2NZ + ATOMICLEAN | `BmmDequant`+`BmmDequantInitOutput` / `_v3.h`,`_init_output.h` |
| TBE | 0 | bf16 | `BmmDequantBf16` / `_bf16.h` |
| TBE | 1 | pertoken | `BmmDequantPertoken` / `_pertoken.h` |
| OPT | 0/1 | bf16 / pertoken | `BmmDequantBf16Opt` / `BmmDequantPertokenOpt` |
| BASIC | 0 | 标准/bf16 基本块 | `QuantBatchMatmulV3BaseKernel`‖`BmmBasicDequantBf16` / `_bf16_basic.h` |
| BASIC | 1 | pertoken 基本块(含 IS_AICAIV_1_2) | `BmmDequantPertokenBasic` / `_pertoken_basic.h` |
| PPMATMUL | 0 | 仅 `__CCE_AICORE__==200`(310P) | `PpMatmul` / `arch20/pp_matmul_kernel.h` |

### 4.5 ND/NZ 编译期裁剪
- `ASCENDC_TPL_SEL`(`tiling_key.h:72-266`)按 `FORMAT_X1/X2` 门控:多数分支 `#if FORMAT_X1 != FRACTAL_NZ`(`:111,126,137,143,152`)——**x1=NZ 只留 NOT_TRANS/B_TRANS,A_TRANS/ALL_TRANS 不生成**。
- **IS_AICAIV_1_2 专属 "x1=ND + x2=NZ"**:host `basic_tiling.cpp:194-196`(`aFormat==ND&&bFormat==NZ`)↔ kernel `#if FORMAT_X1!=NZ&&FORMAT_X2==NZ`(`tiling_key.h:159`,`MIX_AIC_1_2`)。int32-splitk 同条件走 `MIX_AIC_1_0`+ATOMICLEAN(`tiling_key.h:117-118`)。
- **BasicTiling 拒 x1=NZ**:`CheckUseBasicTiling` 在 `aFormat==NZ` `return false`(`basic_tiling.cpp:294-296`)。

---

## 5. V4 kernel 路径（A8W4 / perblock / pergroup）

### 5.1 tiling 注册 + 4 模式覆盖
`quant_batch_matmul_v4_tiling_registry.cpp:44-73`:A2A3(`910B/910_93`)只注册 `{MSD, PERBLOCK, PERGROUP}`(`:65`);`LUT/BASIC_PERBLOCK/PERGROUP_ARCH35` A2A3 不进。910b/910_93 的 `config/.../quant_batch_matmul_v4_binary.json` 一致,共 7 bin,全落下述 4 模式:

| 模式 | 有核依据(bin + kernel 220) | tiling / kernel 类 | 关键约束 | aclnn 文档 |
|---|---|---|---|---|
| **A8W4 perchannel(K_C)** | `..._int8_int4_*`(binary.json:5,160,315,470) + `v4.cpp:67-96`(INT8×INT4,K_C,ND+NZ) | `...MsdTiling` / `QuantBatchMatmulV4Msd<int4b_t,int4b_t,float,DTYPE_Y,K_C>` | x1 INT8/x2 INT4;x1Scale FLOAT32、x2Scale UINT64、yOffset FLOAT32;out fp16/bf16;groupSize=0;K≤29576;MSD 拆 int8→2×int4 | V5.md:35-38 ✓ |
| **A8W4 pergroup(K_G)** | 同 int8_int4 bin;`v4.cpp:79-83`(K_G,仅 NOT_TRANS+ND) | 同 MSD `<...,K_G>` | x1Scale/x2Scale 均 1D→K_C 否则→K_G(`_msd_tiling.cpp:123-126`);K_G 强制 groupSize=256、transB=F、K%256==0 | A2/A3 段缺等式(**有核·文档留白**) |
| **A8W8 perblock** | `..._int8_int8_bf16`(binary.json:625) + `v4.cpp:98-111`(INT8×INT8,`#if !(__NPU_ARCH__==3003)`→A2A3 生效) | `...PerblockTiling`(IsCapable=true) / `QuantBatchMatmulV4Perblock<int8,int8,float,float,float,bf16>` | x1/x2 INT8;x1Scale/x2Scale FLOAT32(N 向 128、K 向 groupSizeK 分块);bias FLOAT32;out fp16/bf16 | V5.md:101-104 ✓ |
| **A4W4 pergroup 非对称** | `..._int4_int4_*`(binary.json:780,935) + `v4.cpp:112-124`(int4×int4,PERGROUP,NOT_TRANS) | `...PergroupTiling` / `QuantBatchMatmulV4Pergroup<int4b_t,int4b_t,float,float,DTYPE_Y>`(复用 V3TilingData) | x1/x2 INT4;x1Scale FLOAT32、**x2Scale UINT64**(bin 实证)、x2Offset FLOAT16;groupSizeK=256;out fp16/bf16 | V5.md:41-44(doc 写 FLOAT32,bin UINT64) |

> `isA8W4` 谓词(§2.2)在 v4 tiling 内再按 scale 维度分 K_C/K_G。**MX/FP8-E8M0/mxfp4/B-B/T-CG/G-B 全 950-only**:kernel 里 `QuantType::MX(=4)`(`v4.cpp:100,113`)**只是 TilingKey 字段槽位复用**(perblock 实例 `<int8,int8,float>`、pergroup `<int4,int4,float>`),真 MX 在 `arch35/*`,A2A3 registry 不注册。

### 5.2 v4 TilingKey
`quant_batch_matmul_v4_tiling_key.h:44-58`,5 字段:`TRANS(2)|QUANT_TYPE(4:NONE0/PER_TENSOR1/PER_CHANNEL2/PER_GROUP3/MX4/K_C5/K_G6)|OPTION_ATTRS(2)|WEIGHTNZ(1)|KTT(4:MSD0/PERBLOCK1/PERGROUP2)`。各模板:MSD→`(trans,K_C/K_G,0,weightNz,MSD)`;Perblock→`(trans,MX,0,0,PERBLOCK)`;Pergroup→`(0,MX,…,PERGROUP)`。

---

## 6. 端到端走线

**① Pertoken BF16 + WeightNz（x1=ND, x2=NZ）**
```
torch(x1=INT8[ND],x2=INT8[NZ],scale=BF16[N],pertoken_scale=FLOAT32[M],out=BF16)
 └ NZ → aclnnQuantMatmulWeightNz(op-plugin:144) → CommonProcess(v4.cpp:1756) 非 A8W4 → l0op::QuantBatchMatmulV3(:1845,v3 核)
    └ tiling: BasicTiling(prio0), cDtype=BF16 且 IsPertokenBasicSwitchCondition 满足 → isBasicTiling=1,isAicAiv1_2=1,trans=0,pertoken=1
    └ TilingKey = OPT(10)PERT(1)KTT(01)TRANS(00) = 84
    └ kernel: BASIC+IS_PERTOKEN+IS_AICAIV_1_2 → BmmDequantPertokenBasic(MIX_AIC_1_2)
```

**② A8W4-int（ND）**
```
torch(x1=INT8[ND],x2=INT32[ND],scale=FLOAT32,offset→y_offset,group_sizes=[M,N,K],out=BF16)
 └ ND → aclnnQuantMatmulV5 → A4W4CaseProcess:x2 INT32→INT4 视图 → isA8W4IntAfterPre=T → l0op::QuantBatchMatmulV4(v5.cpp:1160,v4 核)
    └ tiling: MSD; x1Scale/x2Scale 维度 → K_C 或 K_G
    └ kernel: quant_batch_matmul_v4.cpp:67-96 (QuantType=K_C/K_G) + MSD 预处理
```

---

## 附录

### 文件路径索引
| 层 | 路径 |
|---|---|
| torch | `op-plugin/op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp` |
| aclnn V5(ND) | `ops-nn/matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp` |
| aclnn WeightNz(NZ) | `ops-nn/matmul/quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.cpp` |
| v3 tiling | `.../quant_batch_matmul_v3/op_host/op_tiling/{..._tiling,_basic_tiling,_tiling_base}.cpp` |
| v3 TilingKey/kernel | `.../quant_batch_matmul_v3/op_kernel/{..._tiling_key.h, quant_batch_matmul_v3.cpp}` |
| v4 tiling/kernel | `.../quant_batch_matmul_v4/op_host/op_tiling/*`、`.../op_kernel/quant_batch_matmul_v4.cpp` |
| def / infershape | `.../quant_batch_matmul_v3/op_host/{..._def,_infershape}.cpp` |

### 证据强度与存疑
- **强证据(直读源码)**:分派谓词与行号、tiling 优先级/IsCapable、TilingKey 字段声明+位域注释、kernel 映射、def dtype/format、NZ 分形常量、NZ transB=false 归一化、v4 registry 与 4 模式 bin/kernel。
- **锚点推导**:TilingKey=84 由 声明序 + `tiling_key.h:74` 注释 + PpMatmul=13 三重印证;`GET_TPL_TILING_KEY` 本体为 CANN 框架宏(不在本仓),硬确认需编译后 tiling log。
- **doc↔code 不一致(以代码/bin 为准)**:A8W4 K_G aclnn A2/A3 段缺等式(有核);A4W4 pergroup x2Scale doc=FLOAT32 / bin=UINT64。
- **未做**:未上板、未跑 UT;"aclnn 9 组即对外真值"依赖 op_api 逐条校验,本文只读到关键分支(如 `v5.cpp:204`),未枚举全 9 组。上板须逐组回验。
