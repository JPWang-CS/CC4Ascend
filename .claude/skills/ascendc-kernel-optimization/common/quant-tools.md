# 量化工具算子范式

ops-nn quant 类（40 算子，cast/scale/round 工具）。内容经真实代码 trace。源：`ops-nn/quant/`。

## 基准算子：ascend_quant（已 trace）

### Kernel 类（已 trace quant/ascend_quant/op_kernel/arch35/ascend_quant.h:31）
```cpp
template <typename T, typename U, uint64_t RoundMode>
class AscendQuantBase {
    constexpr static CastTrait CAST_TRAIT_HALF_TO_FP32 = {..., RoundMode::UNKNOWN};        // :49
    static constexpr CastTrait CAST_TRAIT_FP32_TO_HIFP8 = []() {
        if constexpr (RoundMode == TPL_ROUND_MODE_HYBRID)
            return CastTrait{..., RoundMode::CAST_HYBRID};    // :56
        else
            return CastTrait{..., RoundMode::CAST_ROUND};     // :59
    }();
};
```
用 `AscendC::MicroAPI::CastTrait` + `SatMode::SAT` + `MaskMergeMode::ZEROING`。

## cast/scale/round 范式（已 trace）
量化工具核心 = **cast（dtype 转换）+ scale（缩放）+ round（取整）+ saturate（饱和）**：
- `CastTrait` 封装 round 模式（UNKNOWN / CAST_HYBRID / CAST_ROUND）
- `SatMode::SAT` 饱和处理（溢出截断到范围）
- A5 用 MicroAPI CastTrait，寄存器级 cast

## quant 算子清单（已验证，部分）
`ascend_quant`(+v2) / `ascend_dequant` / `ascend_anti_quant`(+v2) / `anti_mx_quant` / `dequant_bias` / `dequant_swiglu_quant` / `acts_ulq` / `dequantize`

## base 设计理由
- 量化工具是**纯 elementwise cast + scale**，无 reduce 无 matmul
- 模板化 RoundMode 让一个 kernel 支持多种取整策略
- 作为其他算子的**构建块**（matmul/attention 内部调 ascend_quant）

## 通用优化
- **寄存器级 cast**（A5 MicroAPI CastTrait）省 UB 中转
- **sat + round 合并**：一次 cast 完成 scale × round × saturate
- **MX 量化**：anti_mx_quant 处理 per-group scale（e8m0）