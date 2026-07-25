# Reduce / Norm 通用范式

ops-nn norm 类（layer_norm / rms_norm 及融合变体）。内容经真实代码 trace。源：`ops-nn/norm/`。

## 基准算子：add_rms_norm（已 trace）

### Kernel 类（已 trace norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h:50）
```cpp
class KernelAddRmsNormRegBase {   // A5 Regbase
    void Init(x1, x2, gamma, y, rstd, x, tiling);   // :53
    void Process();                                   // :89
    void Compute(uint32_t rowLoopIdx, gammaLocal, curRows);   // :103
};
```
引用共享 base：`../../rms_norm/rms_norm_base.h`（:18）+ `NormCommon::NormCommonRegbase`。

### reduce 原语（已 trace :120）
```cpp
NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(...)   // 平方和归约
```
RMSNorm 核心 = 按行算 x² 的 reduce sum → sqrt(mean) → scale。

## norm 融合变体（ops-nn 特有，已验证清单）
- `add_rms_norm` / `add_rms_norm_cast` / `add_rms_norm_dynamic_quant`(+v2)
- `add_layer_norm`(+grad+quant+v2) / `ada_layer_norm`(+grad+quant+v2)
- 融合点：norm 前接 add（残差）、后接 quant/cast，减 GM 往返

## base 设计理由
- **reduce**：按行做平方和归约（CalculateSquareReduceSum），需双 pass 或 online 算法
- **融合**：add+norm+quant 融合时，中间结果留 UB，避免逐算子回 GM
- **Regbase**：A5 norm 用 MicroAPI 寄存器级 reduce，比 Membase TBuf 更高效

## 通用 reduce 优化
- **行间并行**：每行独立 reduce，核间按行分（rowLoopIdx）
- ** UB reduce**：小 reduce 在 UB 完成，避免 GM 中转
- **融合减往返**：norm 的 rstd 结果可留 UB 供后续 scale/quant 复用