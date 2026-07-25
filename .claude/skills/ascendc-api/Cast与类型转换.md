# Cast 与类型转换 API

trace 自真实仓使用模式（attention/bsa_select_block_mask、ops-nn/ascend_quant）+ ops-tensor 头文件。

## 两种 Cast 形态

### 1. Membase Cast（A2A3，LocalTensor）

```cpp
// 基本形式
Cast(dstLocal, srcLocal, RoundMode::XXX, count);
// 显式模板版（跨 dtype）
Cast<DstT, SrcT>(dstLocal, srcLocal, RoundMode::XXX, count);
```

实证（bsa_radix_topk_service.h:525,1028）：
```cpp
Cast(xLocalFp32, xLocalInt16, RoundMode::CAST_NONE, curTileLen);     // int16 → fp32
Cast(xLocalInt32, xLocalFp32, RoundMode::CAST_RINT, curTileLen);     // fp32 → int32（四舍五入）
Cast<uint8_t, half>(maskU8, maskHalf, RoundMode::CAST_NONE, alignLen);
```

**Membase RoundMode 枚举值**：
| 值 | 含义 |
|---|---|
| CAST_NONE | 不取整（浮点间转换 / 升精度） |
| CAST_RINT | 四舍五入（就近偶） |
| CAST_FLOOR | 向下取整 |
| CAST_CEIL | 向上取整 |
| CAST_ROUND | 舍入（远离零） |

**饱和语义（关键）**：Membase Cast **无显式 SatMode 参数**。转 int8/int16 时饱和行为由目标 dtype 隐式决定（默认饱和到范围）。若需精确控制，用 Regbase CastTrait（见下）或手动 Maxs/Mins 截断。

### 2. Regbase CastTrait（A5，MicroAPI）

A5 Regbase 用 CastTrait 结构体显式控制饱和与取整（ascend_quant.h:50-81 实证）：

```cpp
AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_INT8 = {
    AscendC::MicroAPI::RegLayout::ZERO,        // 布局
    AscendC::MicroAPI::SatMode::SAT,            // 饱和（关键！）
    AscendC::MicroAPI::MaskMergeMode::ZEROING,  // 掩码合并
    RoundMode::CAST_ROUND                        // 取整模式
};
// 用 CastTrait 做 cast
```

**SatMode**（饱和，量化必查）：
| 值 | 含义 |
|---|---|
| SAT | 饱和（溢出截到目标范围，如 fp32→int8 截到 [-128,127]） |
| NO_SAT | 不饱和（溢出行为未定义/回绕） |
| UNKNOWN | 默认（由场景定） |

## 量化 kernel 的 Cast 选型（dequant / requant）

| 转换 | 推荐 RoundMode | SatMode | 说明 |
|---|---|---|---|
| int8 → fp32（dequant 搬入） | CAST_NONE | — | 升精度，无需取整 |
| fp32 → int8（requant 输出） | CAST_ROUND / CAST_RINT | **SAT** | 必须饱和，否则溢出 UB |
| fp16 → fp32（升精度算） | CAST_NONE | — | 中间精度 |
| fp32 → fp16（降精度存） | CAST_RINT | 可选 | 取整 |

## 位级重解释 ReinterpretCast（不做值转换）

```cpp
LocalTensor<uint8_t> maskU8 = maskTensor.ReinterpretCast<uint8_t>();
```
**不转值**，只换 dtype 视角（同块内存按新 dtype 读）。用于 mask/位操作，非量化。实证（fia_block_vec_nonquant_mla.h:714）。

## Vector 计算 API（常配合 Cast 用）

### Membase 形式（LocalTensor）
```cpp
Add(dstLocal, src1Local, src2Local, count);     // 向量+向量
Adds(dstLocal, srcLocal, scalarValue, count);    // 向量+标量
Muls(dstLocal, srcLocal, scalarValue, count);    // 向量×标量
Maxs(dstLocal, srcLocal, scalarValue, count);    // 逐元素 max(向量, 标量)
Mins(dstLocal, srcLocal, scalarValue, count);    // 逐元素 min(向量, 标量)
```

### Regbase 形式（RegTensor + mask）
```cpp
Add(dst, src1, src2, mask);
Adds(dst, src, scalar, mask);
Muls(dst, src, scalar, mask);
```

## 量化 elementwise 骨架（dequant→compute→requant）

```cpp
// dequant: int8 → fp32 → ×scale
Cast(xFp32, xInt8, RoundMode::CAST_NONE, count);     // int8→fp32
Muls(xFp32, xFp32, scale1, count);                    // ×scale

// compute: fp32 add
Add(xFp32, xFp32, yFp32, count);

// requant: ÷outScale → round → saturate → int8
Muls(xFp32, xFp32, 1.0f/outScale, count);
// A2A3 Membase: 饱和隐式；若不确定，手动截断
Mins(xFp32, xFp32, 127.0f, count);
Maxs(xFp32, xFp32, -128.0f, count);
Cast(zInt8, xFp32, RoundMode::CAST_ROUND, count);    // fp32→int8
```

> 饱和安全建议：A2A3 若不确定 Cast 默认饱和，用 Maxs/Mins 显式截断再 Cast；A5 用 CastTrait SatMode::SAT。

## 常见坑
- fp32→int8 **不饱和** → 溢出值未定义，精度全错（量化 kernel 最易踩）
- Cast 的 count 是元素数，非字节数
- fp32 中间 buffer 需 sizeof(float)×count = 4× int8（InitBuffer 易写错）
- ReinterpretCast 不等于 Cast（位级 vs 值级）

## 来源
- `ops-transformer_AI/attention/bsa_select_block_mask/op_kernel/arch35/`（Cast 用法）
- `ops-nn/quant/ascend_quant/op_kernel/arch35/ascend_quant.h`（CastTrait + SatMode）
- `ops-nn/activation/gelu` / `norm`（Vector 计算 API 用法）