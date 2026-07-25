# 量化 elementwise 编程范式

量化算子（dequant→compute→requant）的 kernel 写法。结合 [membase.md](membase.md) 骨架 + [Cast 与类型转换](../ascendc-api/Cast与类型转换.md)。

## 典型场景

int8 量化加法：两 int8 输入 + scale → fp32 中间算 → int8 输出。适用于所有"输入量化、输出反量化"的 elementwise。

## 核心模式：dequant → compute → requant

```cpp
template <typename Y_T>   // int8_t 输出
class QuantAddExample {
    TQue<VECIN, BUFFER_NUM> inputQueueX;   // int8_t 输入
    TQue<VECIN, BUFFER_NUM> inputQueueY;
    TQue<VECOUT, BUFFER_NUM> outputQueueZ; // int8_t 输出
    TBuf<VECCALC> xFp32Buf, yFp32Buf;      // fp32 中间（注意 4× int8 大小）
    float scale1, scale2, outScale;
};

void Compute(int32_t progress) {
    LocalTensor<int8_t> xLocal = inputQueueX.DeQue<int8_t>();
    LocalTensor<int8_t> yLocal = inputQueueY.DeQue<int8_t>();
    LocalTensor<int8_t> zLocal = outputQueueZ.AllocTensor<int8_t>();
    LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
    LocalTensor<float> yFp32 = yFp32Buf.Get<float>();

    // ① dequant: int8 → fp32（升精度，不取整）
    Cast(xFp32, xLocal, RoundMode::CAST_NONE, tileLength);
    Cast(yFp32, yLocal, RoundMode::CAST_NONE, tileLength);

    // ② ×scale（dequant 完）
    Muls(xFp32, xFp32, scale1, tileLength);
    Muls(yFp32, yFp32, scale2, tileLength);

    // ③ compute: fp32 add
    Add(xFp32, xFp32, yFp32, tileLength);

    // ④ requant: ÷outScale
    Muls(xFp32, xFp32, 1.0f / outScale, tileLength);

    // ⑤ 饱和（A2A3 显式截断，防溢出）
    Mins(xFp32, xFp32, 127.0f, tileLength);
    Maxs(xFp32, xFp32, -128.0f, tileLength);

    // ⑥ fp32 → int8（取整）
    Cast(zLocal, xFp32, RoundMode::CAST_ROUND, tileLength);

    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
    outputQueueZ.EnQue(zLocal);
}
```

## 关键点（易错）

### buffer 尺寸
fp32 中间 buffer = `tileLength * sizeof(float)` = 4× int8。InitBuffer 必须按 fp32 算：
```cpp
pipe.InitBuffer(xFp32Buf, tileLength * sizeof(float));   // 不是 sizeof(int8_t)
```

### 饱和（最大精度风险）
- **A2A3 Membase**：Cast 无显式 SatMode。**推荐显式 Mins/Maxs 截断**再 Cast，否则 int8 溢出值未定义。
- **A5 Regbase**：用 `MicroAPI::CastTrait{SatMode::SAT, ...}` 显式饱和（见 ascendc-api/Cast与类型转换.md）。

### RoundMode 选型
| 步骤 | RoundMode |
|---|---|
| int8 → fp32 | CAST_NONE（升精度无需取整） |
| fp32 → int8 | CAST_ROUND 或 CAST_RINT（舍入） |

### scale 处理
- 标量 scale：`Muls(dst, src, scalar, count)`
- per-token/per-channel scale（向量）：需把 scale 搬进 UB，用 `Mul(dst, src, scaleLocal, count)`（向量×向量）

## CopyIn / CopyOut（int8 搬运）
```cpp
void CopyIn(int32_t progress) {
    LocalTensor<int8_t> xLocal = inputQueueX.AllocTensor<int8_t>();
    DataCopy(xLocal, inputGMX[progress * tileLength], tileLength);  // int8 搬运
    inputQueueX.EnQue(xLocal);
}
// CopyOut 同理（int8 LocalTensor → GM）
```

## 与纯 elementwise 的区别
- 多了 dequant/requant（Cast + Muls）
- 中间用 fp32 保证精度（int8 直接算会溢出/丢精度）
- 必须处理饱和（否则输出错）

## 何时用此范式
- 量化 matmul 的向量后处理
- 量化激活函数（int8 gelu）
- 量化 add/mul 等 elementwise
- dequant/swiglu_quant 融合算子

## 相关
- [membase.md](membase.md)（基础三阶段骨架）
- ascendc-api [Cast与类型转换.md](../ascendc-api/Cast与类型转换.md)（Cast 签名 + SatMode 详解）