# A2A3 Tiling 优化

## TilingKey 旧系统（宏编码）

A2A3 使用 64-bit 编码 18-20 个字段：

```
bit 0: KernelType, bit 1: VectorAllReduce
bits 5-2: ImplMode, bits 9-6: Layout
bits 13-10: S1TemplateType (4 bit enum)
bits 17-14: S2TemplateType (4 bit enum)
bits 21-18: DTemplateType (4 bit enum)
bits 29-26: BigDoubleBuffer (2 bit)
bits 35-32: PseMode, bits 39-36: BiasMode
bits 43-40: DeqMode, bits 47-44: AntiquantMode
bits 51-48: HasQuant, bits 55-52: HasRope
bits 59-56: HasDelta, bits 63-60: OutDtype
```

旧系统缺点：
- 魔法数字难调试
- 修改一个字段需重新编码整个 key
- 枚举值含义不直观

## TilingData 结构层级

```
BaseParams → CoreParams → TensorSizeParams → SubTiling (Matmul/SoftMax)
```

- `BaseParams`：整体 shape（B, N2, G, S1, S2, D）、dtype
- `CoreParams`：每核处理的 M/N/K 范围及循环次数
- `TensorSizeParams`：buffer 具体大小
- `SubTiling`：子模块 tiling（Matmul 的 baseM/baseN/baseK，SoftMax 的 reduce 参数）

## 分核优化：minimum MTE2 traffic

`quant_grouped_matmul_dequant` 在 Tiling 中选择 M/N split：

```cpp
// 尝试 0..3，计算每个配置的 MTE2 流量
for (int32_t i=1; i<4; i++) {
    int32_t mte2Now = (fracN << (3-i)) + (fracM << i);
    if (mte2Now < mte2Min) { chosen = i; mte2Min = mte2Now; }
}
MCoreNum = 1 << (3-chosen);  // M 方向核数
NCoreNum = 1 << chosen;      // N 方向核数
```

不是简单均分，而是根据 M/N 方向搬运量选择最优切分比例。

## GEMV 阈值切换

当 M 行数 <= GEMV_THRESHOLD(=8) 或 workspace 较小时，从 Normal MM 切换到 GEMV 路径：

```
Normal MM: 一次处理多行，利用 Cube 吞吐
GEMV: 逐行处理，每行独立 Mmad + Dequant → 减少浪费的 padding 计算
```

## workspace-split 大 M 场景

`grouped_matmul_swiglu_quant` 的 SPLITWORKSPACE_TILING_KEY_MODE：
- 当 M 太大导致 workspace 超过 64MB 限制时
- 将 M 拆分为多个 workspace-split 块
- 每块独立走完整 pipeline，通过 GM workspace 传递中间结果

## 静态 Tiling 优化

当 tiling 参数在编译期已知时使用 `MatmulApiStaticTiling`：
```cpp
if constexpr (s1TemplateType != UNKNOW) {
    // 使用静态 tiling，消除运行时参数计算开销
    MatmulApiStaticTiling<...> matmulTiling;
}
```
