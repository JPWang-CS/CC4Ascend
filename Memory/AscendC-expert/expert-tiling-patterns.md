# Tiling 策略专家技巧

## TilingKey 系统选择

### 新旧系统判断

| 维度 | 旧系统（宏编码） | 新系统（模板参数） |
|------|------|------|
| 方式 | `#define KEY_123 0x...` | `ASCENDC_TPL_ARGS_DECL/SEL` |
| 编译时选择 | `if (key == VALUE)` 运行时分支 | `if constexpr` 编译时分支 |
| TilingKey 传递 | 单个 64-bit 整数 | 结构化 bitfield |
| 调试性 | 难读（魔法数字） | 可读（枚举名） |
| 扩展性 | 重新编码整个 key | 加一个 bitfield 即可 |

**选择原则**：
- 新算子 → 直接用新系统（ASCENDC_TPL_*）
- 旧算子维护 → 保持旧系统，不要混用

### TilingKey bitfield 设计原则

1. **层级分组**：硬件特性 → 算法选择 → 特征标志
2. **预留 padding**：关键字段预留扩展位
3. **互斥字段合并**：如 S1/S2/D alignment 用 lookup table 而非独立 bit

```cpp
// 好的设计：S1/S2/D 组合成一个 Config 枚举
ASCENDC_TPL_UINT_DECL(Config, 10, ...) // 19 个枚举值

// 坏的设计：S1/S2/D 各占 bitfield
// → 组合爆炸 → TilingKey 空间浪费
```

## 双缓冲判定策略

### 何时用 double buffer

```
tile_size × 2 ≤ UB_SIZE → 可以用
tile_size > UB_SIZE / 2 → 被迫单缓冲
```

A2A3 显式在 TilingKey 中暴露 BigDoubleBuffer 选项，A5 由硬件自动管理。

### 三重缓冲

仅在以下场景考虑：
- CopyIn + Compute + CopyOut 三阶段完全解耦
- tile 远小于 UB_SIZE / 3
- Pipeline 深度确实需要（极少见）

**反模式**：为用三重缓冲而强行减小 tile → 循环次数增加 → 总吞吐下降

## 分核策略

### A2A3：需要错位避免同地址冲突

```
核0: offset=0 → 数据[0:N]
核1: offset=offset_1 → 数据[offset_1:N+offset_1]  // 偏移避免冲突
```

### A5：不需要错位

```
核0: 数据[0:N]
核1: 数据[N:2N]  // 直接均分，硬件支持同地址并行
```

### GMM 类特殊分核

Grouped MatMul 有对角线分核 / 横向分核 / 对角线分组三种方案，根据 group 数量和大小选择：
- 组数多、每组小 → 对角线分核
- 组数少、每组大 → 横向分核

## 量化 Tiling 特殊考虑

### Scale 数据搬入

量化场景 UB Buffer 需为 scale 预留空间：
```
UB_Budget = 激活 × 2(double buffer) + Weight(NZ) + Scale + Workspace
```

Scale 大小取决于量化模式：
- Pertensor: 可忽略
- PerChannel: N × sizeof(scale_type)
- PerToken: M × sizeof(scale_type)
- PerGroup: (M × K/group_size) × sizeof(scale_type)

PerGroup/PerBlock 场景 scale 可能比数据还大，是 UB 预算的隐藏杀手。

### A5 MX 量化

MXFP8/MXFP4 的 scale 是 FLOAT8_E8M0，group_size=32。需在 tiling 中区分普通量化 vs MX 量化的 buffer 分配。
