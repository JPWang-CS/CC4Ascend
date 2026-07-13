# Regbase 编程范式（A5 专属）

## 概述

Ascend 950 引入 **Regbase** 编程范式（`AscendC::MicroAPI`），相比传统 Membase（Vector API）更接近底层硬件寄存器操作，提供更精细的向量化控制。

## Membase vs Regbase 对比

| 特性 | Membase (Vector API) | Regbase (MicroAPI) |
|------|---------------------|---------------------|
| 数据载体 | `LocalTensor<T>` + Queue 机制 | `RegTensor<T>` 寄存器 |
| 内存管理 | 显式 AllocTensor/EnQue/DeQue/FreeTensor | 寄存器自动分配 |
| 掩码控制 | 函数参数控制 | `MaskReg` 寄存器控制 |
| 数据搬运 | `DataCopy`/`DataCopyPad` | `MicroAPI::DataCopy` + 分发模式 |

## Regbase 核心 API

### 寄存器与掩码类型

| API | 说明 |
|-----|------|
| `RegTensor<T>` | 向量寄存器张量类型 |
| `MaskReg` | 掩码寄存器类型 |
| `CreateMask<T, Pattern>()` | 创建掩码（ALL/HALF 等） |
| `UpdateMask<T>(count)` | 根据剩余元素数动态更新掩码 |
| `AddrReg` | 地址偏移寄存器 |

### 标量与向量操作

| API | 说明 |
|-----|------|
| `Duplicate(reg, val, mask)` | 标量值广播到寄存器所有元素 |
| `Arange(reg, start)` | 生成连续序列 [start, start+1, ...] |
| `Add/Sub/Mul/Div(dst, src1, src2, mask)` | 向量算术运算 |
| `Adds/Muls(dst, src, scalar, mask)` | 向量与标量运算 |
| `Cast<DT, ST>(dst, src, mask)` | 数据类型转换 |
| `Compare<T, CMPMODE>(mask, src1, src2, pred)` | 向量比较生成掩码 |

### 数据搬运

| API | 说明 |
|-----|------|
| `DataCopy<T, LoadDist>(reg, addr)` | UB → 寄存器加载 |
| `DataCopy<T>(addr, reg, mask)` | 寄存器 → UB 存储 |
| `DataCopyGather(dst, base, indices, mask)` | 根据索引收集数据 |
| `CreateAddrReg<T>(loop, stride)` | 循环地址偏移寄存器 |

### 分发模式 (LoadDist)

| 模式 | 说明 | 典型用途 |
|------|------|----------|
| `DIST_NORM` | 正常连续加载 | 连续数据处理 |
| `DIST_UNPACK_B16` | 16位解包 (FP16/BF16→FP32) | 混合精度 |
| `DIST_BRC_B32/B16` | 广播加载 | Scale/标量广播 |
| `DIST_E2B_B32` | 标量→向量广播 | 索引值广播 |

## 代码示例

### 基础 Regbase 操作

```cpp
__simd_vf__ __aicore__ void GenIndexBuf(ubuf int32_t* helpAddr, int32_t colFactor) {
    // 声明寄存器
    AscendC::MicroAPI::RegTensor<int32_t> v0, v1, vd1;

    // 创建全量掩码
    auto preg = AscendC::MicroAPI::CreateMask<int32_t,
                   AscendC::MicroAPI::MaskPattern::ALL>();

    // 标量→寄存器
    AscendC::MicroAPI::Duplicate(v1, colFactor, preg);
    // 序列生成 [0, 1, 2, ...]
    AscendC::MicroAPI::Arange(v0, 0);
    // 向量除法
    AscendC::MicroAPI::Div(vd1, v0, v1, preg);
    // 写回 UB
    AscendC::MicroAPI::DataCopy(helpAddr, vd1, preg);
}
```

### 动态掩码（尾部处理）

```cpp
__simd_vf__ __aicore__ void Process(ubuf int8_t* curYAddr,
                                     uint16_t repeatTimes, uint16_t computeSize) {
    MicroAPI::RegTensor<int8_t> vregTemp;
    MicroAPI::MaskReg preg;

    for (uint16_t r = 0; r < repeatTimes; r++) {
        preg = MicroAPI::UpdateMask<int8_t>(remaining);
        auto offset = MicroAPI::CreateAddrReg<int8_t>(r, computeSize);
        MicroAPI::DataCopy(vregTemp, curXAddr, offset);
        MicroAPI::DataCopy(curYAddr, vregTemp, offset, preg);
    }
}
```

### Gather 操作

```cpp
__VEC_SCOPE__ {
    MicroAPI::RegTensor<uint32_t> indicesReg;
    MicroAPI::RegTensor<int32_t> vd0;

    for (uint16_t i = 0; i < loopNum; i++) {
        // E2B 广播模式加载索引
        MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_E2B_B32>(
            indicesReg, indicesAddr);
        // 按索引 Gather
        MicroAPI::DataCopyGather(vd0, curXAddr, indicesReg, preg);
        // Block Copy 输出
        MicroAPI::DataCopy<int32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            curYAddr, vd0, blockStride, preg);
    }
}
```

## 适用场景建议

| 场景 | 推荐范式 |
|------|----------|
| 简单的连续数据搬运+计算 | Membase |
| 双缓冲流水线 | Membase |
| 精细寄存器控制 | Regbase |
| 复杂掩码逻辑 | Regbase |
| Gather/Scatter 访存 | Regbase |
| 混合场景 | 两者结合（Regbase 核心计算 + Membase 数据搬运） |

## 来源
- `ops-transformer_AI/docs/zh/develop/cross_platform_migration_guide.md` §Regbase
