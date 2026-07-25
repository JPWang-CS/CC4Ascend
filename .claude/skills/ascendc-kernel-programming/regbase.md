# Regbase 编程范式（A5）

A5（Ascend 950）的寄存器级编程范式，用 `AscendC::MicroAPI` 命名空间。比 Membase（TQue/UB）更接近硬件寄存器，提供更精细的向量化控制。

## 与 Membase 的区别

| | Membase | Regbase |
|---|---|---|
| 数据载体 | LocalTensor（UB） | RegTensor（向量寄存器） |
| 缓冲管理 | AllocTensor/EnQue/DeQue/Free | 寄存器自动分配 |
| 掩码 | 函数参数 | MaskReg 寄存器 |
| 函数标记 | `__aicore__ inline` | `__simd_vf__ __aicore__` + `__VEC_SCOPE__` |
| 搬运 | DataCopy/DataCopyPad | MicroAPI::DataCopy + LoadDist/StoreDist |

## 核心类型（MicroAPI 命名空间）

```cpp
MicroAPI::RegTensor<T>      // 向量寄存器张量
MicroAPI::MaskReg           // 掩码寄存器
MicroAPI::AddrReg           // 地址偏移寄存器
```

## 代码骨架（trace 自 attention/posembedding arch35）

### 函数声明与作用域
```cpp
__simd_vf__ __aicore__ void MyCompute(ubuf float* inAddr, ubuf float* outAddr, int32_t count) {
    __VEC_SCOPE__ {   // Regbase 计算作用域
        MicroAPI::RegTensor<float> inReg, outReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        MicroAPI::DataCopy(inReg, inAddr);          // UB → 寄存器
        MicroAPI::Mul(outReg, inReg, inReg, preg);  // 寄存器级计算
        MicroAPI::DataCopy(outAddr, outReg, preg);  // 寄存器 → UB
    }
}
```

### 动态掩码（尾部处理）
```cpp
MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(remaining);   // 按剩余元素更新
MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<float>(loop, stride);
MicroAPI::DataCopy(reg, addr, offset);
MicroAPI::DataCopy(addr, reg, offset, preg);   // 带掩码存储
```

### 常用寄存器运算
```cpp
MicroAPI::Duplicate(reg, scalar, mask);   // 标量广播
MicroAPI::Arange(reg, start);             // 序列 [start, start+1, ...]
MicroAPI::Add/Sub/Mul/Div(dst, s1, s2, mask);
MicroAPI::Cast<DT, ST>(dst, src, mask);
MicroAPI::Compare<T, CMPMODE>(mask, s1, s2, pred);
MicroAPI::DataCopyGather(dst, base, indices, mask);   // 按索引 gather
```

### LoadDist / StoreDist（加载/存储分发模式）
```cpp
MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(reg, addr);  // BF16 解包升 FP32
MicroAPI::DataCopy<T, MicroAPI::LoadDist::DINTLV_B32>(reg, addr);       // De-Interleave（INTERLEAVE RoPE）
MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, reg);  // 4 元素打包（int8 量化）
```
| LoadDist | 用途 |
|---|---|
| DIST_NORM | 连续加载 |
| DIST_UNPACK_B16 | FP16/BF16 → FP32 |
| DIST_BRC_B32/B16 | 标量广播 |
| DIST_E2B_B32 | 标量→向量广播 |
| DINTLV_B32 | De-Interleave |

## 何时用 Regbase
- 需精细寄存器控制（复杂掩码、Gather/Scatter）
- A5 上想让计算留在寄存器不回 UB
- 需加载即转换（LoadDist 解包/拆分）

## 何时不用
- 简单连续数据处理 → Membase 更省事
- 矩阵乘 → 高阶 Matmul<> 或 Blaze

## A5 kernel 入口后缀
A5 Regbase kernel 入口文件名带 `_apt`（Ascend Parallel Template），如 `apply_rotary_pos_emb_apt.cpp`。详见 ascendc-development §Regbase（含 API 表）。