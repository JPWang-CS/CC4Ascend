# DataCopy / DataCopyPad 数据搬运 API

## DataCopy

**功能**：GM 与 Local Memory (UB/L1) 之间的数据搬运，支持连续和非连续搬运。

### 函数原型

```cpp
// 连续搬运
DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t dataSize);
DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t dataSize);

// 非连续搬运
DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params);
```

### DataCopyParams 结构体

```cpp
struct DataCopyParams {
    uint16_t blockCount = 0;   // 连续传输数据块个数 [1, 4095]
    uint16_t blockLen   = 0;   // 每个数据块长度，单位 DataBlock(32B)
    uint16_t srcStride  = 0;   // 源相邻数据块间隔
    uint16_t dstStride  = 0;   // 目的相邻数据块间隔
};
```

### 关键约束

| 约束 | 说明 |
|------|------|
| 32字节对齐 | UB 起始地址和数据长度必须 32B 对齐 |
| 搬运粒度 | 以 DataBlock（32B）为单位 |
| 性能 | 一次 DataCopy 远优于逐元素 for 循环（~65% 提升） |

### A5 变化

- **L1→GM 直接搬移被删除**：必须先搬到 UB 再搬出到 GM
- **ND→NZ 随路转换**：DataCopy + Nd2NzParams 一步完成格式转换

```cpp
Nd2NzParams nd2nzPara;
nd2nzPara.ndNum = 1;
nd2nzPara.nValue = srcN;
nd2nzPara.dValue = srcD;
nd2nzPara.srcDValue = srcDstride;
nd2nzPara.dstNzC0Stride = CeilDiv(srcN, 16) * 16;
nd2nzPara.dstNzNStride = 1;
DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);  // GM(ND) → L1(NZ) 一步完成
```

## DataCopyPad

**功能**：支持非对齐搬运（GM↔VECIN/VECOUT），可自动填充 padding。

适用于当数据长度不是 32B 对齐时的尾部处理场景。

## 增强 DataCopy（A5）

A5 新增 CO1→CO2 通路的随路计算，在搬运过程中完成简单算术操作。
