# Cube-Vector 融合专家技巧

## 核心思想

A5 的 CV 直连通路（UB2L1 / L0C2UB）让 Cube 和 Vector 在片上直接交换数据，**消除 GM 往返**。

## 切 K 累加优化

### A2A3 模式（低效）
```
Loop over K chunks:
  Cube(L0A, L0B) → L0C
  L0C → GM (Fixpipe)
  GM → UB (DataCopy)       ← 每轮都走 GM
  Vector(UB) 累加
```

### A5 模式（高效）
```
Loop over K chunks:
  Cube(L0A, L0B) → L0C
  L0C → UB (Fixpipe L0C2UB)  ← 直达，不走 GM
  Vector(UB) 累加
```

## 后处理融合管线

```
Cube MatMul → L0C → L0C2UB → Vector(Activation/Quant/RMSNorm) → UB2L1 → 下一轮 Cube
```

## 同步时序

CV 通路引入后，需显式梳理同步顺序：
```
MTE2(数据搬入 L1) → Wait MTE2
  → Cube(L1→L0A/L0B, 计算) → Wait Cube
    → Fixpipe(L0C→UB) → Wait Fixpipe    ← 关键：等 L0C 数据到 UB
      → Vector(UB 上后处理) → Wait Vector
        → DataCopy UB2L1 → Wait MTE1
          → 下一轮 Cube
```

每个箭头都有对应的事件同步，漏掉一个 → 数据可见性问题。

**关键使能接口**：
- `Fixpipe(LocalTensor<T>& dst, LocalTensor<U>& src, FixpipeParamsC310)` — L0C→UB
- `DataCopy(LocalTensor<T>& dst, LocalTensor<T>& src, Nd2NzParams)` — UB→L1 且随路 ND→NZ

## 适用场景判断

| 场景 | 是否用 CV 通路 |
|------|:---:|
| 单次 MatMul → 直接输出 | 不需要（Fixpipe 直接写 GM） |
| 切 K 累加 | **强烈推荐** L0C2UB |
| MatMul + 后处理(Act/Quant/Norm) | **推荐** 全链路 |
| 多轮 MatMul 间有 UB 计算 | **必用** 避免 GM 中转 |
