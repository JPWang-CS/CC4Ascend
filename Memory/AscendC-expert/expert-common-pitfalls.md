# 常见陷阱速查

## 类型推导陷阱

### Tensor vs Scalar 推导方向不同

```cpp
// Tensor-Tensor: f16 + f32 → f32 (向更宽类型提升)
aclnnAdd(f16_tensor, f32_tensor)

// Tensor-Scalar: f16 Tensor + f32 Scalar → f16 (Scalar 向 Tensor 靠拢)
aclnnAdds(f16_tensor, f32_scalar)  // 结果仍是 f16!
```

**后果**：如果 Scalar 值超出 f16 表示范围，静默截断。

### broadcast 限制（特殊类型）

COMPLEX64/128, DOUBLE, INT16, UINT16, UINT64 的 broadcast 有额外限制：合并后维度 < 6。其他类型无此限制。

## A5 迁移陷阱

### 1. L1→GM 被删
`DataCopy(GM, L1)` → A5 报错。
修: `DataCopy(UB, L1)` → `DataCopy(GM, UB)`

### 2. GM→L0A 被删
直连不再可用。
修: `DataCopy(L1, GM)` → `DataCopy(L0A, L1)`

### 3. int4_t Cube 不支持
A2A3 的 `int4_t` Cube 代码 → A5 编译不通过。
修: 改用 `int8_t`，更新量化解算。

### 4. 4:2 稀疏不支持
A2A3 的 4:2 稀疏优化 → A5 无效。
修: 改为稠密或其他稀疏策略。

## 命名陷阱

- `_apt.cpp` = A5 (Ascend Parallel Template)，不是"apt"英文
- `arch35` = A5 (950)，不是"版本 3.5"
- `arch22` = A2A3 (910B/910C)，不是"版本 2.2"
- `arch31` = 310P，不是"版本 3.1"

## aclnn 调用陷阱

1. 第二段接口**不可重复调用**（aclOpExecutor 在调用后自动释放）
2. workspace 即使为 0 也要传有效指针
3. `aclrtSynchronizeStream` 后才可以读输出
4. `aclCreateTensor` 完后需要 `aclDestroyTensor`
